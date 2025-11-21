#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4206784
#define LAMBDA 0.90
#define EPSILON 1e-6
#define MAX_ITER 100
#define ELEMS_A_MOSTRAR 10

// ----------------------------------------------------------------------
// Estructura sparse: lista de adyacencia para grafo dirigido
// adj[u][k] = v  (enlace u -> v)
// ----------------------------------------------------------------------
typedef struct {
    int *data;
    int size;
    int cap;
} Vec;

static inline void vec_init(Vec *v) {
    v->size = 0;
    v->cap = 4;
    v->data = (int*) malloc(v->cap * sizeof(int));
}

static inline void vec_push(Vec *v, int x) {
    if (v->size == v->cap) {
        v->cap *= 2;
        v->data = (int*) realloc(v->data, v->cap * sizeof(int));
    }
    v->data[v->size++] = x;
}

// ----------------------------------------------------------------------
// PageRank usando estructura sparse
// ----------------------------------------------------------------------
void pagerank(Vec *adj, int *outdeg, double *p) {
    double *p_new = (double*) malloc(N * sizeof(double));

    for (int i = 0; i < N; i++)
        p[i] = 1.0 / N;

    int iter = 0;

    while (iter < MAX_ITER) {

        for (int i = 0; i < N; i++)
            p_new[i] = (1 - LAMBDA) / N;

        double dangling_sum = 0.0;
        for (int u = 0; u < N; u++)
            if (outdeg[u] == 0)
                dangling_sum += p[u];

        double add_dang = LAMBDA * dangling_sum / N;

        for (int u = 0; u < N; u++)
            for (int k = 0; k < adj[u].size; k++) {
                int v = adj[u].data[k];
                p_new[v] += LAMBDA * (p[u] / outdeg[u]);
            }

        for (int i = 0; i < N; i++)
            p_new[i] += add_dang;

        double diff = 0.0;
        for (int i = 0; i < N; i++) {
            diff += fabs(p_new[i] - p[i]);
            p[i] = p_new[i];
        }

        if (diff < EPSILON) break;
        iter++;
    }

    printf("Convergencia en %d iteraciones\n", iter);
    free(p_new);
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main() {

    FILE *file = fopen("enwiki-2013.txt", "r");
    if (!file) {
        printf("No se pudo abrir el fichero\n");
        return 1;
    }

    Vec *adj = (Vec*) malloc(N * sizeof(Vec));
    int *outdeg = (int*) calloc(N, sizeof(int));

    for (int i = 0; i < N; i++)
        vec_init(&adj[i]);

    int u, v;
    char line[128];
	while (fgets(line, sizeof(line), file)) {

		if (line[0] == '#') continue;  // saltar comentarios

		if (sscanf(line, "%d %d", &u, &v) == 2) {
			if (u < N && v < N) {
				vec_push(&adj[u], v);
				outdeg[u]++;
			}
		}
	}

    double *p = (double*) malloc(N * sizeof(double));
    pagerank(adj, outdeg, p);

    // Mostrar los 10 nodos con mayor PageRank
    int idx[ELEMS_A_MOSTRAR];
    double val[ELEMS_A_MOSTRAR];
    for (int i = 0; i < ELEMS_A_MOSTRAR; i++) {
        idx[i] = -1;
        val[i] = -1.0;
    }
    for (int i = 0; i < N; i++) {
        // Buscar si p[i] es mayor que alguno de los 10 actuales
        int min_idx = 0;
        for (int j = 1; j < ELEMS_A_MOSTRAR; j++)
            if (val[j] < val[min_idx]) min_idx = j;
        if (p[i] > val[min_idx]) {
            val[min_idx] = p[i];
            idx[min_idx] = i;
        }
    }
    // Mostrar los resultados
    //TODO: que se muestren ordenados
    for (int i = 0; i < ELEMS_A_MOSTRAR; i++)
        printf("p[%d] = %.10f\n", idx[i], val[i]);

    for (int i = 0; i < N; i++)
        free(adj[i].data);

    free(adj);
    free(outdeg);
    free(p);

    return 0;
}
