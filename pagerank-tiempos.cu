#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

static inline void *safe_malloc(size_t bytes) {
    void *ptr = malloc(bytes);
    if (!ptr) {
        fprintf(stderr, "Error: sin memoria (%zu bytes)\n", bytes);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static inline void *safe_calloc(size_t count, size_t bytes) {
    void *ptr = calloc(count, bytes);
    if (!ptr) {
        fprintf(stderr, "Error: sin memoria (%zu bytes)\n", count * bytes);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static inline void *safe_realloc(void *ptr, size_t bytes) {
    void *new_ptr = realloc(ptr, bytes);
    if (!new_ptr) {
        fprintf(stderr, "Error: sin memoria (%zu bytes)\n", bytes);
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

static inline void vec_init(Vec *v) {
    v->size = 0;
    v->cap = 0;
    v->data = NULL;
}

static inline void vec_push(Vec *v, int x) {
    if (v->cap == 0) {
        v->cap = 4;
        v->data = (int*) safe_malloc(v->cap * sizeof(int));
    } else if (v->size == v->cap) {
        v->cap *= 2;
        v->data = (int*) safe_realloc(v->data, v->cap * sizeof(int));
    }
    v->data[v->size++] = x;
}

double pagerank_time = 0.0;
double matrixvec_time = 0.0;

// ----------------------------------------------------------------------
// PageRank usando estructura sparse
// ----------------------------------------------------------------------
void pagerank(Vec *adj, int *outdeg, double *p) {
    double *p_new = (double*) safe_malloc(N * sizeof(double));

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

        clock_t start_time_mv, end_time_mv;
        start_time_mv = clock();
        for (int u = 0; u < N; u++)
            for (int k = 0; k < adj[u].size; k++) {
                int v = adj[u].data[k];
                p_new[v] += LAMBDA * (p[u] / outdeg[u]);
            }
        end_time_mv = clock();
        matrixvec_time += (double)(end_time_mv - start_time_mv) / CLOCKS_PER_SEC;

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

    Vec *adj = (Vec*) safe_malloc(N * sizeof(Vec));
    int *outdeg = (int*) safe_calloc(N, sizeof(int));

    for (int i = 0; i < N; i++)
        vec_init(&adj[i]);

    int u, v;
    char line[128];
    size_t invalid_edges = 0;
	while (fgets(line, sizeof(line), file)) {

		if (line[0] == '#') continue;  // saltar comentarios

		if (sscanf(line, "%d %d", &u, &v) == 2) {
			if (u >= 0 && v >= 0 && u < N && v < N) {
				vec_push(&adj[u], v);
				outdeg[u]++;
			} else {
				invalid_edges++;
			}
		}
	}

    fclose(file);

    if (invalid_edges > 0)
        fprintf(stderr, "Advertencia: se descartaron %zu aristas fuera de rango [0,%d)\n", invalid_edges, N);

    double *p = (double*) safe_malloc(N * sizeof(double));
    clock_t start_time_pr, end_time_pr;
    matrixvec_time = 0.0;
    pagerank_time = 0.0;
    start_time_pr = clock();
    pagerank(adj, outdeg, p);
    end_time_pr = clock();
    pagerank_time = (double)(end_time_pr - start_time_pr) / CLOCKS_PER_SEC;
    printf("Tiempo total PageRank: %.6f segundos\n", pagerank_time);
    printf("Tiempo total Matrix-Vector: %.6f segundos\n", matrixvec_time);
    
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
