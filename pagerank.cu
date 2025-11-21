#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4206784   // Cambiar por el número real de nodos
#define LAMBDA 0.9
#define EPSILON 1e-6
#define MAX_ITER 10
#define ELEMS_A_MOSTRAR 10

int main() {
    FILE *file = fopen("enwiki-2013.txt", "r");
    if (!file) {
        printf("No se pudo abrir el fichero\n");
        return 1;
    }

    // 1. Inicializar matriz de adyacencia A y vector de out-degree
    int **A = (int**)malloc(N * sizeof(int*));
    int *outDegree = (int*)calloc(N, sizeof(int));

    for (int i = 0; i < N; i++) {
        A[i] = (int*)calloc(N, sizeof(int));
    }

    int from, to;
    while (fscanf(file, "%d %d", &from, &to) == 2) {
        if (from >= N || to >= N) continue; // seguridad
        A[to][from] = 1;    // transpuesta para PageRank
        outDegree[from]++;
    }
    fclose(file);

    // 2. Crear matriz estocástica M
    double **M = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        M[i] = (double*)calloc(N, sizeof(double));
        for (int j = 0; j < N; j++) {
            if (outDegree[j] > 0)
                M[i][j] = (double)A[i][j] / outDegree[j];
            else
                M[i][j] = 0.0; // dangling nodes
        }
    }

    // 3. Inicializar vector PageRank p
    double *p = (double*)malloc(N * sizeof(double));
    double *p_new = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        p[i] = 1.0 / N;
    }

    // 4. Iteración de PageRank
    int iter = 0;
    double diff;
    do {
        for (int i = 0; i < N; i++) {
            p_new[i] = 0.0;
            for (int j = 0; j < N; j++) {
                p_new[i] += LAMBDA * M[i][j] * p[j];
            }
            p_new[i] += (1.0 - LAMBDA) / N; // teleport
        }

        // Calcular diferencia para convergencia
        diff = 0.0;
        for (int i = 0; i < N; i++) {
            diff += fabs(p_new[i] - p[i]);
            p[i] = p_new[i];
        }

        iter++;
    } while (diff > EPSILON && iter < MAX_ITER);

    printf("PageRank calculado en %d iteraciones\n", iter);
    for (int i = 0; i < ELEMS_A_MOSTRAR; i++) {
        printf("p[%d] = %f\n", i, p[i]);
    }

    // Liberar memoria
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(M[i]);
    }
    free(A);
    free(M);
    free(p);
    free(p_new);

    return 0;
}
