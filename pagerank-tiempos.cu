#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>

#include "params.h"
#include "utils.cu"

#ifndef SIZE
#define SIZE 32
#endif 

#ifndef PINNED
#define PINNED 0
#endif

// Time measurement
#include <time.h>
time_t total_start, total_end;  // para medir tiempo total
double total_time = 0.0;        // tiempo total
time_t mv_start, mv_end;        // para medir tiempo de multiplicación matriz-vector
double accum_mv_time = 0.0;     // tiempo acumulado de multiplicaciones matriz-vector
time_t pr_start, pr_end;        // para medir tiempo de PageRank
double pr_time = 0.0;           // tiempo de PageRank
time_t load_start, load_end;    // para medir tiempo de carga de datos
double load_time = 0.0;         // tiempo de carga de datos
time_t res_start, res_end;      // para medir tiempo de resultados
double res_time = 0.0;          // tiempo de resultados
time_t sparse_start, sparse_end; // para medir tiempo de conversión a CSR
double sparse_time = 0.0;       // tiempo de conversión a CSR


// ----------------------------------------------------------------------
// PageRank usando estructura sparse (CSR)
// ----------------------------------------------------------------------
void pagerank(int *row_ptr, int *col_idx, int *outdeg, double *p) {
    double *p_new = (double*) malloc(NB_NODES * sizeof(double));

    for (int i = 0; i < NB_NODES; i++)
        p[i] = 1.0 / NB_NODES;

    int iter = 0;

    while (iter < MAX_ITER) {

        for (int i = 0; i < NB_NODES; i++)
            p_new[i] = (1 - LAMBDA) / NB_NODES;

        double dangling_sum = 0.0;
        for (int u = 0; u < NB_NODES; u++)
            if (outdeg[u] == 0)
                dangling_sum += p[u];

        double add_dang = LAMBDA * dangling_sum / NB_NODES;
        mv_start = clock();
        for (int u = 0; u < NB_NODES; u++)
            for (int k = row_ptr[u]; k < row_ptr[u + 1]; k++) {
                int v = col_idx[k];
                p_new[v] += LAMBDA * p[u] / outdeg[u];
            }
        mv_end = clock();
        accum_mv_time += (double)(mv_end - mv_start) / CLOCKS_PER_SEC;

        for (int i = 0; i < NB_NODES; i++)
            p_new[i] += add_dang;

        double diff = 0.0;
        for (int i = 0; i < NB_NODES; i++) {
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
    total_start = clock();
    load_start = clock();
    FILE *fgraph, *fmap;
    load_files(&fgraph, &fmap);

    Vec *adj = (Vec*) malloc(NB_NODES * sizeof(Vec));
    int *outdeg = (int*) calloc(NB_NODES, sizeof(int));
    load_graph(fgraph, adj, outdeg);
    std::map<int, std::string> id_to_title;
    load_map(fmap, id_to_title);
    load_end = clock();

    sparse_start = clock();
    int *row_ptr, *col_idx;
    convert_to_csr(adj, &row_ptr, &col_idx);
    sparse_end = clock();
    sparse_time = (double)(sparse_end - sparse_start) / CLOCKS_PER_SEC;

    pr_start = clock();
    double *p = (double*) malloc(NB_NODES * sizeof(double));
    pagerank(row_ptr, col_idx, outdeg, p);
    pr_end = clock();

    res_start = clock();
    print_results(p, id_to_title);
    res_end = clock();

    free(row_ptr);
    free(col_idx);
    free(adj);
    free(outdeg);
    free(p);
    total_end = clock();

    // Print timing results
    total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    load_time = (double)(load_end - load_start) / CLOCKS_PER_SEC;
    pr_time = (double)(pr_end - pr_start) / CLOCKS_PER_SEC;
    res_time = (double)(res_end - res_start) / CLOCKS_PER_SEC;
    printf("Tiempo total: %.6f segundos\n", total_time);
    printf("Tiempo de carga de datos: %.6f segundos\n", load_time);
    printf("Tiempo de conversión a CSR: %.6f segundos\n", sparse_time);
    printf("Tiempo de PageRank: %.6f segundos\n", pr_time);
    printf("Tiempo de resultados: %.6f segundos\n", res_time);
    printf("Tiempo acumulado de multiplicaciones matriz-vector: %.6f segundos\n", accum_mv_time);

    return 0;
}
