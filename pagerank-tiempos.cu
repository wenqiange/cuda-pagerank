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

// Time measurement
#include <time.h>
time_t total_start, total_end;   // for measuring total time
double total_time = 0.0;         // total time
time_t mv_start, mv_end;         // for measuring matrix-vector multiplication time
double accum_mv_time = 0.0;      // accumulated matrix-vector multiplication time
time_t pr_start, pr_end;         // for measuring PageRank time
double pr_time = 0.0;            // PageRank time
time_t load_start, load_end;     // for measuring data loading time
double load_time = 0.0;          // data loading time
time_t res_start, res_end;       // for measuring results time
double res_time = 0.0;           // results time
time_t sparse_start, sparse_end; // for measuring CSR conversion time
double sparse_time = 0.0;        // CSR conversion time


// ----------------------------------------------------------------------
// PageRank using sparse structure (CSR)
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

    printf("Convergence in %d iterations\n", iter);
    free(p_new);
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main() {
    printf("CUDA PageRank (pagerank-tiempos.cu)\n");

    total_start = clock();
    load_start = clock();
    FILE *fgraph, *fmap;
    load_files(&fgraph, &fmap);

    AdjMat adj(NB_NODES);
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
    free(outdeg);
    free(p);
    total_end = clock();

    // Time reporting
    total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    load_time = (double)(load_end - load_start) / CLOCKS_PER_SEC;
    pr_time = (double)(pr_end - pr_start) / CLOCKS_PER_SEC;
    res_time = (double)(res_end - res_start) / CLOCKS_PER_SEC;
    printf("Total time: %.6f seconds\n", total_time);
    printf("Data loading time: %.6f seconds\n", load_time);
    printf("CSR conversion time: %.6f seconds\n", sparse_time);
    printf("PageRank time: %.6f seconds\n", pr_time);
    printf("Results time: %.6f seconds\n", res_time);
    printf("Accumulated matrix-vector multiplication time: %.6f seconds\n", accum_mv_time);

    return 0;
}
