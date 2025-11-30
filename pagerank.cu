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
        for (int u = 0; u < NB_NODES; u++)
            for (int k = row_ptr[u]; k < row_ptr[u + 1]; k++) {
                int v = col_idx[k];
                p_new[v] += LAMBDA * p[u] / outdeg[u];
            }

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
    FILE *fgraph, *fmap;
    load_files(&fgraph, &fmap);

    Vec *adj = (Vec*) malloc(NB_NODES * sizeof(Vec));
    int *outdeg = (int*) calloc(NB_NODES, sizeof(int));
    load_graph(fgraph, adj, outdeg);
    std::map<int, std::string> id_to_title;
    load_map(fmap, id_to_title);

    // convertir a formato CSR
    int *row_ptr, *col_idx;
    convert_to_csr(adj, &row_ptr, &col_idx);

    double *p = (double*) malloc(NB_NODES * sizeof(double));
    pagerank(row_ptr, col_idx, outdeg, p);
 
    print_results(p, id_to_title);

    free(row_ptr);
    free(col_idx);
    free(adj);
    free(outdeg);
    free(p);

    return 0;
}
