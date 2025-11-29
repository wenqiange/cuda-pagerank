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
    FILE *fgraph, *fmap;
    load_files(&fgraph, &fmap);

    Vec *adj = (Vec*) malloc(N * sizeof(Vec));
    int *outdeg = (int*) calloc(N, sizeof(int));
    load_graph(fgraph, adj, outdeg);

    std::map<int, std::string> id_to_title;
    load_map(fmap, id_to_title);

    double *p = (double*) malloc(N * sizeof(double));
    pagerank(adj, outdeg, p);
 
    print_results(p, id_to_title);

    free(adj);
    free(outdeg);
    free(p);

    return 0;
}
