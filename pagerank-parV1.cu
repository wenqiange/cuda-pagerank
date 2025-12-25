#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <time.h> // Necesario para clock()

#include "params.h"
#include "utils.cu"

// ----------------------------------------------------------------------
// CUDA helper: AtomicAdd para Double (Compatible con Pascal y anteriores)
// ----------------------------------------------------------------------
__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// ----------------------------------------------------------------------
// KERNEL: Push PageRank
// ----------------------------------------------------------------------
__global__ void pr_push_kernel(const int *row_ptr, const int *col_idx,
                               const int *outdeg, const double *p,
                               double *p_new, int N) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;

    int deg = outdeg[u];
    if (deg <= 0) return;

    double contrib = LAMBDA * p[u] / (double)deg;
    for (int k = row_ptr[u]; k < row_ptr[u + 1]; ++k) {
        int v = col_idx[k];
        atomicAdd_double(&p_new[v], contrib);
    }
}

// ----------------------------------------------------------------------
// PageRank Host (Llamada híbrida)
// Modificado para recibir puntero a accum_mv_time
// ----------------------------------------------------------------------
void pagerank(int *row_ptr, int *col_idx, int *outdeg, double *p, double *accum_mv_time) {
    double *p_new = (double*) malloc(NB_NODES * sizeof(double));

    for (int i = 0; i < NB_NODES; i++)
        p[i] = 1.0 / NB_NODES;

    // --- Reservas en GPU ---
    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    int *d_outdeg  = nullptr;
    double *d_p    = nullptr;
    double *d_p_new = nullptr;

    const int N = NB_NODES;
    const int nnz = row_ptr[NB_NODES];

    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_outdeg, N * sizeof(int));
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_p_new, N * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outdeg, outdeg, N * sizeof(int), cudaMemcpyHostToDevice);

    int iter = 0;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // Inicializamos el acumulador a 0
    *accum_mv_time = 0.0;

    while (iter < MAX_ITER) {
        for (int i = 0; i < NB_NODES; i++)
            p_new[i] = (1 - LAMBDA) / NB_NODES;

        double dangling_sum = 0.0;
        for (int u = 0; u < NB_NODES; u++)
            if (outdeg[u] == 0)
                dangling_sum += p[u];

        double add_dang = LAMBDA * dangling_sum / NB_NODES;

        // ------------------------------
        // INICIO MEDICIÓN PARTE GPU (Matrix-Vector)
        // ------------------------------
        clock_t t_mv_start = clock();

        cudaMemcpy(d_p, p, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p_new, p_new, N * sizeof(double), cudaMemcpyHostToDevice);

        pr_push_kernel<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_outdeg, d_p, d_p_new, N);
        
        // Importante: Synchronize para asegurar que el tiempo incluye la ejecución del kernel
        cudaDeviceSynchronize(); 

        cudaMemcpy(p_new, d_p_new, N * sizeof(double), cudaMemcpyDeviceToHost);

        clock_t t_mv_end = clock();
        *accum_mv_time += (double)(t_mv_end - t_mv_start) / CLOCKS_PER_SEC;
        // ------------------------------
        // FIN MEDICIÓN PARTE GPU
        // ------------------------------

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

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_outdeg);
    cudaFree(d_p);
    cudaFree(d_p_new);

    free(p_new);
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main() {
    printf("CUDA PageRank (pagerank-tiempos.cu)\n");

    // 1. Declaración de variables de tiempo (Corrección principal)
    clock_t total_start, total_end;
    clock_t load_start, load_end;
    clock_t sparse_start, sparse_end;
    clock_t pr_start, pr_end;
    clock_t res_start, res_end;
    
    double total_time, load_time, sparse_time, pr_time, res_time;
    double accum_mv_time = 0.0; // Variable para acumular tiempo GPU

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
    
    // Cálculo inmediato para evitar error de variable no usada si fuera el caso
    sparse_time = (double)(sparse_end - sparse_start) / CLOCKS_PER_SEC;

    pr_start = clock();
    double *p = (double*) malloc(NB_NODES * sizeof(double));
    
    // Pasamos la dirección de accum_mv_time
    pagerank(row_ptr, col_idx, outdeg, p, &accum_mv_time); 
    
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
    
    // Ahora esta variable sí tiene el valor correcto calculado dentro de pagerank
    printf("Accumulated matrix-vector multiplication time: %.6f seconds\n", accum_mv_time);

    return 0;
}