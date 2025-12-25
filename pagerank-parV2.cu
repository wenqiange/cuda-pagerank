#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Necesario para clock()

#include "params.h"   // NB_NODES, MAX_ITER, LAMBDA, EPSILON
#include "utils.cu"

// --- VARIABLES GLOBALES PARA TIEMPOS (Añadidas) ---
time_t total_start, total_end;
double total_time = 0.0;
time_t load_start, load_end;
double load_time = 0.0;
time_t sparse_start, sparse_end;
double sparse_time = 0.0;
time_t pr_start, pr_end;
double pr_time = 0.0;
time_t res_start, res_end;
double res_time = 0.0;
// accum_mv_time se deja a 0.0 porque medirlo dentro del while en GPU mata el rendimiento
double accum_mv_time = 0.0; 

// Kernel para inicializar p
__global__ void init_p(double *p, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) p[i] = 1.0 / (double)N;
}

// Kernel para inicializar p_new con (1 - LAMBDA)/N
__global__ void init_p_new(double *p_new, int N, double base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) p_new[i] = base;
}

// Kernel para calcular suma de dangling nodes parcial
__global__ void dangling_partial(const double *p, const int *outdeg,
                                 double *partial, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    double val = 0.0;
    if (i < N && outdeg[i] == 0)
        val = p[i];

    sdata[tid] = val;
    __syncthreads();

    // reducción en el bloque
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

// Kernel para un paso de PageRank sobre las aristas
__global__ void pagerank_step(const int *row_ptr, const int *col_idx,
                              const int *outdeg, const double *p,
                              double *p_new, int N, double lambda) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N && outdeg[u] > 0) {
        double contrib = lambda * p[u] / (double)outdeg[u];
        for (int k = row_ptr[u]; k < row_ptr[u + 1]; ++k) {
            int v = col_idx[k];
            atomicAdd(&p_new[v], contrib);
        }
    }
}

// Kernel para sumar el término de dangling a todos los nodos
__global__ void add_dangling(double *p_new, int N, double add_dang) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) p_new[i] += add_dang;
}

// Kernel para calcular diferencia L1 y actualizar p
__global__ void update_and_diff(const double *p_new, double *p,
                                double *partial, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    double local = 0.0;
    if (i < N) {
        double diff = fabs(p_new[i] - p[i]);
        p[i] = p_new[i];
        local = diff;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

// ----------------------------------------------------------------------
// PageRank en CUDA (host)
// ----------------------------------------------------------------------
void pagerank_cuda(int *h_row_ptr, int *h_col_idx, int *h_outdeg, double *h_p) {
    int N = NB_NODES;

    int *d_row_ptr, *d_col_idx, *d_outdeg;
    double *d_p, *d_p_new;
    double *d_partial;      // para reducciones (dangling y diff)

    int threads = 256;
    int blocksN = (N + threads - 1) / threads;
    int partial_size = blocksN;

    // Reservar memoria
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, h_row_ptr[N] * sizeof(int));
    cudaMalloc(&d_outdeg, N * sizeof(int));
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_p_new, N * sizeof(double));
    cudaMalloc(&d_partial, partial_size * sizeof(double));

    // Copiar datos H->D
    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, h_row_ptr[N] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outdeg, h_outdeg, N * sizeof(int), cudaMemcpyHostToDevice);

    // Inicializar p
    init_p<<<blocksN, threads>>>(d_p, N);
    cudaDeviceSynchronize();

    double base = (1.0 - LAMBDA) / (double)N;
    int iter = 0;

    std::vector<double> h_partial(partial_size);

    while (iter < MAX_ITER) {
        // 1. Inicializar p_new
        init_p_new<<<blocksN, threads>>>(d_p_new, N, base);

        // 2. Dangling partial
        dangling_partial<<<blocksN, threads, threads * sizeof(double)>>>
            (d_p, d_outdeg, d_partial, N);

        // Copiar parcial al host y sumar (parte secuencial pequeña)
        cudaMemcpy(h_partial.data(), d_partial,
                   partial_size * sizeof(double), cudaMemcpyDeviceToHost);

        double dangling_sum = 0.0;
        for (int i = 0; i < partial_size; ++i)
            dangling_sum += h_partial[i];

        double add_dang = LAMBDA * dangling_sum / (double)N;

        // 3. PageRank Step (PUSH)
        pagerank_step<<<blocksN, threads>>>(
            d_row_ptr, d_col_idx, d_outdeg, d_p, d_p_new, N, LAMBDA);

        // 4. Add Dangling
        add_dangling<<<blocksN, threads>>>(d_p_new, N, add_dang);
        
        // Sincronización necesaria antes de update_and_diff para asegurar que p_new está listo
        cudaDeviceSynchronize(); 

        // 5. Update and Diff
        update_and_diff<<<blocksN, threads, threads * sizeof(double)>>>(
            d_p_new, d_p, d_partial, N);

        cudaMemcpy(h_partial.data(), d_partial,
                   partial_size * sizeof(double), cudaMemcpyDeviceToHost);

        double diff = 0.0;
        for (int i = 0; i < partial_size; ++i)
            diff += h_partial[i];

        if (diff < EPSILON) break;
        iter++;
    }

    printf("Convergencia en %d iteraciones (CUDA)\n", iter);

    // Copiar resultado final a host
    cudaMemcpy(h_p, d_p, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_outdeg);
    cudaFree(d_p);
    cudaFree(d_p_new);
    cudaFree(d_partial);
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main() {
    printf("CUDA PageRank (Instrumented)\n");

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
    
    // --- CORRECCIÓN AQUÍ: Llamar a pagerank_cuda, no pagerank ---
    pagerank_cuda(row_ptr, col_idx, outdeg, p);
    
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
    printf("PageRank time (CUDA): %.6f seconds\n", pr_time);
    printf("Results time: %.6f seconds\n", res_time);
    // accum_mv_time será 0, ya que no se mide dentro del kernel
    printf("Accumulated matrix-vector multiplication time: %.6f seconds (Not measured in GPU)\n", accum_mv_time);

    return 0;
}