#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "params.h"   // NB_NODES, MAX_ITER, LAMBDA, EPSILON
#include "utils.cu"

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

    // Número máximo de bloques para reducciones (usamos blocksN)
    int partial_size = blocksN;

    cudaError_t err;
    err = cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    if (err != cudaSuccess) { printf("Error cudaMalloc d_row_ptr\n"); exit(1); }
    err = cudaMalloc(&d_col_idx, h_row_ptr[N] * sizeof(int)); // nº de aristas = row_ptr[N]
    if (err != cudaSuccess) { printf("Error cudaMalloc d_col_idx\n"); exit(1); }
    err = cudaMalloc(&d_outdeg, N * sizeof(int));
    if (err != cudaSuccess) { printf("Error cudaMalloc d_outdeg\n"); exit(1); }
    err = cudaMalloc(&d_p, N * sizeof(double));
    if (err != cudaSuccess) { printf("Error cudaMalloc d_p\n"); exit(1); }
    err = cudaMalloc(&d_p_new, N * sizeof(double));
    if (err != cudaSuccess) { printf("Error cudaMalloc d_p_new\n"); exit(1); }
    err = cudaMalloc(&d_partial, partial_size * sizeof(double));
    if (err != cudaSuccess) { printf("Error cudaMalloc d_partial\n"); exit(1); }

    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, h_row_ptr[N] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outdeg, h_outdeg, N * sizeof(int), cudaMemcpyHostToDevice);

    // Inicializar p
    init_p<<<blocksN, threads>>>(d_p, N);
    cudaDeviceSynchronize();

    double base = (1.0 - LAMBDA) / (double)N;
    int iter = 0;

    while (iter < MAX_ITER) {
        // p_new = (1 - lambda)/N
        init_p_new<<<blocksN, threads>>>(d_p_new, N, base);

        // suma parcial de dangling nodes
        dangling_partial<<<blocksN, threads, threads * sizeof(double)>>>
            (d_p, d_outdeg, d_partial, N);

        // copiar parciales a host y sumar
        std::vector<double> h_partial(partial_size);
        cudaMemcpy(h_partial.data(), d_partial,
                   partial_size * sizeof(double), cudaMemcpyDeviceToHost);

        double dangling_sum = 0.0;
        for (int i = 0; i < partial_size; ++i)
            dangling_sum += h_partial[i];

        double add_dang = LAMBDA * dangling_sum / (double)N;

        // contribución a partir de enlaces salientes
        pagerank_step<<<blocksN, threads>>>(
            d_row_ptr, d_col_idx, d_outdeg, d_p, d_p_new, N, LAMBDA);

        // añadir término de dangling
        add_dangling<<<blocksN, threads>>>(d_p_new, N, add_dang);
        cudaDeviceSynchronize();

        // diff = ||p_new - p||_1 y actualizar p
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

    printf("Convergencia en %d iteraciones\n", iter);

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
    printf("CUDA PageRank (pagerank.cu)\n");

    FILE *fgraph, *fmap;
    load_files(&fgraph, &fmap);

    AdjMat adj(NB_NODES);
    int *outdeg = (int*) calloc(NB_NODES, sizeof(int));
    load_graph(fgraph, adj, outdeg);
    std::map<int, std::string> id_to_title;
    load_map(fmap, id_to_title);

    // convert to CSR format
    int *row_ptr, *col_idx;
    convert_to_csr(adj, &row_ptr, &col_idx);

    double *p = (double*) malloc(NB_NODES * sizeof(double));
    pagerank_cuda(row_ptr, col_idx, outdeg, p);
 
    print_results(p, id_to_title);

    free(row_ptr);
    free(col_idx);
    free(outdeg);
    free(p);

    return 0;
}
