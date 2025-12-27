#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "params.h"
#include "utils.cu"

// --- VARIABLES GLOBALES PARA TIEMPOS ---
time_t total_start, total_end;
double total_time = 0.0;
time_t load_start, load_end;
double load_time = 0.0;
time_t sparse_start, sparse_end;
double sparse_time = 0.0;
time_t mapping_start, mapping_end; // Nuevo: Tiempo para mapeo de aristas
double mapping_time = 0.0;
time_t pr_start, pr_end;
double pr_time = 0.0;
time_t res_start, res_end;
double res_time = 0.0;

// --- KERNELS ---

__global__ void init_p(double *p, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) p[i] = 1.0 / (double)N;
}

__global__ void init_p_new(double *p_new, int N, double base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) p_new[i] = base;
}

__global__ void dangling_partial(const double *p, const int *outdeg, double *partial, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    double val = 0.0;
    if (i < N && outdeg[i] == 0) val = p[i];

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void add_dangling(double *p_new, int N, double add_dang) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) p_new[i] += add_dang;
}

__global__ void update_and_diff(const double *p_new, double *p, double *partial, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    double local = 0.0;
    if (i < N) {
        double diff = fabs(p_new[i] - p[i]);
        p[i] = p_new[i];
        local = diff;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// --- FUNCIÓN DE MAPEO ---
void create_edge_mapping(int N, int num_edges, const int *h_row_ptr, int *h_edge_src) {
    for (int u = 0; u < N; ++u) {
        for (int k = h_row_ptr[u]; k < h_row_ptr[u + 1]; ++k) {
            h_edge_src[k] = u;
        }
    }
}

__global__ void pagerank_edge_step(const int *edge_src, const int *col_idx,
                                   const int *outdeg, const double *p,
                                   double *p_new, int num_edges, double lambda) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < num_edges) {
        int u = edge_src[k];    // Nodo origen
        int v = col_idx[k];     // Nodo destino
        
        // Cada hilo procesa exactamente una arista
        double contrib = lambda * p[u] / (double)outdeg[u];
        atomicAdd(&p_new[v], contrib);
    }
}

void pagerank_cuda_edge(int *h_row_ptr, int *h_col_idx, int *h_edge_src, int *h_outdeg, double *h_p) {
    int N = NB_NODES;
    int num_edges = h_row_ptr[N];

    int *d_row_ptr, *d_col_idx, *d_edge_src, *d_outdeg;
    double *d_p, *d_p_new, *d_partial, *d_contribs;

    int threads = 256;
    int blocksN = (N + threads - 1) / threads;
    int blocksEdges = (num_edges + threads - 1) / threads;
    int partial_size = blocksN;

    // Reservar memoria
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, num_edges * sizeof(int));
    cudaMalloc(&d_edge_src, num_edges * sizeof(int));
    cudaMalloc(&d_outdeg, N * sizeof(int));
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_p_new, N * sizeof(double));
    cudaMalloc(&d_partial, partial_size * sizeof(double));
    cudaMalloc(&d_contribs, N * sizeof(double)); // Espacio para pre-cálculo

    // Copias iniciales
    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_src, h_edge_src, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outdeg, h_outdeg, N * sizeof(int), cudaMemcpyHostToDevice);

    init_p<<<blocksN, threads>>>(d_p, N);

    double base = (1.0 - LAMBDA) / (double)N;
    int iter = 0;

    // Envolvemos punteros para usar con Thrust (Reducción en GPU)
    thrust::device_ptr<double> dev_ptr_partial(d_partial);

    while (iter < MAX_ITER) {
        init_p_new<<<blocksN, threads>>>(d_p_new, N, base);

        dangling_partial<<<blocksN, threads, threads * sizeof(double)>>>(d_p, d_outdeg, d_partial, N);
        double dangling_sum = thrust::reduce(dev_ptr_partial, dev_ptr_partial + partial_size);
        double add_dang = LAMBDA * dangling_sum / (double)N;

        pagerank_edge_step<<<blocksEdges, threads>>>(d_edge_src, d_col_idx, d_outdeg, d_p, d_p_new, num_edges, LAMBDA);

        add_dangling<<<blocksN, threads>>>(d_p_new, N, add_dang);
        
        update_and_diff<<<blocksN, threads, threads * sizeof(double)>>>(d_p_new, d_p, d_partial, N);
        double diff = thrust::reduce(dev_ptr_partial, dev_ptr_partial + partial_size);

        if (diff < EPSILON) break;
        iter++;
    }

    printf("Convergencia en %d iteraciones\n", iter);
    cudaMemcpy(h_p, d_p, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Limpieza
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_edge_src);
    cudaFree(d_outdeg); cudaFree(d_p); cudaFree(d_p_new); 
    cudaFree(d_partial); cudaFree(d_contribs);
}

int main() {
    printf("CUDA PageRank (pagerank-parV3)\n");

    total_start = clock();
    
    // 1. Carga de ficheros
    load_start = clock();
    FILE *fgraph, *fmap;
    load_files(&fgraph, &fmap);
    AdjMat adj(NB_NODES);
    int *outdeg = (int*) calloc(NB_NODES, sizeof(int));
    load_graph(fgraph, adj, outdeg);
    std::map<int, std::string> id_to_title;
    load_map(fmap, id_to_title);
    load_end = clock();

    // 2. Conversión CSR
    sparse_start = clock();
    int *row_ptr, *col_idx;
    convert_to_csr(adj, &row_ptr, &col_idx);
    sparse_end = clock();
    int num_edges = row_ptr[NB_NODES];

    // 3. Mapeo de Aristas (NUEVO)
    mapping_start = clock();
    int *edge_src = (int*) malloc(num_edges * sizeof(int));
    create_edge_mapping(NB_NODES, num_edges, row_ptr, edge_src);
    mapping_end = clock();

    // 4. PageRank
    pr_start = clock();
    double *p = (double*) malloc(NB_NODES * sizeof(double));
    pagerank_cuda_edge(row_ptr, col_idx, edge_src, outdeg, p);
    pr_end = clock();

    // 5. Resultados
    res_start = clock();
    print_results(p, id_to_title);
    res_end = clock();

    // Liberar memoria
    free(row_ptr); free(col_idx); free(edge_src); free(outdeg); free(p);
    total_end = clock();

    // Reporte de tiempos
    total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    load_time = (double)(load_end - load_start) / CLOCKS_PER_SEC;
    sparse_time = (double)(sparse_end - sparse_start) / CLOCKS_PER_SEC;
    mapping_time = (double)(mapping_end - mapping_start) / CLOCKS_PER_SEC;
    pr_time = (double)(pr_end - pr_start) / CLOCKS_PER_SEC;
    res_time = (double)(res_end - res_start) / CLOCKS_PER_SEC;
    
    printf("Total time: %.6f s\n", total_time);
    printf("  - Data loading: %.6f s\n", load_time);
    printf("  - CSR conversion: %.6f s\n", sparse_time);
    printf("  - Edge mapping: %.6f s\n", mapping_time);
    printf("  - PageRank (CUDA Edge): %.6f s\n", pr_time);
    printf("  - Results formatting: %.6f s\n", res_time);

    return 0;
}