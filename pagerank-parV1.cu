#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <time.h> // Para clock()

#include "params.h"
#include "utils.cu"

// --- VARIABLES GLOBALES PARA TIEMPOS ---
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
time_t mv_start, mv_end;         // Medirá Transferencia + Kernel
double accum_mv_time = 0.0;

// ----------------------------------------------------------------------
// KERNEL CUDA (Sustituye al doble bucle for)
// ----------------------------------------------------------------------
__global__ void kernel_pagerank_push(int N, const int *row_ptr, const int *col_idx, 
                                     const int *outdeg, const double *p, 
                                     double *p_new, double lambda) {
    // Cada hilo procesa un nodo origen 'u'
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < N && outdeg[u] > 0) {
        double contrib = lambda * p[u] / (double)outdeg[u];
        
        // Recorrer los vecinos
        for (int k = row_ptr[u]; k < row_ptr[u + 1]; k++) {
            int v = col_idx[k];
            // atomicAdd para double es nativo en tu arquitectura (sm_86/89)
            atomicAdd(&p_new[v], contrib);
        }
    }
}

// ----------------------------------------------------------------------
// PageRank V1 (Híbrido CPU/GPU)
// ----------------------------------------------------------------------
void pagerank_cuda_v1(int *h_row_ptr, int *h_col_idx, int *h_outdeg, double *h_p) {
    int N = NB_NODES;
    int num_edges = h_row_ptr[N]; 

    // --- 1. Reserva de memoria en GPU ---
    int *d_row_ptr, *d_col_idx, *d_outdeg;
    double *d_p, *d_p_new;

    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, num_edges * sizeof(int));
    cudaMalloc(&d_outdeg, N * sizeof(int));
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_p_new, N * sizeof(double));

    // --- 2. Copia de datos ESTÁTICOS (solo una vez) ---
    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outdeg, h_outdeg, N * sizeof(int), cudaMemcpyHostToDevice);

    // Inicialización inicial de P en CPU
    for (int i = 0; i < N; i++) h_p[i] = 1.0 / N;
    
    // Buffer temporal en Host para p_new
    double *h_p_new = (double*) malloc(N * sizeof(double));

    // Configuración de ejecución
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int iter = 0;
    while (iter < MAX_ITER) {
        
        // --- [CPU] Lógica secuencial (Inicializar p_new y Dangling) ---
        for (int i = 0; i < N; i++) 
            h_p_new[i] = (1.0 - LAMBDA) / N;

        double dangling_sum = 0.0;
        for (int u = 0; u < N; u++) {
            if (h_outdeg[u] == 0)
                dangling_sum += h_p[u];
        }

        // --- [GPU] PARTE OFFLOADED (El doble bucle) ---
        mv_start = clock(); // Iniciamos cronómetro de la parte "acelerada"

        // A) Copiar p (actualizado) y p_new (inicializado) a la GPU
        cudaMemcpy(d_p, h_p, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p_new, h_p_new, N * sizeof(double), cudaMemcpyHostToDevice);

        // B) Lanzar Kernel
        kernel_pagerank_push<<<blocksPerGrid, threadsPerBlock>>>(
            N, d_row_ptr, d_col_idx, d_outdeg, d_p, d_p_new, LAMBDA
        );
        
        // C) Recuperar resultados
        cudaMemcpy(h_p_new, d_p_new, N * sizeof(double), cudaMemcpyDeviceToHost);

        mv_end = clock();
        accum_mv_time += (double)(mv_end - mv_start) / CLOCKS_PER_SEC;

        // --- [CPU] Lógica secuencial (Convergenica y Actualización) ---
        double add_dang = LAMBDA * dangling_sum / N;
        double diff = 0.0;

        for (int i = 0; i < N; i++) {
            h_p_new[i] += add_dang;     
            diff += fabs(h_p_new[i] - h_p[i]); 
            h_p[i] = h_p_new[i];        
        }

        if (diff < EPSILON) break;
        iter++;
    }

    printf("Convergencia en %d iteraciones\n", iter);

    // Liberar recursos
    free(h_p_new);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_outdeg);
    cudaFree(d_p);
    cudaFree(d_p_new);
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main() {
    printf("CUDA PageRank (pagerank-parV1)\n");

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
    
    // LLAMADA A LA NUEVA FUNCIÓN
    pagerank_cuda_v1(row_ptr, col_idx, outdeg, p);
    
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
    printf("Accumulated MV time (Transfer + Kernel): %.6f seconds\n", accum_mv_time);

    return 0;
}