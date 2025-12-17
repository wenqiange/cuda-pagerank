#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "params.h"
// #include "utils.cu" // Asumo que esto ya está incluido o manejado externamente

// Macro para chequear errores de CUDA (Buenas prácticas de TGA)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ----------------------------------------------------------------------
// Kernel: Inicialización de p_new con el factor de amortiguación base
// ----------------------------------------------------------------------
__global__ void init_p_new_kernel(double *p_new, double base_value, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        p_new[idx] = base_value;
    }
}

// ----------------------------------------------------------------------
// Kernel: PageRank Update (Estrategia PUSH con Atomics)
// ----------------------------------------------------------------------
__global__ void pagerank_update_kernel(const int *row_ptr, const int *col_idx, 
                                       const int *outdeg, const double *p, 
                                       double *p_new, int num_nodes) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < num_nodes) {
        if (outdeg[u] > 0) {
            // Calculamos la contribución de este nodo u
            double contribution = LAMBDA * p[u] / (double)outdeg[u];
            
            // Recorremos los vecinos (CSR traversal)
            int start = row_ptr[u];
            int end = row_ptr[u + 1];

            for (int k = start; k < end; k++) {
                int v = col_idx[k];
                // ATENCIÓN: Múltiples hilos pueden escribir en p_new[v] simultáneamente.
                // Usamos atomicAdd para serializar las sumas en esa dirección de memoria.
                atomicAdd(&p_new[v], contribution);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Kernel: Añadir Dangling Factor
// ----------------------------------------------------------------------
__global__ void apply_dangling_kernel(double *p_new, double add_dang, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        p_new[idx] += add_dang;
    }
}

// ----------------------------------------------------------------------
// Functores para Thrust (Reducciones)
// ----------------------------------------------------------------------

// Calcula |a - b|
struct abs_diff_functor {
    __host__ __device__
    double operator()(const double& a, const double& b) const {
        return fabs(a - b);
    }
};

// Filtra nodos dangling (outdeg == 0) y devuelve su valor p, sino 0.0
struct dangling_filter_functor {
    const int* outdeg;
    const double* p;
    
    dangling_filter_functor(const int* _outdeg, const double* _p) : outdeg(_outdeg), p(_p) {}

    __host__ __device__
    double operator()(const int& idx) const {
        if (outdeg[idx] == 0) return p[idx];
        return 0.0;
    }
};


// ----------------------------------------------------------------------
// PageRank using CUDA
// ----------------------------------------------------------------------
void pagerank(int *h_row_ptr, int *h_col_idx, int *h_outdeg, double *h_p) {
    
    // 1. Reservar memoria en GPU (Device)
    int *d_row_ptr, *d_col_idx, *d_outdeg;
    double *d_p, *d_p_new;

    // Calculamos tamaño del array de aristas (nnz)
    int nnz = h_row_ptr[NB_NODES]; // El último elemento de row_ptr tiene el total de aristas

    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (NB_NODES + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_outdeg, NB_NODES * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_p, NB_NODES * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_p_new, NB_NODES * sizeof(double)));

    // 2. Copiar datos estáticos del Grafo (Host -> Device)
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (NB_NODES + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outdeg, h_outdeg, NB_NODES * sizeof(int), cudaMemcpyHostToDevice));
    
    // Inicialización inicial de P en el Host y copia
    for (int i = 0; i < NB_NODES; i++) h_p[i] = 1.0 / NB_NODES;
    CUDA_CHECK(cudaMemcpy(d_p, h_p, NB_NODES * sizeof(double), cudaMemcpyHostToDevice));

    // Configuración de ejecución (Grid-Stride loop standard)
    int blockSize = 256;
    int gridSize = (NB_NODES + blockSize - 1) / blockSize;

    int iter = 0;
    double diff = 1.0;

    // --- BUCLE PRINCIPAL ---
    while (iter < MAX_ITER && diff > EPSILON) {
        
        // A. Inicializar p_new con (1 - LAMBDA) / N
        double base_val = (1.0 - LAMBDA) / NB_NODES;
        init_p_new_kernel<<<gridSize, blockSize>>>(d_p_new, base_val, NB_NODES);
        CUDA_CHECK(cudaGetLastError());

        // B. Calcular Dangling Sum (Reducción)
        // Usamos Thrust para transformar (filtrar por outdeg==0) y reducir (sumar)
        // Generamos una secuencia de índices [0, 1, ... NB_NODES-1]
        thrust::counting_iterator<int> iter_begin(0);
        thrust::counting_iterator<int> iter_end(NB_NODES);
        
        double dangling_sum = thrust::transform_reduce(
            thrust::device,
            iter_begin, iter_end,
            dangling_filter_functor(d_outdeg, d_p),
            0.0,
            thrust::plus<double>()
        );

        // C. Núcleo del algoritmo: Distribución de ranks (Push)
        pagerank_update_kernel<<<gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_outdeg, d_p, d_p_new, NB_NODES);
        CUDA_CHECK(cudaGetLastError());

        // D. Añadir factor dangling a todos
        double add_dang = LAMBDA * dangling_sum / NB_NODES;
        apply_dangling_kernel<<<gridSize, blockSize>>>(d_p_new, add_dang, NB_NODES);
        CUDA_CHECK(cudaGetLastError());

        // E. Calcular Diferencia (Norma L1) y Actualizar P
        // Calculamos diff = sum(|p_new - p|)
        thrust::device_ptr<double> thr_p(d_p);
        thrust::device_ptr<double> thr_p_new(d_p_new);
        
        diff = thrust::transform_reduce(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(thr_p_new, thr_p)),
            thrust::make_zip_iterator(thrust::make_tuple(thr_p_new + NB_NODES, thr_p + NB_NODES)),
            abs_diff_functor(),
            0.0,
            thrust::plus<double>()
        );

        // Actualizamos P para la siguiente iteración (Swap de punteros es imposible aquí 
        // porque la función recibe p_new calculado en base a p, necesitamos copiar 
        // o simplemente copiar el contenido de p_new a p para mantener consistencia con 'diff')
        // La forma más rápida es copiar device-to-device:
        CUDA_CHECK(cudaMemcpy(d_p, d_p_new, NB_NODES * sizeof(double), cudaMemcpyDeviceToDevice));

        iter++;
    }

    printf("Convergencia en %d iteraciones (CUDA)\n", iter);

    // 3. Copiar resultado final (Device -> Host)
    CUDA_CHECK(cudaMemcpy(h_p, d_p, NB_NODES * sizeof(double), cudaMemcpyDeviceToHost));

    // 4. Liberar memoria GPU
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_outdeg));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_new));
}