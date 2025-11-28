#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>


#ifndef SIZE
#define SIZE 32
#endif 

#ifndef PINNED
#define PINNED 0
#endif 

#define N 4206784
#define LAMBDA 0.90
#define EPSILON 1e-6
#define MAX_ITER 100
#define ELEMS_A_MOSTRAR 10

// ----------------------------------------------------------------------
// Estructura sparse: lista de adyacencia para grafo dirigido
// adj[u][k] = v  (enlace u -> v)
// ----------------------------------------------------------------------
typedef struct {
    int *data;
    int size;
    int cap;
} Vec;

static inline void *safe_malloc(size_t bytes) {
    void *ptr = malloc(bytes);
    if (!ptr) {
        fprintf(stderr, "Error: sin memoria (%zu bytes)\n", bytes);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static inline void *safe_calloc(size_t count, size_t bytes) {
    void *ptr = calloc(count, bytes);
    if (!ptr) {
        fprintf(stderr, "Error: sin memoria (%zu bytes)\n", count * bytes);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static inline void *safe_realloc(void *ptr, size_t bytes) {
    void *new_ptr = realloc(ptr, bytes);
    if (!new_ptr) {
        fprintf(stderr, "Error: sin memoria (%zu bytes)\n", bytes);
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static inline void vec_init(Vec *v) {
    v->size = 0;
    v->cap = 0;
    v->data = NULL;
}

static inline void vec_push(Vec *v, int x) {
    if (v->cap == 0) {
        v->cap = 4;
        v->data = (int*) safe_malloc(v->cap * sizeof(int));
    } else if (v->size == v->cap) {
        v->cap *= 2;
        v->data = (int*) safe_realloc(v->data, v->cap * sizeof(int));
    }
    v->data[v->size++] = x;
}

static void build_csr_from_vecs(const Vec *adj, int **row_offsets,
                                int **col_indices, size_t *edge_count) {
    size_t total = 0;
    for (int i = 0; i < N; i++)
        total += (size_t) adj[i].size;

    int *offsets = (int*) safe_malloc((N + 1) * sizeof(int));
    int *indices = NULL;
    if (total > 0)
        indices = (int*) safe_malloc(total * sizeof(int));

    offsets[0] = 0;
    size_t pos = 0;
    for (int i = 0; i < N; i++) {
        offsets[i + 1] = offsets[i] + adj[i].size;
        if (adj[i].size > 0) {
            memcpy(indices + pos, adj[i].data, adj[i].size * sizeof(int));
            pos += adj[i].size;
        }
    }

    *row_offsets = offsets;
    *col_indices = indices;
    *edge_count = total;
}
__global__ void scatter_kernel(
        const int *row_offsets, const int *col_indices,
        const int *outdeg, const double *p, double *p_new)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;

    if (outdeg[u] == 0) return;

    double contrib = LAMBDA * (p[u] / outdeg[u]);

    int start = row_offsets[u];
    int end = row_offsets[u + 1];
    for (int idx = start; idx < end; idx++) {
        int v = col_indices[idx];
        atomicAdd(&p_new[v], contrib);
    }
}

void pagerank(Vec *adj, int *outdeg, double *p)
{
    double *p_new = (double*) safe_malloc(N * sizeof(double));

    int *row_offsets = NULL;
    int *col_indices = NULL;
    size_t edge_count = 0;
    build_csr_from_vecs(adj, &row_offsets, &col_indices, &edge_count);

    // Inicialización CPU (identica a tu código)
    for (int i = 0; i < N; i++)
        p[i] = 1.0 / N;

    // Copiar datos a GPU
    int *d_outdeg;
    double *d_p;
    double *d_p_new;
    int *d_row_offsets;
    int *d_col_indices = NULL;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_offsets, row_offsets, (N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    if (edge_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_col_indices, edge_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_col_indices, col_indices,
                              edge_count * sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_outdeg, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_p,      N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p_new,  N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_outdeg, outdeg, N * sizeof(int),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    int iter = 0;

    while (iter < MAX_ITER) {

        // p_new[i] = (1 - LAMBDA)/N
        double base = (1.0 - LAMBDA) / N;
        for (int i = 0; i < N; i++)
            p_new[i] = base;

        // copiar p y p_new al device
        CUDA_CHECK(cudaMemcpy(d_p,     p,     N*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p_new, p_new, N*sizeof(double), cudaMemcpyHostToDevice));

        // calcular dangling_sum (CPU, igual que tu código)
        double dangling_sum = 0.0;
        for (int u = 0; u < N; u++)
            if (outdeg[u] == 0)
                dangling_sum += p[u];

        double add_dang = LAMBDA * dangling_sum / N;

        scatter_kernel<<<blocks, threads>>>(d_row_offsets, d_col_indices,
                            d_outdeg, d_p, d_p_new);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // volver a copiar p_new
        CUDA_CHECK(cudaMemcpy(p_new, d_p_new, N*sizeof(double), cudaMemcpyDeviceToHost));

        // añadir dangling en CPU 
        for (int i = 0; i < N; i++)
            p_new[i] += add_dang;

        // diff + copiar p_new → p 
        double diff = 0.0;
        for (int i = 0; i < N; i++) {
            diff += fabs(p_new[i] - p[i]);
            p[i] = p_new[i];
        }

        if (diff < EPSILON) break;
        iter++;
    }

    printf("Convergencia en %d iteraciones\n", iter);

    // liberar GPU
    CUDA_CHECK(cudaFree(d_outdeg));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_new));
    CUDA_CHECK(cudaFree(d_row_offsets));
    if (d_col_indices)
        CUDA_CHECK(cudaFree(d_col_indices));

    free(row_offsets);
    free(col_indices);

    free(p_new);
}
int main() {

    FILE *file = fopen("enwiki-2013.txt", "r");
    if (!file) {
        printf("No se pudo abrir el fichero\n");
        return 1;
    }

    Vec *adj = (Vec*) safe_malloc(N * sizeof(Vec));
    int *outdeg = (int*) safe_calloc(N, sizeof(int));

    for (int i = 0; i < N; i++)
        vec_init(&adj[i]);

    int u, v;
    char line[128];
    size_t invalid_edges = 0;
	while (fgets(line, sizeof(line), file)) {

		if (line[0] == '#') continue;  // saltar comentarios

		if (sscanf(line, "%d %d", &u, &v) == 2) {
			if (u >= 0 && v >= 0 && u < N && v < N) {
				vec_push(&adj[u], v);
				outdeg[u]++;
			} else {
				invalid_edges++;
			}
		}
	}

    fclose(file);

    if (invalid_edges > 0)
        fprintf(stderr, "Advertencia: se descartaron %zu aristas fuera de rango [0,%d)\n", invalid_edges, N);

    double *p = (double*) safe_malloc(N * sizeof(double));
    clock_t start_time_pr, end_time_pr;
    start_time_pr = clock();
    pagerank(adj, outdeg, p);
    end_time_pr = clock();
    double pagerank_time = (double)(end_time_pr - start_time_pr) / CLOCKS_PER_SEC;
    printf("Tiempo total PageRank: %.6f segundos\n", pagerank_time);

    // Mostrar los 10 nodos con mayor PageRank
    int idx[ELEMS_A_MOSTRAR];
    double val[ELEMS_A_MOSTRAR];
    for (int i = 0; i < ELEMS_A_MOSTRAR; i++) {
        idx[i] = -1;
        val[i] = -1.0;
    }
    for (int i = 0; i < N; i++) {
        // Buscar si p[i] es mayor que alguno de los 10 actuales
        int min_idx = 0;
        for (int j = 1; j < ELEMS_A_MOSTRAR; j++)
            if (val[j] < val[min_idx]) min_idx = j;
        if (p[i] > val[min_idx]) {
            val[min_idx] = p[i];
            idx[min_idx] = i;
        }
    }
    // Mostrar los resultados
    //TODO: que se muestren ordenados
    for (int i = 0; i < ELEMS_A_MOSTRAR; i++)
        printf("p[%d] = %.10f\n", idx[i], val[i]);

    for (int i = 0; i < N; i++)
        free(adj[i].data);

    free(adj);
    free(outdeg);
    free(p);

    return 0;
}
