// utils.cu
#include <map>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>

#include "params.h"

// ----------------------------------------------------------------------
// Estructura sparse: lista de adyacencia para grafo dirigido
// adj[u][k] = v  (enlace u -> v)
// ----------------------------------------------------------------------
typedef struct {
    int *data;
    int size;
    int cap;
} Vec;

static inline void vec_init(Vec *v) {
    v->size = 0;
    v->cap = 4;
    v->data = (int*) malloc(v->cap * sizeof(int));
}

static inline void vec_push(Vec *v, int x) {
    if (v->size == v->cap) {
        v->cap *= 2;
        v->data = (int*) realloc(v->data, v->cap * sizeof(int));
    }
    v->data[v->size++] = x;
}

void load_files(FILE **fgraph, FILE **fmap) {
    *fgraph = fopen("enwiki-2013.txt", "r");
    if (!*fgraph) {
        printf("No se pudo abrir el fichero del grafo\n");
        exit(1);
    }
    *fmap = fopen("enwiki-2013-names.csv", "r");
    if (!*fmap) {
        printf("No se pudo abrir el fichero del mapa\n");
        exit(1);
    }
}

// Cargar el grafo desde el fichero
void load_graph(FILE *fgraph, Vec *adj, int *outdeg) {
    for (int i = 0; i < N; i++)
        vec_init(&adj[i]);

    int u, v;
    char line[128];
    while (fgets(line, sizeof(line), fgraph)) {
        if (line[0] == '#') continue;  // saltar comentarios
        if (sscanf(line, "%d %d", &u, &v) == 2) {
            if (u < N && v < N) {
                vec_push(&adj[u], v);
                outdeg[u]++;
            }
        }
    }
}

// Cargar el mapa de IDs a títulos
void load_map(FILE *fmap, std::map<int, std::string> &id_to_title) {
    char line[512];
    while (fgets(line, sizeof(line), fmap)) {
        int id;
        char name[480];
        // Skip header line
        if (strncmp(line, "\"node_id\"", 9) == 0)
            continue;
        // Parse: id,"name"
        if (sscanf(line, "%d,\"%[^\"]\"", &id, name) == 2) {
            id_to_title[id] = std::string(name);
        }
    }
}

// Mostrar los ELEMS_A_MOSTRAR nodos con mayor PageRank
void print_results(double *p, std::map<int, std::string> &id_to_title) {
    struct PRNode {
        int idx;
        double val;
        bool operator<(const PRNode& other) const {
            return val > other.val; // Para min-heap
        }
    };

    std::priority_queue<PRNode, std::vector<PRNode>> minheap;
    for (int i = 0; i < N; i++) {
        if ((int)minheap.size() < ELEMS_A_MOSTRAR) {
            minheap.push({i, p[i]});
        } else if (p[i] > minheap.top().val) {
            minheap.pop();
            minheap.push({i, p[i]});
        }
    }

    // Extraer y ordenar resultados
    std::vector<PRNode> top_nodes;
    while (!minheap.empty()) {
        top_nodes.push_back(minheap.top());
        minheap.pop();
    }
    std::sort(top_nodes.begin(), top_nodes.end(), [](const PRNode& a, const PRNode& b) {
        return a.val > b.val;
    });

    for (int i = 0; i < (int)top_nodes.size(); i++)
        printf("p[%d] = %.10f\t%s\n", top_nodes[i].idx, top_nodes[i].val, id_to_title[top_nodes[i].idx].c_str());
}
