// utils.cu
#include <map>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>

#include "params.h"

using Vec = std::vector<int>;
using AdjMat = std::vector<Vec>;

// Load the graph and map files
void load_files(FILE **fgraph, FILE **fmap) {
    *fgraph = fopen(GRAPH_FILE, "r");
    if (!*fgraph) {
        printf("No se pudo abrir el fichero del grafo\n");
        exit(1);
    }
    *fmap = fopen(MAP_FILE, "r");
    if (!*fmap) {
        printf("No se pudo abrir el fichero del mapa\n");
        exit(1);
    }
}

// Load the graph from file into adjacency list and outdegree array
void load_graph(FILE *fgraph, AdjMat &adj, int *outdeg) {
    adj = AdjMat(NB_NODES);
    
    int u, v;
    char line[128];
    while (fgets(line, sizeof(line), fgraph)) {
        if (line[0] == '#') continue;  // skip comments
        if (sscanf(line, "%d %d", &u, &v) == 2) {
            if (u < NB_NODES && v < NB_NODES) {
                // Memory optimization: reserve extra space if needed
                if (adj[u].capacity() == adj[u].size())
                    adj[u].reserve(adj[u].size()*2);
                adj[u].push_back(v);
                outdeg[u]++;
            }
        }
    }
}

// Load the map from file into ID-to-title map
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

// Convert adjacency list representation to CSR
void convert_to_csr(AdjMat &adj, int **row_ptr_ptr, int **col_idx_ptr) {
    int *row_ptr = (int*) malloc((NB_NODES + 1) * sizeof(int));
    row_ptr[0] = 0;

    int total_edges = 0;
    for (int i = 0; i < NB_NODES; i++)
        total_edges += adj[i].size();

    int *col_idx = (int*) malloc(total_edges * sizeof(int));

    int pos = 0;
    for (int u = 0; u < NB_NODES; u++) {
        row_ptr[u+1] = row_ptr[u] + adj[u].size();
        for (int k = 0; k < adj[u].size(); k++)
            col_idx[pos++] = adj[u][k];
    }

    *row_ptr_ptr = row_ptr;
    *col_idx_ptr = col_idx;
}

// Show the top ELEMS_TO_SHOW nodes with highest PageRank
void print_results(double *p, std::map<int, std::string> &id_to_title) {
    struct PRNode {
        int idx;
        double val;
        bool operator<(const PRNode& other) const {
            return val > other.val; // For min-heap
        }
    };

    std::priority_queue<PRNode, std::vector<PRNode>> minheap;
    for (int i = 0; i < NB_NODES; i++) {
        if ((int)minheap.size() < ELEMS_TO_SHOW) {
            minheap.push({i, p[i]});
        } else if (p[i] > minheap.top().val) {
            minheap.pop();
            minheap.push({i, p[i]});
        }
    }

    // Extract and sort results
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

