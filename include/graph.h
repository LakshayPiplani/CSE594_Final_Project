#ifndef BAM_GRAPH_H
#define BAM_GRAPH_H

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>

// ============================================================================
// CSR (Compressed Sparse Row) graph representation
//
// row_offsets[i] = start index into col_indices for node i's neighbors
// row_offsets[i+1] - row_offsets[i] = degree of node i
// col_indices[row_offsets[i] .. row_offsets[i+1]-1] = neighbors of node i
// ============================================================================

struct CSRGraph {
    int      num_nodes;
    int64_t  num_edges;
    int*     row_offsets;    // [num_nodes + 1]
    int*     col_indices;    // [num_edges]
    size_t   edges_size_bytes;
    size_t   offsets_size_bytes;
};

// ============================================================================
// Generate a synthetic RMAT graph (power-law degree distribution)
// This produces graphs similar to the GAP Benchmark Suite's kron graphs
// used in the BaM paper.
//
// Parameters:
//   scale: num_nodes = 2^scale
//   edge_factor: num_edges = num_nodes * edge_factor
//   a,b,c,d: RMAT probability parameters (default: 0.57, 0.19, 0.19, 0.05)
// ============================================================================

inline CSRGraph generate_rmat_graph(int scale, int edge_factor,
                                     double a = 0.57, double b = 0.19,
                                     double c = 0.19, double d = 0.05) {
    int num_nodes = 1 << scale;
    int64_t num_edges_target = (int64_t)num_nodes * edge_factor;

    printf("Generating RMAT graph: scale=%d, nodes=%d, target_edges=%lld\n",
           scale, num_nodes, (long long)num_edges_target);

    // Generate edge list
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Use vectors for edge list, then convert to CSR
    std::vector<std::pair<int,int>> edges;
    edges.reserve(num_edges_target);

    for (int64_t e = 0; e < num_edges_target; e++) {
        int u = 0, v = 0;
        for (int level = scale - 1; level >= 0; level--) {
            double r = dist(rng);
            if (r < a) {
                // quadrant (0,0)
            } else if (r < a + b) {
                v |= (1 << level);
            } else if (r < a + b + c) {
                u |= (1 << level);
            } else {
                u |= (1 << level);
                v |= (1 << level);
            }
        }
        if (u != v) {  // no self-loops
            edges.push_back({u, v});
            edges.push_back({v, u});  // undirected
        }
    }

    // Sort by source node
    std::sort(edges.begin(), edges.end());

    // Remove duplicates
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    int64_t num_edges = edges.size();
    printf("  After dedup: %lld directed edges (%.1f MB edge data)\n",
           (long long)num_edges,
           (double)num_edges * sizeof(int) / (1024*1024));

    // Build CSR
    CSRGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = num_edges;
    g.offsets_size_bytes = (num_nodes + 1) * sizeof(int);
    g.edges_size_bytes = num_edges * sizeof(int);

    g.row_offsets = (int*)malloc(g.offsets_size_bytes);
    g.col_indices = (int*)malloc(g.edges_size_bytes);

    memset(g.row_offsets, 0, g.offsets_size_bytes);

    // Count degrees
    for (int64_t i = 0; i < num_edges; i++) {
        g.row_offsets[edges[i].first + 1]++;
    }

    // Prefix sum
    for (int i = 1; i <= num_nodes; i++) {
        g.row_offsets[i] += g.row_offsets[i-1];
    }

    // Fill col_indices
    std::vector<int> current_pos(num_nodes, 0);
    for (int64_t i = 0; i < num_edges; i++) {
        int src = edges[i].first;
        int dst = edges[i].second;
        int pos = g.row_offsets[src] + current_pos[src];
        g.col_indices[pos] = dst;
        current_pos[src]++;
    }

// Print stats
    std::vector<int> degrees(num_nodes, 0);
    int max_degree = 0;
    int64_t total_degree = 0;
    
    for (int i = 0; i < num_nodes; i++) {
        int deg = g.row_offsets[i+1] - g.row_offsets[i];
        degrees[i] = deg;
        max_degree = std::max(max_degree, deg);
        total_degree += deg;
    }
    
    // Sort to find percentiles
    std::sort(degrees.begin(), degrees.end());
    int q1 = degrees[num_nodes * 0.25];
    int median = degrees[num_nodes * 0.50];
    int q3 = degrees[num_nodes * 0.75];

    printf("  Avg degree: %.1f, Max degree: %d\n",
           (double)total_degree / num_nodes, max_degree);
    printf("  Degree Distribution -> Q1: %d | Median: %d | Q3: %d\n", q1, median, q3);

    return g;
}

// Simple ring + random graph for quick testing
inline CSRGraph generate_simple_graph(int num_nodes, int avg_degree) {
    printf("Generating simple graph: nodes=%d, avg_degree=%d\n",
           num_nodes, avg_degree);

    std::mt19937 rng(42);
    std::vector<std::pair<int,int>> edges;

    // Ring edges (ensure connectivity)
    for (int i = 0; i < num_nodes; i++) {
        int next = (i + 1) % num_nodes;
        edges.push_back({i, next});
        edges.push_back({next, i});
    }

    // Random edges
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);
    int64_t random_edges = (int64_t)num_nodes * avg_degree / 2;
    for (int64_t e = 0; e < random_edges; e++) {
        int u = node_dist(rng);
        int v = node_dist(rng);
        if (u != v) {
            edges.push_back({u, v});
            edges.push_back({v, u});
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    int64_t num_edges = edges.size();
    printf("  Edges: %lld (%.1f MB)\n", (long long)num_edges,
           (double)num_edges * sizeof(int) / (1024*1024));

    CSRGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = num_edges;
    g.offsets_size_bytes = (num_nodes + 1) * sizeof(int);
    g.edges_size_bytes = num_edges * sizeof(int);

    g.row_offsets = (int*)malloc(g.offsets_size_bytes);
    g.col_indices = (int*)malloc(g.edges_size_bytes);
    memset(g.row_offsets, 0, g.offsets_size_bytes);

    for (int64_t i = 0; i < num_edges; i++)
        g.row_offsets[edges[i].first + 1]++;
    for (int i = 1; i <= num_nodes; i++)
        g.row_offsets[i] += g.row_offsets[i-1];

    std::vector<int> pos(num_nodes, 0);
    for (int64_t i = 0; i < num_edges; i++) {
        int src = edges[i].first;
        g.col_indices[g.row_offsets[src] + pos[src]++] = edges[i].second;
    }

    return g;
}

// ============================================================================
// Generate a Uniform (Erdős–Rényi) random graph
// This guarantees a tight Poisson degree distribution centered at target_degree.
// Eliminates the power-law skew that causes warp starvation.
// ============================================================================

inline CSRGraph generate_uniform_graph(int scale, int target_degree) {
    int num_nodes = 1 << scale;
    // We divide by 2 because each undirected edge adds 2 directed edges later
    int64_t num_edges_target = (int64_t)num_nodes * target_degree / 2;

    printf("Generating Uniform graph: scale=%d, nodes=%d, target_degree=%d\n",
           scale, num_nodes, target_degree);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, num_nodes - 1);

    std::vector<std::pair<int,int>> edges;
    edges.reserve(num_edges_target * 2);

    // Generate random uniformly distributed edges
    for (int64_t e = 0; e < num_edges_target; e++) {
        int u = dist(rng);
        int v = dist(rng);
        if (u != v) {
            edges.push_back({u, v});
            edges.push_back({v, u}); // Undirected
        }
    }

    // Sort and remove duplicates
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    int64_t num_edges = edges.size();
    printf("  After dedup: %lld directed edges (%.1f MB edge data)\n",
           (long long)num_edges,
           (double)num_edges * sizeof(int) / (1024*1024));

    // Build CSR
    CSRGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = num_edges;
    g.offsets_size_bytes = (num_nodes + 1) * sizeof(int);
    g.edges_size_bytes = num_edges * sizeof(int);

    g.row_offsets = (int*)malloc(g.offsets_size_bytes);
    g.col_indices = (int*)malloc(g.edges_size_bytes);

    memset(g.row_offsets, 0, g.offsets_size_bytes);

    // Count degrees
    for (int64_t i = 0; i < num_edges; i++) {
        g.row_offsets[edges[i].first + 1]++;
    }

    // Prefix sum
    for (int i = 1; i <= num_nodes; i++) {
        g.row_offsets[i] += g.row_offsets[i-1];
    }

    // Fill col_indices
    std::vector<int> current_pos(num_nodes, 0);
    for (int64_t i = 0; i < num_edges; i++) {
        int src = edges[i].first;
        int dst = edges[i].second;
        int pos = g.row_offsets[src] + current_pos[src];
        g.col_indices[pos] = dst;
        current_pos[src]++;
    }

    // Calculate and print strict quartiles
    std::vector<int> degrees(num_nodes, 0);
    int max_degree = 0;
    int64_t total_degree = 0;
    for (int i = 0; i < num_nodes; i++) {
        int deg = g.row_offsets[i+1] - g.row_offsets[i];
        degrees[i] = deg;
        max_degree = std::max(max_degree, deg);
        total_degree += deg;
    }
    std::sort(degrees.begin(), degrees.end());
    int q1 = degrees[num_nodes * 0.25];
    int median = degrees[num_nodes * 0.50];
    int q3 = degrees[num_nodes * 0.75];

    printf("  Avg degree: %.1f, Max degree: %d\n",
           (double)total_degree / num_nodes, max_degree);
    printf("  Degree Distribution -> Q1: %d | Median: %d | Q3: %d\n", q1, median, q3);

    return g;
}

inline void free_graph(CSRGraph* g) {
    free(g->row_offsets);
    free(g->col_indices);
}

#endif // BAM_GRAPH_H
