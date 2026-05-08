#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include "graph.h" // Assuming this contains generate_uniform_graph / generate_rmat_graph

int main(int argc, char** argv) {
    int scale = 20;
    int target_degree = 2048;
    std::string graph_type = "uniform";
    std::string out_dir = "./";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) scale = atoi(argv[++i]);
        else if (strcmp(argv[i], "--target_degree") == 0 && i + 1 < argc) target_degree = atoi(argv[++i]);
        else if (strcmp(argv[i], "--graph_type") == 0 && i + 1 < argc) graph_type = argv[++i];
        else if (strcmp(argv[i], "--out_dir") == 0 && i + 1 < argc) out_dir = argv[++i];
    }

    if (out_dir.back() != '/') out_dir += "/";

    printf("=== Generating %s Graph (Scale %d) ===\n", graph_type.c_str(), scale);
    
    CSRGraph g;
    if (graph_type == "uniform") {
        g = generate_uniform_graph(scale, target_degree, out_dir);
    } else if (graph_type == "rmat") {
        g = generate_rmat_graph(scale, target_degree, out_dir);
    } else {
        printf("Invalid graph type.\n"); exit(1);
    }

    // Find heaviest node to use as source
    int source = 0, max_deg = 0;
    for (int i = 0; i < g.num_nodes; i++) {
        int deg = g.row_offsets[i+1] - g.row_offsets[i];
        if (deg > max_deg) { max_deg = deg; source = i; }
    }

    printf("Nodes: %d, Edges: %lld, Source: %d\n", g.num_nodes, (long long)g.num_edges, source);
    printf("Writing to disk...\n");

    // 1. Write Metadata (num_nodes, num_edges, source, max_deg)
    int64_t metadata[4] = {g.num_nodes, g.num_edges, source, max_deg};
    std::ofstream m_file(out_dir + "metadata.bin", std::ios::binary);
    m_file.write(reinterpret_cast<char*>(metadata), sizeof(metadata));
    m_file.close();

    // 2. Write Offsets
    std::ofstream o_file(out_dir + "offsets.bin", std::ios::binary);
    o_file.write(reinterpret_cast<char*>(g.row_offsets), g.offsets_size_bytes);
    o_file.close();

    // 3. Write Edges
    std::ofstream e_file(out_dir + "edges.bin", std::ios::binary);
    e_file.write(reinterpret_cast<char*>(g.col_indices), g.edges_size_bytes);
    e_file.close();

    free_graph(&g);
    printf("Done.\n");
    return 0;
}