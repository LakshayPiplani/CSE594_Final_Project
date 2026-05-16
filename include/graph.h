#ifndef BAM_GRAPH_H
#define BAM_GRAPH_H

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>
#include <string>
#include <omp.h>

/*
 CSR (Compressed Sparse Row) graph

 row_offsets[i] = start index into col_indices for node i's neighbors
 row_offsets[i+1] - row_offsets[i] = degree of node i
 col_indices[row_offsets[i] .. row_offsets[i+1]-1] = neighbors of node i
*/

struct CSRGraph
{
    uint64_t num_nodes;
    uint64_t num_edges;
    uint32_t *row_offsets; // [num_nodes + 1]
    int *col_indices;      // [num_edges]
    size_t edges_size_bytes;
    size_t offsets_size_bytes;
};

inline CSRGraph generate_rmat_graph(int scale, int edge_factor,
                                    const std::string &out_dir,
                                    double a = 0.57, double b = 0.19,
                                    double c = 0.19, double d = 0.05)
{
    uint64_t num_nodes = 1 << scale;
    uint64_t num_edges_target = (uint64_t)num_nodes * edge_factor;

    printf("Generating RMAT graph: scale=%d, nodes=%lld, target_edges=%lld\n",
           scale, num_nodes, (long long)num_edges_target);

    // Generate edge list
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<uint64_t> edges;
    edges.reserve(num_edges_target * 2);

    for (int64_t e = 0; e < (int64_t)num_edges_target; e++)
    {
        uint32_t u = 0, v = 0;
        for (int level = scale - 1; level >= 0; level--)
        {
            double r = dist(rng);
            if (r < a)
            {
                // quadrant (0,0)
            }
            else if (r < a + b)
            {
                v |= (1u << level);
            }
            else if (r < a + b + c)
            {
                u |= (1u << level);
            }
            else
            {
                u |= (1u << level);
                v |= (1u << level);
            }
        }
        if (u != v)
        {
            edges.push_back(((uint64_t)u << 32) | v);
            edges.push_back(((uint64_t)v << 32) | u);
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    edges.shrink_to_fit();

    int64_t num_edges = (int64_t)edges.size();
    printf("  After dedup: %lld directed edges (%.1f MB edge data)\n",
           (long long)num_edges,
           (double)num_edges * sizeof(int) / (1024 * 1024));

    // Build CSR
    CSRGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = num_edges;
    g.offsets_size_bytes = (num_nodes + 1) * sizeof(uint32_t);
    g.edges_size_bytes = (size_t)num_edges * sizeof(int);

    g.row_offsets = (uint32_t *)malloc(g.offsets_size_bytes);
    if (!g.row_offsets)
    {
        fprintf(stderr, "Fatal: malloc failed for row_offsets (%zu MB)\n", g.offsets_size_bytes >> 20);
        exit(1);
    }
    memset(g.row_offsets, 0, g.offsets_size_bytes);

    // Build row_offsets (prefix sum) from edges
    for (int64_t i = 0; i < num_edges; i++)
    {
        uint32_t src = (uint32_t)(edges[i] >> 32);
        g.row_offsets[src + 1]++;
    }
    for (uint64_t i = 1; i <= num_nodes; i++)
    {
        g.row_offsets[i] += g.row_offsets[i - 1];
    }

    // Spill dst values (already in CSR order since edges is sorted) to disk,
    // then free edges before allocating col_indices to avoid peak RAM overlap
    std::string tmp_path = out_dir + "edges_tmp.bin";
    {
        FILE *tmp = fopen(tmp_path.c_str(), "wb");
        if (!tmp)
        {
            fprintf(stderr, "Fatal: cannot open %s\n", tmp_path.c_str());
            exit(1);
        }
        // Write in chunks to avoid per-element call overhead
        const int64_t CHUNK = 1 << 20; // 1M edges per chunk
        std::vector<int> buf(CHUNK);
        for (int64_t i = 0; i < num_edges; i += CHUNK)
        {
            int64_t count = std::min(CHUNK, num_edges - i);
            for (int64_t j = 0; j < count; j++)
                buf[j] = (int)(edges[i + j] & 0xFFFFFFFF);
            fwrite(buf.data(), sizeof(int), count, tmp);
        }
        fclose(tmp);
    }
    {
        std::vector<uint64_t>().swap(edges);
    }

    g.col_indices = (int *)malloc(g.edges_size_bytes);
    if (!g.col_indices)
    {
        fprintf(stderr, "Fatal: malloc failed for col_indices (%zu MB)\n", g.edges_size_bytes >> 20);
        remove(tmp_path.c_str());
        exit(1);
    }
    {
        FILE *tmp = fopen(tmp_path.c_str(), "rb");
        if (!tmp)
        {
            fprintf(stderr, "Fatal: cannot reopen %s\n", tmp_path.c_str());
            exit(1);
        }
        size_t n = fread(g.col_indices, sizeof(int), num_edges, tmp);
        fclose(tmp);
        remove(tmp_path.c_str());
        if ((int64_t)n != num_edges)
        {
            fprintf(stderr, "Fatal: read %zu edges, expected %lld\n", n, (long long)num_edges);
            exit(1);
        }
    }

    // Print stats
    std::vector<int> degrees(num_nodes, 0);
    int max_degree = 0;
    int64_t total_degree = 0;

    for (int i = 0; i < num_nodes; i++)
    {
        int deg = (int)(g.row_offsets[i + 1] - g.row_offsets[i]);
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

// Generate a Uniform random graph

inline CSRGraph generate_uniform_graph(int scale, int target_degree, const std::string &out_dir)
{
    int num_nodes = 1 << scale;
    // We divide by 2 because each undirected edge adds 2 directed edges later
    int64_t num_edges_target = (int64_t)num_nodes * target_degree / 2;

    printf("Generating Uniform graph: scale=%d, nodes=%d, target_degree=%d\n",
           scale, num_nodes, target_degree);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)num_nodes - 1);

    std::vector<uint64_t> edges;
    edges.reserve((size_t)num_edges_target * 2);

    for (int64_t e = 0; e < num_edges_target; e++)
    {
        uint32_t u = dist(rng);
        uint32_t v = dist(rng);
        if (u != v)
        {
            edges.push_back(((uint64_t)u << 32) | v);
            edges.push_back(((uint64_t)v << 32) | u);
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    edges.shrink_to_fit();

    int64_t num_edges = (int64_t)edges.size();
    printf("  After dedup: %lld directed edges (%.1f MB edge data)\n",
           (long long)num_edges,
           (double)num_edges * sizeof(int) / (1024 * 1024));

    // Build CSR
    CSRGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = num_edges;
    g.offsets_size_bytes = (num_nodes + 1) * sizeof(uint32_t);
    g.edges_size_bytes = (size_t)num_edges * sizeof(int);

    g.row_offsets = (uint32_t *)malloc(g.offsets_size_bytes);
    if (!g.row_offsets)
    {
        fprintf(stderr, "Fatal: malloc failed for row_offsets (%zu MB)\n", g.offsets_size_bytes >> 20);
        exit(1);
    }
    memset(g.row_offsets, 0, g.offsets_size_bytes);

    // Build row_offsets (prefix sum) from edges
    for (int64_t i = 0; i < num_edges; i++)
    {
        uint32_t src = (uint32_t)(edges[i] >> 32);
        g.row_offsets[src + 1]++;
    }
    for (uint64_t i = 1; i <= (uint64_t)num_nodes; i++)
    {
        g.row_offsets[i] += g.row_offsets[i - 1];
    }

    // Spill dst values (already in CSR order since edges is sorted) to disk,
    // then free edges before allocating col_indices to avoid peak RAM overlap
    std::string tmp_path = out_dir + "edges_tmp.bin";
    {
        FILE *tmp = fopen(tmp_path.c_str(), "wb");
        if (!tmp)
        {
            fprintf(stderr, "Fatal: cannot open %s\n", tmp_path.c_str());
            exit(1);
        }
        // Write in chunks to avoid per-element call overhead
        const int64_t CHUNK = 1 << 20;
        std::vector<int> buf(CHUNK);
        for (int64_t i = 0; i < num_edges; i += CHUNK)
        {
            int64_t count = std::min(CHUNK, num_edges - i);
            for (int64_t j = 0; j < count; j++)
                buf[j] = (int)(edges[i + j] & 0xFFFFFFFF);
            fwrite(buf.data(), sizeof(int), count, tmp);
        }
        fclose(tmp);
    }
    {
        std::vector<uint64_t>().swap(edges);
    }

    g.col_indices = (int *)malloc(g.edges_size_bytes);
    if (!g.col_indices)
    {
        fprintf(stderr, "Fatal: malloc failed for col_indices (%zu MB)\n", g.edges_size_bytes >> 20);
        remove(tmp_path.c_str());
        exit(1);
    }
    {
        FILE *tmp = fopen(tmp_path.c_str(), "rb");
        if (!tmp)
        {
            fprintf(stderr, "Fatal: cannot reopen %s\n", tmp_path.c_str());
            exit(1);
        }
        size_t n = fread(g.col_indices, sizeof(int), num_edges, tmp);
        fclose(tmp);
        remove(tmp_path.c_str());
        if ((int64_t)n != num_edges)
        {
            fprintf(stderr, "Fatal: read %zu edges, expected %lld\n", n, (long long)num_edges);
            exit(1);
        }
    }

    // Calculate and print strict quartiles
    std::vector<int> degrees(num_nodes, 0);
    int max_degree = 0;
    int64_t total_degree = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        int deg = (int)(g.row_offsets[i + 1] - g.row_offsets[i]);
        degrees[i] = deg;
        max_degree = std::max(max_degree, deg);
        total_degree += deg;
    }
    std::sort(degrees.begin(), degrees.end());
    int min = degrees[0];
    int q1 = degrees[num_nodes * 0.25];
    int median = degrees[num_nodes * 0.50];
    int q3 = degrees[num_nodes * 0.75];

    printf("  Avg degree: %.1f, Max degree: %d, Min degree: %d\n",
           (double)total_degree / num_nodes, max_degree, min);
    printf("  Degree Distribution -> Q1: %d | Median: %d | Q3: %d\n", q1, median, q3);

    return g;
}

inline void free_graph(CSRGraph *g)
{
    free(g->row_offsets);
    free(g->col_indices);
}

#endif // BAM_GRAPH_H
