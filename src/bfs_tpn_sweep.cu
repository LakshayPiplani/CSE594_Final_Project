#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include "cache.cuh"
#include "three_tier_tlb.cuh"
#include "graph.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define INF_DIST 0xFFFFFFFF

// ============================================================================
// Templatized BFS kernels
//
// TPN = Threads Per Node
//   TPN=1:  one thread per node, no edge-level parallelism
//   TPN=4:  4 threads stride through each node's edges
//   TPN=16: half-warp per node
//   TPN=32: full warp per node (real BaM style, maximum coalescing)
//   TPN=64: two warps per node
//   TPN=128: full block per node
//
// nodeIdx = tid / TPN       (which node)
// laneIdx = tid % TPN       (which sub-thread within the node's group)
//
// Warp coalescing in Tier 2 only helps when TPN >= 32, because
// __match_any_sync operates within a 32-thread warp. With TPN < 32,
// threads in the same warp process different nodes with different
// page_ids, so coalescing finds few matches.
// ============================================================================

template <int TPN>
__global__
void bfs_tpn_target(const int* h_edges, const int* d_offsets,
                     int* d_distances, int num_nodes, int level,
                     int* d_frontier_size,
                     unsigned long long* d_edges_accessed) {
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t nodeIdx = tid / TPN;
    const uint64_t laneIdx = tid % TPN;

    if (nodeIdx < (uint64_t)num_nodes && d_distances[nodeIdx] == level) {
        const int start = d_offsets[nodeIdx];
        const int end   = d_offsets[nodeIdx + 1];

        for (int i = start + (int)laneIdx; i < end; i += TPN) {
            atomicAdd(d_edges_accessed, 1ULL);
            const int next = h_edges[i];

            if (d_distances[next] == (int)INF_DIST) {
                int prev = atomicExch(&d_distances[next], level + 1);
                if (prev == (int)INF_DIST)
                    atomicAdd(d_frontier_size, 1);
            }
        }
    }
}

template <int TPN>
__global__
void bfs_tpn_bam(BamCache cache, const int* d_offsets,
                  int* d_distances, int num_nodes, int level,
                  int* d_frontier_size, const char* backing,
                  unsigned long long* d_edges_accessed) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t nodeIdx = tid / TPN;
    const uint64_t laneIdx = tid % TPN;

    Tier1Local t1;
    t1_init(&t1);

    if (nodeIdx < (uint64_t)num_nodes && d_distances[nodeIdx] == level) {
        const int start = d_offsets[nodeIdx];
        const int end   = d_offsets[nodeIdx + 1];

        for (int i = start + (int)laneIdx; i < end; i += TPN) {
            atomicAdd(d_edges_accessed, 1ULL);
            const int next = t1_read(&t1, &t2, &cache, i, backing);

            if (d_distances[next] == (int)INF_DIST) {
                int prev = atomicExch(&d_distances[next], level + 1);
                if (prev == (int)INF_DIST)
                    atomicAdd(d_frontier_size, 1);
            }
        }
    }

    t1_fini(&t1, &t2);
    t2_fini(&t2, &cache);
}

// ============================================================================
// BFS result
// ============================================================================

struct BFSResult {
    float total_ms;
    int max_level;
    int nodes_reached;
    unsigned long long hits, misses, edges_accessed;
};

// ============================================================================
// Templatized BFS runners
// ============================================================================

template <int TPN>
BFSResult run_target_tpn(const int* h_edges, const int* d_offsets,
                          int num_nodes, int source) {
    BFSResult r = {};
    int* d_dist;
    CHECK_CUDA(cudaMalloc(&d_dist, num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_dist, 0xFF, num_nodes * sizeof(int)));
    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));
    int* d_fs; CHECK_CUDA(cudaMalloc(&d_fs, sizeof(int)));
    unsigned long long* d_ea;
    CHECK_CUDA(cudaMalloc(&d_ea, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_ea, 0, sizeof(unsigned long long)));

    int blk = 128;
    int64_t total = (int64_t)num_nodes * TPN;
    int grd = (int)((total + blk - 1) / blk);

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));

    int level = 0, fs = 1;
    while (fs > 0) {
        CHECK_CUDA(cudaMemset(d_fs, 0, sizeof(int)));
        bfs_tpn_target<TPN><<<grd, blk>>>(
            h_edges, d_offsets, d_dist, num_nodes, level, d_fs, d_ea);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&fs, d_fs, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
    }

    CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&r.total_ms, t0, t1));
    r.max_level = level - 1;

    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_dist, d_dist, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_nodes; i++)
        if (h_dist[i] != (int)INF_DIST) r.nodes_reached++;
    CHECK_CUDA(cudaMemcpy(&r.edges_accessed, d_ea, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    free(h_dist); cudaFree(d_dist); cudaFree(d_fs); cudaFree(d_ea);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return r;
}

template <int TPN>
BFSResult run_bam_tpn(const int* h_edges, const int* d_offsets,
                       int num_nodes, int64_t num_edges, int source,
                       size_t cache_bytes, uint32_t num_pages) {
    BFSResult r = {};
    BamCache cache;
    cache_init(&cache, cache_bytes, num_pages);

    int* d_dist;
    CHECK_CUDA(cudaMalloc(&d_dist, num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_dist, 0xFF, num_nodes * sizeof(int)));
    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));
    int* d_fs; CHECK_CUDA(cudaMalloc(&d_fs, sizeof(int)));
    unsigned long long* d_ea;
    CHECK_CUDA(cudaMalloc(&d_ea, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_ea, 0, sizeof(unsigned long long)));

    int blk = 128;
    int64_t total = (int64_t)num_nodes * TPN;
    int grd = (int)((total + blk - 1) / blk);

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));

    int level = 0, fs = 1;
    while (fs > 0) {
        CHECK_CUDA(cudaMemset(d_fs, 0, sizeof(int)));
        bfs_tpn_bam<TPN><<<grd, blk>>>(
            cache, d_offsets, d_dist, num_nodes, level, d_fs,
            (const char*)h_edges, d_ea);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&fs, d_fs, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
        if (level > 100) { printf("  STUCK\n"); break; }
    }

    CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&r.total_ms, t0, t1));
    r.max_level = level - 1;

    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_dist, d_dist, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_nodes; i++)
        if (h_dist[i] != (int)INF_DIST) r.nodes_reached++;

    CacheStats s = cache_get_stats(&cache);
    r.hits = s.hits;
    r.misses = s.misses;
    CHECK_CUDA(cudaMemcpy(&r.edges_accessed, d_ea, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    free(h_dist); cudaFree(d_dist); cudaFree(d_fs); cudaFree(d_ea);
    cache_destroy(&cache);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return r;
}

// ============================================================================
// Dispatch: call the right template instantiation based on runtime TPN value
// ============================================================================

BFSResult dispatch_target(int tpn, const int* h_edges, const int* d_offsets,
                           int num_nodes, int source) {
    switch (tpn) {
        case 1:   return run_target_tpn<1>(h_edges, d_offsets, num_nodes, source);
        case 4:   return run_target_tpn<4>(h_edges, d_offsets, num_nodes, source);
        case 16:  return run_target_tpn<16>(h_edges, d_offsets, num_nodes, source);
        case 32:  return run_target_tpn<32>(h_edges, d_offsets, num_nodes, source);
        case 64:  return run_target_tpn<64>(h_edges, d_offsets, num_nodes, source);
        case 128: return run_target_tpn<128>(h_edges, d_offsets, num_nodes, source);
        default:
            fprintf(stderr, "Unsupported TPN=%d\n", tpn);
            exit(1);
    }
}

BFSResult dispatch_bam(int tpn, const int* h_edges, const int* d_offsets,
                        int num_nodes, int64_t num_edges, int source,
                        size_t cache_bytes, uint32_t num_pages) {
    switch (tpn) {
        case 1:   return run_bam_tpn<1>(h_edges, d_offsets, num_nodes, num_edges, source, cache_bytes, num_pages);
        case 4:   return run_bam_tpn<4>(h_edges, d_offsets, num_nodes, num_edges, source, cache_bytes, num_pages);
        case 16:  return run_bam_tpn<16>(h_edges, d_offsets, num_nodes, num_edges, source, cache_bytes, num_pages);
        case 32:  return run_bam_tpn<32>(h_edges, d_offsets, num_nodes, num_edges, source, cache_bytes, num_pages);
        case 64:  return run_bam_tpn<64>(h_edges, d_offsets, num_nodes, num_edges, source, cache_bytes, num_pages);
        case 128: return run_bam_tpn<128>(h_edges, d_offsets, num_nodes, num_edges, source, cache_bytes, num_pages);
        default:
            fprintf(stderr, "Unsupported TPN=%d\n", tpn);
            exit(1);
    }
}

void print_table_header() {
    printf("\n=======================================================================================================\n");
    printf("%-10s | %-4s | %-10s | %-12s | %-12s | %-12s | %-12s | %-6s\n",
           "Method", "TPN", "Time (ms)", "Edges Acc", "T1 Hits", "T2/T3 Hits", "DRAM Misses", "Status");
    printf("-----------+------+------------+--------------+--------------+--------------+--------------+--------\n");
}

void print_result_row(const char* method, int tpn, float ms, unsigned long long edges_accessed,
                      unsigned long long hits, unsigned long long misses, const char* status) {
    
    // Target T bypasses software cache, so we shouldn't print cache stats for it.
    if (strcmp(method, "Target_T") == 0) {
        printf("%-10s | %-4d | %10.2f | %12llu | %-12s | %-12s | %-12s | %-6s\n",
               method, tpn, ms, edges_accessed, "N/A", "N/A", "N/A", status);
        return;
    }

    unsigned long long t2_t3_hits = hits;
    unsigned long long dram_misses = misses;
    
    // Calculate T1 Hits safely to prevent underflow if counters drift
    unsigned long long t1_hits = 0;
    if (edges_accessed >= (t2_t3_hits + dram_misses)) {
        t1_hits = edges_accessed - t2_t3_hits - dram_misses;
    }

    printf("%-10s | %-4d | %10.2f | %12llu | %12llu | %12llu | %12llu | %-6s\n",
           method, tpn, ms, edges_accessed, t1_hits, t2_t3_hits, dram_misses, status);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    int scale = 20;
    float cache_perc = 1.0f;
    int tpn = 32;
    int target_edges = 2048;
    std::string graph_type = "uniform";

    // Manual named argument parsing loop
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            scale = atoi(argv[++i]); // increment 'i' to grab the value, then advance
        } 
        else if (strcmp(argv[i], "--cache_perc") == 0 && i + 1 < argc) {
            cache_perc = atof(argv[++i]);
        } 
        else if (strcmp(argv[i], "--tpn") == 0 && i + 1 < argc) {
            tpn = atoi(argv[++i]);
        } 
            else if (strcmp(argv[i], "--target_edges") == 0 && i + 1 < argc) {
            target_edges = atoi(argv[++i]);
        } 
        else if (strcmp(argv[i], "--graph_type") == 0 && i + 1 < argc) {
            graph_type = argv[++i];
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --scale <N>    : Graph scale (2^N nodes). Default: %i\n", scale);
            printf("  --cache <F>    : VRAM cache percentage (0.0 to 1.0). Default: %.2f\n", cache_perc);
            printf("  --tpn <N>      : Threads Per Node (1, 4, 16, 32, 64, 128). Default: %i\n", tpn);
            printf("  --target_edges <N>      : Degree Per Node. Default: %i\n", target_edges);
            printf("  --graph_type <String> : 'uniform' or 'rmat'. Default: %s\n", graph_type.c_str());
            exit(0);
        } 
        else {
            fprintf(stderr, "Error: Unknown argument or missing value for '%s'\n", argv[i]);
            fprintf(stderr, "Run '%s --help' for usage.\n", argv[0]);
            exit(1);
        }
    }


    if (cache_perc <= 0.0f || cache_perc > 1.0f) {
        std::cerr << "Error: cache_perc must be between 0.0 and 1.0\n";
        exit(1);
    }
    CSRGraph g;
    if (graph_type == "uniform") {
        printf("\nGENERATING GRAPH USING UNIFORM DISTRIBUTION...\n");
        g = generate_uniform_graph(scale, target_edges);
    }
    else if (graph_type == "rmat") {
        printf("\nGENERATING GRAPH USING RMAT DISTRIBUTION...\n");
        g = generate_rmat_graph(scale, target_edges);
    }
    else {
        fprintf(stderr, "Error: Invalid graph type '%s'. Must be 'uniform' or 'rmat'.\n", graph_type.c_str());
        exit(1);
    }
    
    uint32_t num_pages = (g.num_edges + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    int source = 0, max_deg = 0;
    for (int i = 0; i < g.num_nodes; i++) {
        int deg = g.row_offsets[i+1] - g.row_offsets[i];
        if (deg > max_deg) { max_deg = deg; source = i; }
    }

    uint32_t num_cache_pages = (uint32_t)(num_pages * cache_perc);
    size_t cache_bytes = (size_t)num_cache_pages * CACHE_LINE_SIZE;

    int nodes_per_block = 128 / tpn;
    int64_t total_threads = (int64_t)g.num_nodes * tpn;
    int blocks = (int)((total_threads + 128 - 1) / 128);

    
    printf("\n======================================================================\n");
    printf("                  BFS BENCHMARK CONFIGURATION SUMMARY                     \n");
    printf("======================================================================\n");
    
    printf("[1] UNDIRECTED GRAPH TOPOLOGY\n");
    printf("    Generator Type   : %s\n", graph_type.c_str());
    printf("    Nodes            : %d\n", g.num_nodes);
    printf("    Total Degree (2*Undirected edges)   : %lld\n", (long long)g.num_edges);
    printf("    Source Node      : %d (Degree: %d)\n", source, max_deg);
    printf("    Edge Array Size  : %.2f MB\n", (double)g.edges_size_bytes / (1024*1024));
    
    printf("\n[2] MEMORY & CACHE HIERARCHY\n");
    printf("    T3 VRAM Cache Line (Page Size)    : %d Bytes\n", CACHE_LINE_SIZE);
    printf("    Elements per Cache Line    : %d\n", CL_ELEMS_INT);
    printf("    Edge Array on Pinned DRAM: %u pages | %.2f MB\n", num_pages, (double)g.edges_size_bytes / (1024*1024));
    printf("    VRAM BaM Cache For Edge Array    \n");
    printf("    T3 Cache Target  : %.1f%%\n", cache_perc * 100);
    printf("    T3 Capacity      : %u data slots in global VRAM | %.2f MB\n", num_cache_pages, (double)cache_bytes / (1024*1024));
    printf("    T2 Capacity      : %d mapping slots shared per block\n", T2_NUM_ENTRIES);
    
    printf("\n[3] HARDWARE EXECUTION PARAMETERS\n");
    printf("    Threads Per Node : %d\n", tpn);
    printf("    Threads Per Block: 128\n");
    printf("    Nodes Per Block  : %d\n", nodes_per_block);
    printf("    Total Threads    : %lld (per level)\n", (long long)total_threads);
    printf("    Blocks Launched  : %d blocks (per level)\n", blocks);
    printf("    Launch Strategy  : All blocks launch every level\n");
    printf("======================================================================\n");

    int* h_edges;
    CHECK_CUDA(cudaHostAlloc(&h_edges, g.edges_size_bytes, cudaHostAllocDefault));
    memcpy(h_edges, g.col_indices, g.edges_size_bytes);

    int* d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, g.offsets_size_bytes));
    CHECK_CUDA(cudaMemcpy(d_offsets, g.row_offsets, g.offsets_size_bytes,
                           cudaMemcpyHostToDevice));




    

    print_table_header();

    int ref_reached = -1, ref_levels = -1;

    // --- Run Target T ---
    BFSResult rt = dispatch_target(tpn, h_edges, d_offsets, g.num_nodes, source);
    
    ref_reached = rt.nodes_reached;
    ref_levels = rt.max_level;
    
    print_result_row("Target_T", tpn, rt.total_ms, rt.edges_accessed, 0, 0, "REF");

    // --- Run BaM ---
    BFSResult rb = dispatch_bam(tpn, h_edges, d_offsets, g.num_nodes, g.num_edges, source, cache_bytes, num_pages);
    
    const char* correct = "FAIL";
    if (rb.nodes_reached == ref_reached && rb.max_level == ref_levels) {
        correct = "PASS";
    }

    print_result_row("BaM", tpn, rb.total_ms, rb.edges_accessed, rb.hits, rb.misses, correct);

    printf("=======================================================================================================\n");
    
    cudaFreeHost(h_edges);
    cudaFree(d_offsets);
    free_graph(&g);

    printf("\n=== Done ===\n");
    return 0;
}
