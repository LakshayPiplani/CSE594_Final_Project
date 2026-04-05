#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "cache.cuh"
#include "three_tier_tlb.cuh"
#include "graph.h"

// ============================================================================
// Warp-Coalesced BFS matching real BaM's kernel_coalesce_ptr_pc
//
// Key difference from our previous BFS:
//   Old: 1 thread per node, each thread iterates its own edge list
//   New: 1 warp (32 threads) per node, threads cooperate on one edge list
//
// This ensures:
//   - All 32 threads in a warp are at the same instruction (no divergence)
//   - Memory accesses are aligned and coalesced
//   - __match_any_sync in Tier 2 sees all 32 threads at same call site
//   - Tier 1 TLB benefits from sequential stride pattern per thread
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define INF_DIST      0xFFFFFFFF
#define WARP_SHIFT    5
#define WARP_SIZE_    32

// ============================================================================
// BFS kernel: warp-coalesced with three-tier TLB
// Matches real BaM's kernel_coalesce_ptr_pc (main.cu lines 590-618)
//
// Each warp processes one node:
//   warpIdx = which node
//   laneIdx = which thread in the warp (0..31)
//   Threads stride through the edge list: i = shift_start + laneIdx,
//   i += 32 each iteration
//
// shift_start aligns to 16-element boundary for coalesced memory access
// ============================================================================

__global__ __launch_bounds__(128, 16)
void bfs_coalesce_3t_kernel(BamCache cache, const int* d_offsets,
                             int* d_distances, int num_nodes, int level,
                             int* d_frontier_size, const char* backing,
                            unsigned long long* d_edges_accessed) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & (WARP_SIZE_ - 1);

    Tier1Local t1;
    t1_init(&t1);

    if (warpIdx < (uint64_t)num_nodes && d_distances[warpIdx] == level) {
        const int start = d_offsets[warpIdx];
        // Align to 16-element boundary for coalesced access
        const int shift_start = start & (~0xF);
        const int end = d_offsets[warpIdx + 1];

        for (int i = shift_start + laneIdx; i < end; i += WARP_SIZE_) {
            if (i >= start) {
                atomicAdd(d_edges_accessed, 1ULL);
                const int next = t1_read(&t1, &t2, &cache, i, backing);

                if (d_distances[next] == (int)INF_DIST) {
                    // Use atomicExch to avoid duplicate discoveries
                    int prev = atomicExch(&d_distances[next], level + 1);
                    if (prev == (int)INF_DIST) {
                        atomicAdd(d_frontier_size, 1);
                    }
                }
            }
        }
    }

    t1_fini(&t1, &t2);
    t2_fini(&t2, &cache);
}

// ============================================================================
// BFS kernel: warp-coalesced Target T (baseline, no cache)
// Matches real BaM's kernel_coalesce (main.cu lines 529-556)
// ============================================================================

__global__ __launch_bounds__(128, 16)
void bfs_coalesce_target_kernel(const int* h_edges, const int* d_offsets,
                                 int* d_distances, int num_nodes, int level,
                                 int* d_frontier_size,
                                unsigned long long* d_edges_accessed) {
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & (WARP_SIZE_ - 1);

    if (warpIdx < (uint64_t)num_nodes && d_distances[warpIdx] == level) {
        const int start = d_offsets[warpIdx];
        const int shift_start = start & (~0xF);
        const int end = d_offsets[warpIdx + 1];

        for (int i = shift_start + laneIdx; i < end; i += WARP_SIZE_) {
            if (i >= start) {
                atomicAdd(d_edges_accessed, 1ULL);
                const int next = h_edges[i];

                if (d_distances[next] == (int)INF_DIST) {
                    int prev = atomicExch(&d_distances[next], level + 1);
                    if (prev == (int)INF_DIST) {
                        atomicAdd(d_frontier_size, 1);
                    }
                }
            }
        }
    }
}

// ============================================================================
// BFS kernel: our old per-node baseline (for comparison)
// ============================================================================

__global__
void bfs_per_node_target_kernel(const int* h_edges, const int* d_offsets,
                                 int* d_distances, int num_nodes, int level,
                                 int* d_frontier_size,
                                unsigned long long* d_edges_accessed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    if (d_distances[tid] != level) return;

    int start = d_offsets[tid];
    int end   = d_offsets[tid + 1];

    for (int e = start; e < end; e++) {
        atomicAdd(d_edges_accessed, 1ULL);
        int next = h_edges[e];
        if (d_distances[next] == (int)INF_DIST) {
            d_distances[next] = level + 1;
            atomicAdd(d_frontier_size, 1);
        }
    }
}

// ============================================================================
// BFS runner
// ============================================================================

struct BFSResult {
    float total_ms;
    int max_level;
    int nodes_reached;
    unsigned long long hits, misses, edges_accessed;
};

BFSResult run_bfs(const char* name, int mode,
                   const int* h_edges, int64_t num_edges,
                   const int* d_offsets, int num_nodes, int source,
                   size_t cache_size) {
    BFSResult result = {};
    uint32_t num_pages = (num_edges + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    BamCache cache;
    if (mode == 0)
        cache_init(&cache, cache_size, num_pages);

    int* d_dist;
    CHECK_CUDA(cudaMalloc(&d_dist, num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_dist, 0xFF, num_nodes * sizeof(int)));
    int src_level = 0;
    CHECK_CUDA(cudaMemcpy(d_dist + source, &src_level, sizeof(int),
                           cudaMemcpyHostToDevice));

    int* d_fs;
    CHECK_CUDA(cudaMalloc(&d_fs, sizeof(int)));

    // Variable to retrieve total edges accessed
    unsigned long long* d_edges_accessed;
    CHECK_CUDA(cudaMalloc(&d_edges_accessed, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_edges_accessed, 0, sizeof(unsigned long long)));

    // Warp-coalesced: launch num_nodes * 32 threads
    // Each warp processes one node
    int numthreads = 128;

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));

    int level = 0, fs = 1;
    while (fs > 0) {
        CHECK_CUDA(cudaMemset(d_fs, 0, sizeof(int)));

        if (mode == 0) {
            // Warp-coalesced BaM with three-tier TLB
            int total_threads = (int)((int64_t)num_nodes * WARP_SIZE_);
            int numblocks = (total_threads + numthreads - 1) / numthreads;
            bfs_coalesce_3t_kernel<<<numblocks, numthreads>>>(
                cache, d_offsets, d_dist, num_nodes, level, d_fs,
                (const char*)h_edges, d_edges_accessed);
        } else if (mode == 1) {
            // Warp-coalesced Target T
            int total_threads = (int)((int64_t)num_nodes * WARP_SIZE_);
            int numblocks = (total_threads + numthreads - 1) / numthreads;
            bfs_coalesce_target_kernel<<<numblocks, numthreads>>>(
                h_edges, d_offsets, d_dist, num_nodes, level, d_fs,
            d_edges_accessed);
        } else {
            // Per-node Target T (old style)
            int numblocks = (num_nodes + numthreads - 1) / numthreads;
            bfs_per_node_target_kernel<<<numblocks, numthreads>>>(
                h_edges, d_offsets, d_dist, num_nodes, level, d_fs,
            d_edges_accessed);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&fs, d_fs, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
        if (level > 100) { printf("  STUCK at level %d\n", level); break; }
    }

    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&result.total_ms, t0, t1));
    result.max_level = level - 1;

    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_dist, d_dist, num_nodes * sizeof(int),
                           cudaMemcpyDeviceToHost));
    result.nodes_reached = 0;
    for (int i = 0; i < num_nodes; i++)
        if (h_dist[i] != (int)INF_DIST) result.nodes_reached++;

    if (mode == 0) {
        CacheStats s = cache_get_stats(&cache);
        result.hits = s.hits;
        result.misses = s.misses;
        cache_destroy(&cache);
    }

    unsigned long long edges_accessed;
    CHECK_CUDA(cudaMemcpy(&edges_accessed, d_edges_accessed, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    result.edges_accessed = edges_accessed;

    free(h_dist);
    cudaFree(d_dist);
    cudaFree(d_fs);
    cudaFree(d_edges_accessed);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return result;
}

// ============================================================================
// Main
// ============================================================================

void generate_row(const char* method, float ms, unsigned long long edges_accessed,
    unsigned long long hits, unsigned long long misses) {
        unsigned long long bam_cache_misses = misses;
        // t1_hits + t2_t3_hits == total BaM cache hits
        unsigned long long t2_t3_hits = hits;
        unsigned long long t1_hits = edges_accessed - t2_t3_hits;

        printf("%-25s | %10.2f | %12llu | %12llu | %12llu | %12llu\n", method, ms, edges_accessed, t1_hits, t2_t3_hits, bam_cache_misses);
        
    }

int main(int argc, char** argv) {
    int scale = 18;
    if (argc > 1) scale = atoi(argv[1]);

    printf("=== Warp-Coalesced BFS Benchmark (scale=%d) ===\n\n", scale);

    CSRGraph g = generate_rmat_graph(scale, 16);
    uint32_t num_pages = (g.num_edges + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    printf("Graph: %d nodes, %lld edges (%.1f MB)\n",
           g.num_nodes, (long long)g.num_edges,
           (double)g.edges_size_bytes / (1024*1024));

    int* h_edges;
    CHECK_CUDA(cudaHostAlloc(&h_edges, g.edges_size_bytes, cudaHostAllocDefault));
    memcpy(h_edges, g.col_indices, g.edges_size_bytes);

    int* d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, g.offsets_size_bytes));
    CHECK_CUDA(cudaMemcpy(d_offsets, g.row_offsets, g.offsets_size_bytes,
                           cudaMemcpyHostToDevice));

    int source = 0, max_deg = 0;
    for (int i = 0; i < g.num_nodes; i++) {
        int deg = g.row_offsets[i+1] - g.row_offsets[i];
        if (deg > max_deg) { max_deg = deg; source = i; }
    }
    printf("Source: %d (degree %d)\n", source, max_deg);

    size_t cache_100 = (size_t)num_pages * CACHE_LINE_SIZE;
    printf("Cache: 100%% = %u slots (%.1f MB)\n\n", num_pages,
           (double)cache_100 / (1024*1024));

    

    // --- Per-node Target T (old baseline) ---
    printf("[1] Per-node Target T...\n");

    BFSResult old_target = run_bfs("PerNode Target", 2, h_edges, g.num_edges,
                                    d_offsets, g.num_nodes, source, 0);
    printf("  %.2f ms, %d levels, %d reached, %llu edges accessed\n\n",
           old_target.total_ms, old_target.max_level, old_target.nodes_reached, old_target.edges_accessed);

    // --- Warp-coalesced Target T ---
    printf("[2] Warp-coalesced Target T...\n");
    BFSResult warp_target = run_bfs("Warp Target", 1, h_edges, g.num_edges,
                                     d_offsets, g.num_nodes, source, 0);
    printf("  %.2f ms, %d levels, %d reached, %llu edges accessed\n\n",
           warp_target.total_ms, warp_target.max_level, warp_target.nodes_reached, warp_target.edges_accessed);

    // --- Warp-coalesced BaM with three-tier TLB ---
    printf("[3] Warp-coalesced BaM (3-tier TLB, 100%% cache)...\n");
    BFSResult warp_bam = run_bfs("Warp BaM", 0, h_edges, g.num_edges,
                                  d_offsets, g.num_nodes, source, cache_100);
    printf("  %.2f ms, %d levels, %d reached, %llu edges accessed\n",
           warp_bam.total_ms, warp_bam.max_level, warp_bam.nodes_reached, warp_bam.edges_accessed);
    printf("  hits=%llu misses=%llu\n\n", warp_bam.hits, warp_bam.misses);

    // --- Correctness check ---
    printf("=== Correctness ===\n");
    if (old_target.nodes_reached == warp_target.nodes_reached &&
        old_target.nodes_reached == warp_bam.nodes_reached &&
        old_target.max_level == warp_target.max_level &&
        old_target.max_level == warp_bam.max_level) {
        printf("PASSED: All reach %d nodes in %d levels\n\n",
               old_target.nodes_reached, old_target.max_level);
    } else {
        printf("MISMATCH:\n");
        printf("  PerNode Target: %d nodes, %d levels\n",
               old_target.nodes_reached, old_target.max_level);
        printf("  Warp Target:    %d nodes, %d levels\n",
               warp_target.nodes_reached, warp_target.max_level);
        printf("  Warp BaM:       %d nodes, %d levels\n\n",
               warp_bam.nodes_reached, warp_bam.max_level);
    }

    // --- Summary ---
    printf("=== Performance Summary ===\n");
    printf("%-25s | %-10s | %-12s | %-12s | %-12s | %-12s\n",
           "Method", "Time(ms)", "Edges Accessed", "T1_Hits", "T2_Hits", "DRAM (~SSD) Access");
    printf("---------------------------------------------------------------------------------------------\n");
    generate_row("Per-node Target T", old_target.total_ms, old_target.edges_accessed, old_target.hits, old_target.misses);
    generate_row("Warp-coal Target T", warp_target.total_ms, warp_target.edges_accessed, warp_target.hits, warp_target.misses);
    generate_row("Warp-coal BaM 3T-TLB", warp_bam.total_ms, warp_bam.edges_accessed, warp_bam.hits, warp_bam.misses);
    // printf("%-25s %10.2f %10s %12s %12s\n",
    //        "Per-node Target T", old_target.total_ms, "1.00x", "-", "-");
    // printf("%-25s %10.2f %10.2fx %12s %12s\n",
    //        "Warp-coal Target T", warp_target.total_ms,
    //        old_target.total_ms / warp_target.total_ms, "-", "-");
    // printf("%-25s %10.2f %10.2fx %12llu %12llu\n",
    //        "Warp-coal BaM 3T-TLB", warp_bam.total_ms,
    //        old_target.total_ms / warp_bam.total_ms,
    //        warp_bam.hits, warp_bam.misses);

    cudaFreeHost(h_edges);
    cudaFree(d_offsets);
    free_graph(&g);
    printf("\n=== Done ===\n");
    return 0;
}
