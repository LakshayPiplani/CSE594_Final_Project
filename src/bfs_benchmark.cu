#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "cache.cuh"
#include "graph.h"

// ============================================================================
// BFS Benchmark: BaM vs Target T vs Proactive Tiling
//
// All three access the edge list (col_indices) from pinned host memory.
// Node data (distances, row_offsets) lives in GPU VRAM for all three.
// The difference is HOW the edge list is accessed.
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define INF_DIST (-1)

// ============================================================================
// BFS Kernel — BaM (reads edges through software cache)
// ============================================================================

__global__
void bfs_bam_kernel(BamCache cache, const int* d_offsets,
                    int* d_distances, int num_nodes, int level,
                    int* d_frontier_size, const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    // Only process nodes at the current BFS level
    if (d_distances[tid] != level) return;

    int start = d_offsets[tid];
    int end   = d_offsets[tid + 1];

    for (int e = start; e < end; e++) {
        // Read neighbor through BaM cache
        int neighbor = cache_read_coalesced(&cache, e, backing);

        if (d_distances[neighbor] == INF_DIST) {
            d_distances[neighbor] = level + 1;
            atomicAdd(d_frontier_size, 1);
        }
    }
}

// ============================================================================
// BFS Kernel — Target T (reads edges directly from pinned host memory)
// Every edge access crosses PCIe — no caching
// ============================================================================

__global__
void bfs_target_kernel(const int* h_edges, const int* d_offsets,
                       int* d_distances, int num_nodes, int level,
                       int* d_frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    if (d_distances[tid] != level) return;

    int start = d_offsets[tid];
    int end   = d_offsets[tid + 1];

    for (int e = start; e < end; e++) {
        // Direct PCIe read from pinned host memory
        int neighbor = h_edges[e];

        if (d_distances[neighbor] == INF_DIST) {
            d_distances[neighbor] = level + 1;
            atomicAdd(d_frontier_size, 1);
        }
    }
}

// ============================================================================
// BFS Kernel — Proactive Tiling (CPU copies tiles, GPU processes)
// Only processes edges that fall within the current tile's range
// ============================================================================

__global__
void bfs_tiled_kernel(const int* d_tile, int tile_start, int tile_end,
                      const int* d_offsets, int* d_distances,
                      int num_nodes, int level, int* d_frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    if (d_distances[tid] != level) return;

    int start = d_offsets[tid];
    int end   = d_offsets[tid + 1];

    // Clamp to this tile's range
    int eff_start = max(start, tile_start);
    int eff_end   = min(end, tile_end);

    for (int e = eff_start; e < eff_end; e++) {
        // Read from the tile buffer in VRAM (tile-local index)
        int neighbor = d_tile[e - tile_start];

        if (d_distances[neighbor] == INF_DIST) {
            d_distances[neighbor] = level + 1;
            atomicAdd(d_frontier_size, 1);
        }
    }
}

// ============================================================================
// BFS Runners
// ============================================================================

struct BFSResult {
    float total_ms;       // total wall-clock time
    float load_ms;        // data loading time (Target T only)
    float compute_ms;     // kernel execution time
    int   max_level;      // BFS depth
    int   nodes_reached;  // nodes reachable from source
    int64_t bytes_transferred;  // estimated PCIe bytes
};

// --- BaM BFS ---
BFSResult run_bfs_bam(const int* h_edges, int64_t num_edges,
                       const int* d_offsets, int num_nodes, int source,
                       size_t cache_size_bytes) {
    BFSResult result = {};

    uint32_t num_logical_pages = (num_edges + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    BamCache cache;
    cache_init(&cache, cache_size_bytes, num_logical_pages);

    int* d_distances;
    CHECK_CUDA(cudaMalloc(&d_distances, num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_distances, 0xFF, num_nodes * sizeof(int)));  // all INF_DIST

    int* d_frontier_size;
    CHECK_CUDA(cudaMalloc(&d_frontier_size, sizeof(int)));

    // Set source distance = 0
    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_distances + source, &zero, sizeof(int),
                           cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (num_nodes + block - 1) / block;

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    CHECK_CUDA(cudaEventRecord(t0));

    int level = 0;
    int frontier_size = 1;

    while (frontier_size > 0) {
        CHECK_CUDA(cudaMemset(d_frontier_size, 0, sizeof(int)));

        bfs_bam_kernel<<<grid, block>>>(cache, d_offsets, d_distances,
                                         num_nodes, level, d_frontier_size,
                                         (const char*)h_edges);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(&frontier_size, d_frontier_size, sizeof(int),
                               cudaMemcpyDeviceToHost));
        level++;
    }

    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&result.total_ms, t0, t1));
    result.compute_ms = result.total_ms;
    result.max_level = level - 1;

    // Count reached nodes
    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_dist, d_distances, num_nodes * sizeof(int),
                           cudaMemcpyDeviceToHost));
    result.nodes_reached = 0;
    for (int i = 0; i < num_nodes; i++)
        if (h_dist[i] != INF_DIST) result.nodes_reached++;

    CacheStats stats = cache_get_stats(&cache);
    result.bytes_transferred = stats.misses * CACHE_LINE_SIZE;

    printf("    BaM: hits=%llu misses=%llu waits=%llu\n",
           stats.hits, stats.misses, stats.coalesced_waits);

    free(h_dist);
    cudaFree(d_distances);
    cudaFree(d_frontier_size);
    cache_destroy(&cache);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return result;
}

// --- Target T BFS ---
BFSResult run_bfs_target(const int* h_edges, int64_t num_edges,
                          const int* d_offsets, int num_nodes, int source) {
    BFSResult result = {};

    int* d_distances;
    CHECK_CUDA(cudaMalloc(&d_distances, num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_distances, 0xFF, num_nodes * sizeof(int)));

    int* d_frontier_size;
    CHECK_CUDA(cudaMalloc(&d_frontier_size, sizeof(int)));

    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_distances + source, &zero, sizeof(int),
                           cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (num_nodes + block - 1) / block;

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    CHECK_CUDA(cudaEventRecord(t0));

    int level = 0;
    int frontier_size = 1;

    while (frontier_size > 0) {
        CHECK_CUDA(cudaMemset(d_frontier_size, 0, sizeof(int)));

        bfs_target_kernel<<<grid, block>>>(h_edges, d_offsets, d_distances,
                                            num_nodes, level, d_frontier_size);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(&frontier_size, d_frontier_size, sizeof(int),
                               cudaMemcpyDeviceToHost));
        level++;
    }

    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&result.total_ms, t0, t1));
    result.compute_ms = result.total_ms;
    result.max_level = level - 1;

    // Count reached nodes and estimate bytes transferred
    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_dist, d_distances, num_nodes * sizeof(int),
                           cudaMemcpyDeviceToHost));
    result.nodes_reached = 0;
    for (int i = 0; i < num_nodes; i++)
        if (h_dist[i] != INF_DIST) result.nodes_reached++;

    // Target T: every edge access crosses PCIe (4 bytes each)
    // But we don't know exactly how many edges were accessed
    // Approximate: all edges of reached nodes
    result.bytes_transferred = num_edges * sizeof(int);  // upper bound

    free(h_dist);
    cudaFree(d_distances);
    cudaFree(d_frontier_size);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return result;
}

// --- Tiled BFS ---
BFSResult run_bfs_tiled(const int* h_edges, int64_t num_edges,
                         const int* d_offsets, int num_nodes, int source,
                         size_t tile_size_bytes) {
    BFSResult result = {};

    int tile_elems = tile_size_bytes / sizeof(int);
    int num_tiles = (num_edges + tile_elems - 1) / tile_elems;

    int* d_tile;
    CHECK_CUDA(cudaMalloc(&d_tile, tile_size_bytes));

    int* d_distances;
    CHECK_CUDA(cudaMalloc(&d_distances, num_nodes * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_distances, 0xFF, num_nodes * sizeof(int)));

    int* d_frontier_size;
    CHECK_CUDA(cudaMalloc(&d_frontier_size, sizeof(int)));

    int zero = 0;
    CHECK_CUDA(cudaMemcpy(d_distances + source, &zero, sizeof(int),
                           cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (num_nodes + block - 1) / block;

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    CHECK_CUDA(cudaEventRecord(t0));

    int level = 0;
    int frontier_size = 1;
    int64_t total_bytes_copied = 0;

    while (frontier_size > 0) {
        CHECK_CUDA(cudaMemset(d_frontier_size, 0, sizeof(int)));

        // For each tile, copy and process
        for (int t = 0; t < num_tiles; t++) {
            int tile_start = t * tile_elems;
            int tile_end = min((int64_t)(t + 1) * tile_elems, num_edges);
            int tile_actual_bytes = (tile_end - tile_start) * sizeof(int);

            // CPU copies tile to GPU
            CHECK_CUDA(cudaMemcpy(d_tile, h_edges + tile_start,
                                   tile_actual_bytes,
                                   cudaMemcpyHostToDevice));
            total_bytes_copied += tile_actual_bytes;

            bfs_tiled_kernel<<<grid, block>>>(d_tile, tile_start, tile_end,
                                               d_offsets, d_distances,
                                               num_nodes, level,
                                               d_frontier_size);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        CHECK_CUDA(cudaMemcpy(&frontier_size, d_frontier_size, sizeof(int),
                               cudaMemcpyDeviceToHost));
        level++;
    }

    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaEventElapsedTime(&result.total_ms, t0, t1));
    result.compute_ms = result.total_ms;
    result.max_level = level - 1;
    result.bytes_transferred = total_bytes_copied;

    int* h_dist = (int*)malloc(num_nodes * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_dist, d_distances, num_nodes * sizeof(int),
                           cudaMemcpyDeviceToHost));
    result.nodes_reached = 0;
    for (int i = 0; i < num_nodes; i++)
        if (h_dist[i] != INF_DIST) result.nodes_reached++;

    free(h_dist);
    cudaFree(d_distances);
    cudaFree(d_frontier_size);
    cudaFree(d_tile);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("=== BFS Benchmark: BaM vs Target T vs Tiling ===\n\n");

    // Default: scale=16 (64K nodes), edge_factor=16 (~1M edges, ~4MB)
    // Small enough to complete quickly, large enough to be meaningful
    int scale = 16;
    int edge_factor = 16;

    if (argc > 1) scale = atoi(argv[1]);
    if (argc > 2) edge_factor = atoi(argv[2]);

    // Generate graph
    CSRGraph g = generate_rmat_graph(scale, edge_factor);
    printf("Graph: %d nodes, %lld edges\n", g.num_nodes, (long long)g.num_edges);
    printf("Edge data: %.1f MB\n\n",
           (double)g.edges_size_bytes / (1024*1024));

    // Put edges in pinned host memory (simulated SSD)
    int* h_edges;
    CHECK_CUDA(cudaHostAlloc(&h_edges, g.edges_size_bytes,
                              cudaHostAllocDefault));
    memcpy(h_edges, g.col_indices, g.edges_size_bytes);

    // Put row_offsets in GPU VRAM (always resident)
    int* d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, g.offsets_size_bytes));
    CHECK_CUDA(cudaMemcpy(d_offsets, g.row_offsets, g.offsets_size_bytes,
                           cudaMemcpyHostToDevice));

    // Find a good source node (high degree for interesting BFS)
    int source = 0;
    int max_deg = 0;
    for (int i = 0; i < g.num_nodes; i++) {
        int deg = g.row_offsets[i+1] - g.row_offsets[i];
        if (deg > max_deg) {
            max_deg = deg;
            source = i;
        }
    }
    printf("Source node: %d (degree %d)\n\n", source, max_deg);

    // Cache size: 100% of edge data (ensures no eviction contention)
    size_t cache_100 = ((g.num_edges + CL_ELEMS_INT - 1) / CL_ELEMS_INT) * CACHE_LINE_SIZE;
    // Cache size: 50% of edge data (forces eviction)
    size_t cache_50 = cache_100 / 2;
    // Tile size for tiling
    size_t tile_size = cache_50;  // same amount of VRAM as 50% cache

    printf("=== Running BFS from source %d ===\n\n", source);

    // --- Target T ---
    printf("[Target T] Direct PCIe reads...\n");
    BFSResult target = run_bfs_target(h_edges, g.num_edges, d_offsets,
                                       g.num_nodes, source);
    printf("  Time: %.2f ms, Levels: %d, Reached: %d/%d\n",
           target.total_ms, target.max_level, target.nodes_reached,
           g.num_nodes);
    printf("  PCIe bytes: %.1f MB (every access)\n\n",
           (double)target.bytes_transferred / (1024*1024));

    // --- BaM 100% cache ---
    printf("[BaM 100%%] Cache = edge data size (%.1f MB)...\n",
           (double)cache_100 / (1024*1024));
    BFSResult bam_100 = run_bfs_bam(h_edges, g.num_edges, d_offsets,
                                     g.num_nodes, source, cache_100);
    printf("  Time: %.2f ms, Levels: %d, Reached: %d/%d\n",
           bam_100.total_ms, bam_100.max_level, bam_100.nodes_reached,
           g.num_nodes);
    printf("  PCIe bytes: %.1f MB (cache misses only)\n\n",
           (double)bam_100.bytes_transferred / (1024*1024));

    // --- BaM 50% cache ---
    printf("[BaM 50%%] Cache = half edge data (%.1f MB)...\n",
           (double)cache_50 / (1024*1024));
    BFSResult bam_50 = run_bfs_bam(h_edges, g.num_edges, d_offsets,
                                    g.num_nodes, source, cache_50);
    printf("  Time: %.2f ms, Levels: %d, Reached: %d/%d\n",
           bam_50.total_ms, bam_50.max_level, bam_50.nodes_reached,
           g.num_nodes);
    printf("  PCIe bytes: %.1f MB (cache misses only)\n\n",
           (double)bam_50.bytes_transferred / (1024*1024));

    // --- Tiled ---
    printf("[Tiled] Tile size = %.1f MB, %d tiles...\n",
           (double)tile_size / (1024*1024),
           (int)((g.num_edges * sizeof(int) + tile_size - 1) / tile_size));
    BFSResult tiled = run_bfs_tiled(h_edges, g.num_edges, d_offsets,
                                     g.num_nodes, source, tile_size);
    printf("  Time: %.2f ms, Levels: %d, Reached: %d/%d\n",
           tiled.total_ms, tiled.max_level, tiled.nodes_reached,
           g.num_nodes);
    printf("  PCIe bytes: %.1f MB (all tiles × all levels)\n\n",
           (double)tiled.bytes_transferred / (1024*1024));

    // --- Correctness check ---
    printf("=== Correctness ===\n");
    if (target.nodes_reached == bam_100.nodes_reached &&
        target.nodes_reached == bam_50.nodes_reached &&
        target.nodes_reached == tiled.nodes_reached &&
        target.max_level == bam_100.max_level &&
        target.max_level == bam_50.max_level &&
        target.max_level == tiled.max_level) {
        printf("  PASSED: All methods reach %d nodes in %d levels\n\n",
               target.nodes_reached, target.max_level);
    } else {
        printf("  MISMATCH:\n");
        printf("    Target:   %d nodes, %d levels\n", target.nodes_reached, target.max_level);
        printf("    BaM 100%%: %d nodes, %d levels\n", bam_100.nodes_reached, bam_100.max_level);
        printf("    BaM 50%%:  %d nodes, %d levels\n", bam_50.nodes_reached, bam_50.max_level);
        printf("    Tiled:    %d nodes, %d levels\n\n", tiled.nodes_reached, tiled.max_level);
    }

    // --- Summary ---
    printf("=== Performance Summary ===\n");
    printf("%-15s %10s %10s %12s %10s\n",
           "Method", "Time(ms)", "Speedup", "PCIe(MB)", "I/O Amp");
    printf("---------------------------------------------------------------\n");
    printf("%-15s %10.2f %10s %12.1f %10s\n",
           "Target T", target.total_ms, "1.00x",
           (double)target.bytes_transferred/(1024*1024), "-");
    printf("%-15s %10.2f %10.2fx %12.1f %10.1fx\n",
           "BaM 100%", bam_100.total_ms,
           target.total_ms / bam_100.total_ms,
           (double)bam_100.bytes_transferred/(1024*1024),
           (double)target.bytes_transferred / bam_100.bytes_transferred);
    printf("%-15s %10.2f %10.2fx %12.1f %10.1fx\n",
           "BaM 50%", bam_50.total_ms,
           target.total_ms / bam_50.total_ms,
           (double)bam_50.bytes_transferred/(1024*1024),
           (double)target.bytes_transferred / bam_50.bytes_transferred);
    printf("%-15s %10.2f %10.2fx %12.1f %10.1fx\n",
           "Tiled", tiled.total_ms,
           target.total_ms / tiled.total_ms,
           (double)tiled.bytes_transferred/(1024*1024),
           (double)target.bytes_transferred / tiled.bytes_transferred);

    // Cleanup
    cudaFreeHost(h_edges);
    cudaFree(d_offsets);
    free_graph(&g);

    printf("\n=== Done ===\n");
    return 0;
}
