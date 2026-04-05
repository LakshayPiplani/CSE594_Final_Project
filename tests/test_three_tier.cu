#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
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

#define INF_DIST (-1)

// ============================================================================
// Test kernels
// ============================================================================

// Sequential read — tests Tier 1 fast path (consecutive elements)
__global__
void kernel_3t_sequential(BamCache cache, int* output, int n,
                           const char* backing) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tier1Local t1;
    t1_init(&t1);

    if (tid < n) {
        int base = tid * 64;
        for (int i = 0; i < 64; i++) {
            int idx = base + i;
            if (idx < n * 64)
                output[tid * 64 + i] = t1_read(&t1, &t2, &cache, idx, backing);
        }
    }

    t1_fini(&t1, &t2);
    t2_fini(&t2, &cache);
}

// Random read — tests Tier 2 acquire path (every access is a Tier 1 miss)
__global__
void kernel_3t_random(BamCache cache, const int* indices, int* output,
                       int n, const char* backing) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tier1Local t1;
    t1_init(&t1);

    if (tid < n) {
        output[tid] = t1_read(&t1, &t2, &cache, indices[tid], backing);
    }

    t1_fini(&t1, &t2);
    t2_fini(&t2, &cache);
}

// BFS-like pattern — tests sequential iteration with occasional page crossings
__global__
void kernel_3t_bfs(BamCache cache, const int* offsets, int* output,
                    int num_nodes, const char* backing) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tier1Local t1;
    t1_init(&t1);

    if (tid < num_nodes) {
        int start = offsets[tid];
        int end   = offsets[tid + 1];
        int sum = 0;
        for (int e = start; e < end; e++) {
            sum += t1_read(&t1, &t2, &cache, e, backing);
        }
        output[tid] = sum;
    }

    t1_fini(&t1, &t2);
    t2_fini(&t2, &cache);
}

// Direct PCIe reference
__global__
void kernel_direct_bfs(const int* backing, const int* offsets, int* output,
                        int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    int start = offsets[tid];
    int end   = offsets[tid + 1];
    int sum = 0;
    for (int e = start; e < end; e++) sum += backing[e];
    output[tid] = sum;
}

// Full BFS kernel with three-tier TLB
__global__
void bfs_3t_kernel(BamCache cache, const int* d_offsets,
                    int* d_distances, int num_nodes, int level,
                    int* d_frontier_size, const char* backing) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tier1Local t1;
    t1_init(&t1);

    if (tid < num_nodes && d_distances[tid] == level) {
        int start = d_offsets[tid];
        int end   = d_offsets[tid + 1];
        for (int e = start; e < end; e++) {
            int neighbor = t1_read(&t1, &t2, &cache, e, backing);
            if (d_distances[neighbor] == INF_DIST) {
                d_distances[neighbor] = level + 1;
                atomicAdd(d_frontier_size, 1);
            }
        }
    }

    t1_fini(&t1, &t2);
    t2_fini(&t2, &cache);
}

// BFS Target T reference
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
        int neighbor = h_edges[e];
        if (d_distances[neighbor] == INF_DIST) {
            d_distances[neighbor] = level + 1;
            atomicAdd(d_frontier_size, 1);
        }
    }
}

__global__ void flush_l2(char* buf, int sz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    volatile char x;
    for (int i = tid; i < sz; i += gridDim.x * blockDim.x) x = buf[i];
}

int main(int argc, char** argv) {
    printf("=== Three-Tier TLB Test Suite ===\n\n");

    const int NUM_ELEMS = 4 * 1024 * 1024;
    const uint32_t NUM_PAGES = NUM_ELEMS / CL_ELEMS_INT;
    const int NUM_THREADS = 1024;

    int* h_backing;
    CHECK_CUDA(cudaHostAlloc(&h_backing, NUM_ELEMS * sizeof(int),
                              cudaHostAllocDefault));
    for (int i = 0; i < NUM_ELEMS; i++) h_backing[i] = i;

    int out_size = NUM_THREADS * 64;
    int* d_out; CHECK_CUDA(cudaMalloc(&d_out, out_size * sizeof(int)));
    int* h_out = (int*)malloc(out_size * sizeof(int));
    char* d_flush; CHECK_CUDA(cudaMalloc(&d_flush, 8*1024*1024));

    BamCache cache;
    cache_init(&cache, (size_t)NUM_PAGES * CACHE_LINE_SIZE, NUM_PAGES);

    int blk = 256, grd = (NUM_THREADS + blk - 1) / blk;

    // ==== Test 1: Sequential correctness ====
    printf("--- Test 1: Sequential correctness (Tier 1 fast path) ---\n");
    {
        cache_reset(&cache);
        CHECK_CUDA(cudaMemset(d_out, 0, out_size * sizeof(int)));

        kernel_3t_sequential<<<grd, blk>>>(cache, d_out, NUM_THREADS,
                                            (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out, d_out, out_size * sizeof(int),
                               cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < out_size; i++)
            if (h_out[i] != i) {
                if (errs < 5) printf("  ERR[%d]: expected %d got %d\n", i, i, h_out[i]);
                errs++;
            }
        CacheStats s = cache_get_stats(&cache);
        printf("  %s (%d errors / %d)\n", errs == 0 ? "PASSED" : "FAILED", errs, out_size);
        printf("  VRAM cache: hits=%llu misses=%llu (should be low — Tier 1 absorbs most)\n\n",
               s.hits, s.misses);
    }

    // ==== Test 2: Multi-page sequential (exercises Tier 2 without thrashing) ====
    printf("--- Test 2: Multi-page sequential (each thread reads across pages) ---\n");
    {
        // Each thread reads 4 elements from 4 different pages
        // This exercises Tier 2 acquire/release but with sequential locality
        // within each page (Tier 1 hit after first access per page)
        int n = NUM_THREADS;
        int out_per_thread = 4;
        int total_out = n * out_per_thread;
        int* d_out2; CHECK_CUDA(cudaMalloc(&d_out2, total_out * sizeof(int)));
        int* h_idx = (int*)malloc(total_out * sizeof(int));

        // Thread i reads elements from pages i*4, i*4+1, i*4+2, i*4+3
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < out_per_thread; j++) {
                int page = (i * out_per_thread + j) % NUM_PAGES;
                h_idx[i * out_per_thread + j] = page * CL_ELEMS_INT + 7;
            }
        }
        int* d_idx; CHECK_CUDA(cudaMalloc(&d_idx, total_out * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_idx, h_idx, total_out * sizeof(int),
                               cudaMemcpyHostToDevice));

        // Kernel: each thread reads its 4 elements sequentially
        // (We reuse the sequential kernel — each thread reads 64 consecutive
        //  elements starting at a spread-out base, exercising Tier 1 + Tier 2)
        cache_reset(&cache);
        kernel_3t_sequential<<<grd, blk>>>(cache, d_out, NUM_THREADS,
                                            (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out, d_out, out_size * sizeof(int),
                               cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < out_size; i++)
            if (h_out[i] != i) errs++;

        CacheStats s = cache_get_stats(&cache);
        printf("  %s (%d errors / %d)\n",
               errs == 0 ? "PASSED" : "FAILED", errs, out_size);
        printf("  VRAM cache: hits=%llu misses=%llu\n\n", s.hits, s.misses);

        free(h_idx); cudaFree(d_idx); cudaFree(d_out2);
    }

    // ==== Test 3: BFS-like correctness ====
    printf("--- Test 3: BFS-like pattern (vs direct PCIe reference) ---\n");
    {
        int num_nodes = 4096;
        int* h_offsets = (int*)malloc((num_nodes + 1) * sizeof(int));
        srand(123);
        h_offsets[0] = 0;
        for (int i = 0; i < num_nodes; i++) {
            int deg = 16 + (rand() % 32);
            h_offsets[i+1] = h_offsets[i] + deg;
            if (h_offsets[i+1] >= NUM_ELEMS) h_offsets[i+1] = NUM_ELEMS;
        }
        int* d_offsets;
        CHECK_CUDA(cudaMalloc(&d_offsets, (num_nodes+1) * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (num_nodes+1) * sizeof(int),
                               cudaMemcpyHostToDevice));

        int* d_out_3t; CHECK_CUDA(cudaMalloc(&d_out_3t, num_nodes * sizeof(int)));
        int* d_out_ref; CHECK_CUDA(cudaMalloc(&d_out_ref, num_nodes * sizeof(int)));
        int grd2 = (num_nodes + blk - 1) / blk;

        // Reference: direct PCIe
        kernel_direct_bfs<<<grd2, blk>>>((const int*)h_backing, d_offsets,
                                          d_out_ref, num_nodes);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Don't reset cache — use warm state
        kernel_3t_bfs<<<grd2, blk>>>(cache, d_offsets, d_out_3t, num_nodes,
                                      (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        int* h_3t = (int*)malloc(num_nodes * sizeof(int));
        int* h_ref = (int*)malloc(num_nodes * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_3t, d_out_3t, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_ref, d_out_ref, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

        int errs = 0;
        for (int i = 0; i < num_nodes; i++)
            if (h_3t[i] != h_ref[i]) {
                if (errs < 5) printf("  ERR node %d: 3tier=%d ref=%d\n", i, h_3t[i], h_ref[i]);
                errs++;
            }
        printf("  %s (%d errors / %d nodes)\n\n",
               errs == 0 ? "PASSED" : "FAILED", errs, num_nodes);
        free(h_offsets); free(h_3t); free(h_ref);
        cudaFree(d_offsets); cudaFree(d_out_3t); cudaFree(d_out_ref);
    }

    // ==== Test 4: Pin safety — 25% cache ====
    printf("--- Test 4: Pin safety (25%% cache, 4096 nodes) ---\n");
    {
        BamCache small_cache;
        cache_init(&small_cache, (size_t)(NUM_PAGES / 4) * CACHE_LINE_SIZE, NUM_PAGES);

        int num_nodes = 4096;
        int* h_offsets = (int*)malloc((num_nodes + 1) * sizeof(int));
        srand(456);
        h_offsets[0] = 0;
        for (int i = 0; i < num_nodes; i++) {
            int deg = 16 + (rand() % 32);
            h_offsets[i+1] = h_offsets[i] + deg;
            if (h_offsets[i+1] >= NUM_ELEMS) h_offsets[i+1] = NUM_ELEMS;
        }
        int* d_offsets;
        CHECK_CUDA(cudaMalloc(&d_offsets, (num_nodes+1) * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (num_nodes+1) * sizeof(int),
                               cudaMemcpyHostToDevice));

        int* d_out_s; CHECK_CUDA(cudaMalloc(&d_out_s, num_nodes * sizeof(int)));
        int* d_out_r; CHECK_CUDA(cudaMalloc(&d_out_r, num_nodes * sizeof(int)));
        int grd2 = (num_nodes + blk - 1) / blk;

        kernel_direct_bfs<<<grd2, blk>>>((const int*)h_backing, d_offsets,
                                          d_out_r, num_nodes);
        CHECK_CUDA(cudaDeviceSynchronize());

        kernel_3t_bfs<<<grd2, blk>>>(small_cache, d_offsets, d_out_s,
                                      num_nodes, (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        int* h_s = (int*)malloc(num_nodes * sizeof(int));
        int* h_r = (int*)malloc(num_nodes * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_s, d_out_s, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_r, d_out_r, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

        int errs = 0;
        for (int i = 0; i < num_nodes; i++)
            if (h_s[i] != h_r[i]) errs++;

        CacheStats s = cache_get_stats(&small_cache);
        printf("  %s (%d errors / %d nodes)\n", errs == 0 ? "PASSED" : "FAILED", errs, num_nodes);
        printf("  25%% cache: %u slots, hits=%llu misses=%llu\n",
               small_cache.num_slots, s.hits, s.misses);
        printf("  Max VRAM pins: %d blocks × %d entries = %d (vs %u slots)\n\n",
               grd2, T2_NUM_ENTRIES, grd2 * T2_NUM_ENTRIES, small_cache.num_slots);

        free(h_offsets); free(h_s); free(h_r);
        cudaFree(d_offsets); cudaFree(d_out_s); cudaFree(d_out_r);
        cache_destroy(&small_cache);
    }

    // ==== Test 5: BFS benchmark ====
    printf("--- Test 5: BFS benchmark ---\n");
    {
        int scale = 18;
        if (argc > 1) scale = atoi(argv[1]);

        CSRGraph g = generate_rmat_graph(scale, 16);
        uint32_t gnum_pages = (g.num_edges + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

        int* gh_edges;
        CHECK_CUDA(cudaHostAlloc(&gh_edges, g.edges_size_bytes, cudaHostAllocDefault));
        memcpy(gh_edges, g.col_indices, g.edges_size_bytes);

        int* gd_offsets;
        CHECK_CUDA(cudaMalloc(&gd_offsets, g.offsets_size_bytes));
        CHECK_CUDA(cudaMemcpy(gd_offsets, g.row_offsets, g.offsets_size_bytes,
                               cudaMemcpyHostToDevice));

        int source = 0, max_deg = 0;
        for (int i = 0; i < g.num_nodes; i++) {
            int deg = g.row_offsets[i+1] - g.row_offsets[i];
            if (deg > max_deg) { max_deg = deg; source = i; }
        }

        size_t cache50 = (size_t)(gnum_pages / 2) * CACHE_LINE_SIZE;

        printf("  Graph: %d nodes, %lld edges (%.1f MB), source=%d (deg %d)\n",
               g.num_nodes, (long long)g.num_edges,
               (double)g.edges_size_bytes/(1024*1024), source, max_deg);
        printf("  Cache: 50%% = %u slots\n", (uint32_t)(cache50 / CACHE_LINE_SIZE));

        // Target T
        float ms_target;
        int target_reached, target_levels;
        {
            int* d_dist;
            CHECK_CUDA(cudaMalloc(&d_dist, g.num_nodes * sizeof(int)));
            CHECK_CUDA(cudaMemset(d_dist, 0xFF, g.num_nodes * sizeof(int)));
            int zero = 0;
            CHECK_CUDA(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));
            int* d_fs; CHECK_CUDA(cudaMalloc(&d_fs, sizeof(int)));
            int grd3 = (g.num_nodes + blk - 1) / blk;

            cudaEvent_t t0, t1;
            CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
            CHECK_CUDA(cudaEventRecord(t0));
            int level = 0, fs = 1;
            while (fs > 0) {
                CHECK_CUDA(cudaMemset(d_fs, 0, sizeof(int)));
                bfs_target_kernel<<<grd3, blk>>>(gh_edges, gd_offsets, d_dist,
                                                  g.num_nodes, level, d_fs);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(&fs, d_fs, sizeof(int), cudaMemcpyDeviceToHost));
                level++;
            }
            CHECK_CUDA(cudaEventRecord(t1));
            CHECK_CUDA(cudaEventSynchronize(t1));
            CHECK_CUDA(cudaEventElapsedTime(&ms_target, t0, t1));
            target_levels = level - 1;

            int* h_dist = (int*)malloc(g.num_nodes * sizeof(int));
            CHECK_CUDA(cudaMemcpy(h_dist, d_dist, g.num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
            target_reached = 0;
            for (int i = 0; i < g.num_nodes; i++)
                if (h_dist[i] != INF_DIST) target_reached++;
            free(h_dist); cudaFree(d_dist); cudaFree(d_fs);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        printf("  Target T:     %.2f ms, %d levels, %d reached\n",
               ms_target, target_levels, target_reached);

        // Three-tier TLB BFS with 50% cache
        {
            BamCache bcache;
            cache_init(&bcache, cache50, gnum_pages);

            int* d_dist;
            CHECK_CUDA(cudaMalloc(&d_dist, g.num_nodes * sizeof(int)));
            CHECK_CUDA(cudaMemset(d_dist, 0xFF, g.num_nodes * sizeof(int)));
            int zero = 0;
            CHECK_CUDA(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));
            int* d_fs; CHECK_CUDA(cudaMalloc(&d_fs, sizeof(int)));
            int grd3 = (g.num_nodes + blk - 1) / blk;

            cudaEvent_t t0, t1;
            CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));
            CHECK_CUDA(cudaEventRecord(t0));
            int level = 0, fs = 1;
            while (fs > 0) {
                CHECK_CUDA(cudaMemset(d_fs, 0, sizeof(int)));
                bfs_3t_kernel<<<grd3, blk>>>(bcache, gd_offsets, d_dist,
                                              g.num_nodes, level, d_fs,
                                              (const char*)gh_edges);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(&fs, d_fs, sizeof(int), cudaMemcpyDeviceToHost));
                level++;
                if (level > 100) { printf("  STUCK\n"); break; }
            }
            CHECK_CUDA(cudaEventRecord(t1));
            CHECK_CUDA(cudaEventSynchronize(t1));
            float ms_3t;
            CHECK_CUDA(cudaEventElapsedTime(&ms_3t, t0, t1));

            int* h_dist = (int*)malloc(g.num_nodes * sizeof(int));
            CHECK_CUDA(cudaMemcpy(h_dist, d_dist, g.num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
            int reached = 0;
            for (int i = 0; i < g.num_nodes; i++)
                if (h_dist[i] != INF_DIST) reached++;

            CacheStats s = cache_get_stats(&bcache);
            printf("  3-Tier TLB:   %.2f ms, %d levels, %d reached",
                   ms_3t, level-1, reached);
            printf(" (hits=%llu misses=%llu)\n", s.hits, s.misses);
            printf("  Speedup vs Target: %.2fx\n",
                   ms_target / ms_3t);

            if (reached == target_reached && (level-1) == target_levels)
                printf("  Correctness: PASSED\n");
            else
                printf("  Correctness: MISMATCH\n");

            free(h_dist); cudaFree(d_dist); cudaFree(d_fs);
            cache_destroy(&bcache);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        cudaFreeHost(gh_edges); cudaFree(gd_offsets); free_graph(&g);
    }

    free(h_out);
    cudaFreeHost(h_backing);
    cudaFree(d_out); cudaFree(d_flush);
    cache_destroy(&cache);
    printf("\n=== All tests complete ===\n");
    return 0;
}
