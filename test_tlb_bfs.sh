cat > /tmp/test_3t_bfs_only.cu << 'EOF'
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

int main(int argc, char** argv) {
    int scale = 18;
    if (argc > 1) scale = atoi(argv[1]);

    printf("=== Three-Tier TLB BFS Only (scale=%d) ===\n\n", scale);

    CSRGraph g = generate_rmat_graph(scale, 16);
    uint32_t num_pages = (g.num_edges + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    printf("Graph: %d nodes, %lld edges (%.1f MB)\n",
           g.num_nodes, (long long)g.num_edges,
           (double)g.edges_size_bytes/(1024*1024));
    printf("Pages: %u\n", num_pages);

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
    printf("Source: %d (degree %d)\n\n", source, max_deg);

    int blk = 256;
    int grd = (g.num_nodes + blk - 1) / blk;

    // Target T
    printf("[Target T]...\n");
    float ms_target;
    int target_reached, target_levels;
    {
        int* d_dist;
        CHECK_CUDA(cudaMalloc(&d_dist, g.num_nodes * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_dist, 0xFF, g.num_nodes * sizeof(int)));
        int zero = 0;
        CHECK_CUDA(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));
        int* d_fs; CHECK_CUDA(cudaMalloc(&d_fs, sizeof(int)));

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));
        CHECK_CUDA(cudaEventRecord(t0));

        int level = 0, fs = 1;
        while (fs > 0) {
            CHECK_CUDA(cudaMemset(d_fs, 0, sizeof(int)));
            bfs_target_kernel<<<grd, blk>>>(h_edges, d_offsets, d_dist,
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

        printf("  %.2f ms, %d levels, %d reached\n\n", ms_target, target_levels, target_reached);
        free(h_dist); cudaFree(d_dist); cudaFree(d_fs);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // Three-tier TLB with 100% cache
    printf("[3-Tier TLB, 100%% cache]...\n");
    {
        size_t cache_bytes = (size_t)num_pages * CACHE_LINE_SIZE;
        printf("  Cache: %u slots (%.1f MB)\n", num_pages,
               (double)cache_bytes/(1024*1024));

        BamCache cache;
        cache_init(&cache, cache_bytes, num_pages);

        int* d_dist;
        CHECK_CUDA(cudaMalloc(&d_dist, g.num_nodes * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_dist, 0xFF, g.num_nodes * sizeof(int)));
        int zero = 0;
        CHECK_CUDA(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));
        int* d_fs; CHECK_CUDA(cudaMalloc(&d_fs, sizeof(int)));

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));
        CHECK_CUDA(cudaEventRecord(t0));

        int level = 0, fs = 1;
        while (fs > 0) {
            CHECK_CUDA(cudaMemset(d_fs, 0, sizeof(int)));
            printf("  Level %d (frontier=%d)...\n", level, fs);
            bfs_3t_kernel<<<grd, blk>>>(cache, d_offsets, d_dist,
                                         g.num_nodes, level, d_fs,
                                         (const char*)h_edges);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("  KERNEL ERROR at level %d: %s\n", level,
                       cudaGetErrorString(err));
                break;
            }
            CHECK_CUDA(cudaMemcpy(&fs, d_fs, sizeof(int), cudaMemcpyDeviceToHost));
            level++;
            if (level > 100) { printf("  STUCK\n"); break; }
        }

        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));

        int* h_dist = (int*)malloc(g.num_nodes * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_dist, d_dist, g.num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        int reached = 0;
        for (int i = 0; i < g.num_nodes; i++)
            if (h_dist[i] != INF_DIST) reached++;

        CacheStats s = cache_get_stats(&cache);
        printf("  %.2f ms, %d levels, %d reached\n", ms, level-1, reached);
        printf("  hits=%llu misses=%llu\n", s.hits, s.misses);
        printf("  Speedup: %.2fx\n\n", ms_target / ms);

        if (reached == target_reached && (level-1) == target_levels)
            printf("  Correctness: PASSED\n");
        else
            printf("  Correctness: MISMATCH (expected %d reached %d levels, got %d reached %d levels)\n",
                   target_reached, target_levels, reached, level-1);

        free(h_dist); cudaFree(d_dist); cudaFree(d_fs);
        cache_destroy(&cache);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cudaFreeHost(h_edges);
    cudaFree(d_offsets);
    free_graph(&g);
    printf("\n=== Done ===\n");
    return 0;
}
EOF

/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o test_3t_bfs /tmp/test_3t_bfs_only.cu 2>/dev/null && timeout 120 ./test_3t_bfs