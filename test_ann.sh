cat > /tmp/ann_final.cu << 'EOF'
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>
#include "cache.cuh"
#include "three_tier_tlb.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define NB_CLUSTERS   1024
#define VEC_DIM       128
#define NPROBE        10
#define NUM_QUERIES   100

int load_to_vram(const char* fn, void* d, size_t b) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) { fprintf(stderr, "Can't open %s\n", fn); return 1; }
    char* h; cudaHostAlloc(&h, b, cudaHostAllocDefault);
    f.read(h, b); f.close();
    cudaMemcpy(d, h, b, cudaMemcpyHostToDevice);
    cudaFreeHost(h); return 0;
}

__global__
void find_top_centroids(const float* d_centroids, const float* d_query,
                        int* d_top_clusters) {
    int tid = threadIdx.x;
    __shared__ float s_dists[NB_CLUSTERS];
    __shared__ int s_ids[NB_CLUSTERS];
    for (int c = tid; c < NB_CLUSTERS; c += blockDim.x) {
        float dist = 0;
        for (int d = 0; d < VEC_DIM; d++) {
            float diff = d_query[d] - d_centroids[c * VEC_DIM + d];
            dist += diff * diff;
        }
        s_dists[c] = dist; s_ids[c] = c;
    }
    __syncthreads();
    if (tid == 0) {
        for (int p = 0; p < NPROBE; p++) {
            int best = p;
            for (int c = p+1; c < NB_CLUSTERS; c++)
                if (s_dists[c] < s_dists[best]) best = c;
            float td=s_dists[p]; s_dists[p]=s_dists[best]; s_dists[best]=td;
            int ti=s_ids[p]; s_ids[p]=s_ids[best]; s_ids[best]=ti;
            d_top_clusters[p] = s_ids[p];
        }
    }
}

__global__
void search_bam(BamCache cache, const float* d_query,
                const int* d_top_clusters, const int* d_offsets,
                int* d_result_id, float* d_result_dist,
                const char* backing) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);
    int lane = threadIdx.x;
    Tier1Local t1;
    t1_init(&t1);
    float best_dist = FLT_MAX;
    int best_id = -1;

    for (int p = 0; p < NPROBE; p++) {
        int cid = d_top_clusters[p];
        int vs = d_offsets[cid], ve = d_offsets[cid+1];
        for (int v = vs + lane; v < ve; v += blockDim.x) {
            int fb = v * VEC_DIM;
            float dist = 0;
            for (int d = 0; d < VEC_DIM; d++) {
                int raw = t1_read(&t1, &t2, &cache, fb+d, backing);
                float val = __int_as_float(raw);
                float diff = d_query[d] - val;
                dist += diff*diff;
            }
            if (dist < best_dist) { best_dist = dist; best_id = v; }
        }
        __syncthreads();
    }

    t1_fini(&t1, &t2);
    __shared__ float sd[256]; __shared__ int si[256];
    sd[lane] = best_dist; si[lane] = best_id;
    __syncthreads();
    if (lane == 0) {
        for (int i = 1; i < blockDim.x; i++)
            if (sd[i] < best_dist) { best_dist = sd[i]; best_id = si[i]; }
        *d_result_id = best_id; *d_result_dist = best_dist;
    }
    t2_fini(&t2, &cache);
}

__global__
void search_target(const float* h_vectors, const float* d_query,
                   const int* d_top_clusters, const int* d_offsets,
                   int* d_result_id, float* d_result_dist) {
    int lane = threadIdx.x;
    float best_dist = FLT_MAX;
    int best_id = -1;
    for (int p = 0; p < NPROBE; p++) {
        int cid = d_top_clusters[p];
        int vs = d_offsets[cid], ve = d_offsets[cid+1];
        for (int v = vs + lane; v < ve; v += blockDim.x) {
            const float* vec = &h_vectors[v * VEC_DIM];
            float dist = 0;
            for (int d = 0; d < VEC_DIM; d++) {
                float diff = d_query[d] - vec[d]; dist += diff*diff;
            }
            if (dist < best_dist) { best_dist = dist; best_id = v; }
        }
    }
    __shared__ float sd[256]; __shared__ int si[256];
    sd[lane] = best_dist; si[lane] = best_id;
    __syncthreads();
    if (lane == 0) {
        for (int i = 1; i < blockDim.x; i++)
            if (sd[i] < best_dist) { best_dist = sd[i]; best_id = si[i]; }
        *d_result_id = best_id; *d_result_dist = best_dist;
    }
}

int main(int argc, char** argv) {
    const char* dir = "/scratch/ljp5718/";
    if (argc > 1) dir = argv[1];

    size_t num_vectors = 1 << 20;
    size_t raw_bytes = num_vectors * VEC_DIM * sizeof(float);
    uint32_t num_pages = (num_vectors * VEC_DIM + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    printf("=== ANN Benchmark: BaM vs Target T (%d queries) ===\n\n", NUM_QUERIES);

    float* d_cent;
    cudaMalloc(&d_cent, NB_CLUSTERS * VEC_DIM * sizeof(float));
    char fn[256];
    sprintf(fn, "%scentroids.bin", dir); load_to_vram(fn, d_cent, NB_CLUSTERS*VEC_DIM*sizeof(float));

    int* d_off;
    cudaMalloc(&d_off, (NB_CLUSTERS+1)*sizeof(int));
    sprintf(fn, "%scluster_offsets.bin", dir); load_to_vram(fn, d_off, (NB_CLUSTERS+1)*sizeof(int));

    float* h_raw;
    cudaHostAlloc(&h_raw, raw_bytes, cudaHostAllocDefault);
    sprintf(fn, "%sraw_vectors.bin", dir);
    std::ifstream rf(fn, std::ios::binary); rf.read((char*)h_raw, raw_bytes); rf.close();

    // Queries = vectors at stride to avoid always hitting cluster 0
    float* d_queries;
    cudaMalloc(&d_queries, NUM_QUERIES * VEC_DIM * sizeof(float));
    {
        size_t stride = num_vectors / NUM_QUERIES;
        std::vector<float> h_queries(NUM_QUERIES * VEC_DIM);
        for (int q = 0; q < NUM_QUERIES; q++)
            memcpy(&h_queries[q*VEC_DIM], &h_raw[q*stride*VEC_DIM], VEC_DIM*sizeof(float));
        cudaMemcpy(d_queries, h_queries.data(), NUM_QUERIES*VEC_DIM*sizeof(float), cudaMemcpyHostToDevice);
    }

    printf("Data loaded. %zu vectors, %u pages\n\n", num_vectors, num_pages);

    int* d_top; cudaMalloc(&d_top, NPROBE * sizeof(int));
    int* d_rid; cudaMalloc(&d_rid, sizeof(int));
    float* d_rdist; cudaMalloc(&d_rdist, sizeof(float));

    // ---- Target T ----
    printf("[Target T] %d queries, 1 at a time...\n", NUM_QUERIES);
    {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        int matches = 0;
        for (int q = 0; q < NUM_QUERIES; q++) {
            find_top_centroids<<<1, 256>>>(d_cent, &d_queries[q*VEC_DIM], d_top);
            search_target<<<1, 256>>>(h_raw, &d_queries[q*VEC_DIM], d_top, d_off,
                                       d_rid, d_rdist);
        }

        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        // Check last result
        int rid; float rdist;
        cudaMemcpy(&rid, d_rid, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&rdist, d_rdist, sizeof(float), cudaMemcpyDeviceToHost);

        printf("  Total: %.2f ms (%.3f ms/query)\n", ms, ms/NUM_QUERIES);
        printf("  PCIe: every float read crosses PCIe\n");
        printf("  Last query result: vec %d, dist %.4f\n\n", rid, rdist);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // ---- BaM ----
    printf("[BaM] %d queries, 1 at a time, cache warms up...\n", NUM_QUERIES);
    {
        // Working set per query ≈ 1300 pages. 100 queries with overlap ≈ 10K pages.
        // Use 20K pages cache = 80MB, well within VRAM
        uint32_t cache_pages = 20000;
        size_t cache_bytes = (size_t)cache_pages * CACHE_LINE_SIZE;
        printf("  Cache: %u pages (%.1f MB)\n", cache_pages,
               (double)cache_bytes/(1024*1024));

        BamCache cache;
        cache_init(&cache, cache_bytes, num_pages);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        for (int q = 0; q < NUM_QUERIES; q++) {
            find_top_centroids<<<1, 256>>>(d_cent, &d_queries[q*VEC_DIM], d_top);
            search_bam<<<1, 256>>>(cache, &d_queries[q*VEC_DIM], d_top, d_off,
                                    d_rid, d_rdist, (const char*)h_raw);
        }

        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        CacheStats s = cache_get_stats(&cache);
        int rid; float rdist;
        cudaMemcpy(&rid, d_rid, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&rdist, d_rdist, sizeof(float), cudaMemcpyDeviceToHost);

        double pcie_mb = (double)s.misses * CACHE_LINE_SIZE / (1024*1024);
        // Each query reads ~10K vecs × 128 floats × 4B = ~5MB useful data
        double useful_per_query = 10000.0 * VEC_DIM * sizeof(float);
        double useful_total = useful_per_query * NUM_QUERIES / (1024*1024);

        printf("  Total: %.2f ms (%.3f ms/query)\n", ms, ms/NUM_QUERIES);
        printf("  Cache: hits=%llu, misses=%llu\n", s.hits, s.misses);
        printf("  PCIe traffic:  %.1f MB (%llu misses × 4KB)\n", pcie_mb, s.misses);
        printf("  Useful data:   ~%.1f MB (%d queries × ~5MB each)\n", useful_total, NUM_QUERIES);
        printf("  I/O reduction: %.1fx less PCIe than Target T\n",
               useful_total / pcie_mb);
        printf("  Last query result: vec %d, dist %.4f\n", rid, rdist);

        cache_destroy(&cache);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cudaFreeHost(h_raw);
    cudaFree(d_cent); cudaFree(d_off); cudaFree(d_queries);
    cudaFree(d_top); cudaFree(d_rid); cudaFree(d_rdist);
    printf("\n=== Done ===\n");
    return 0;
}
EOF

kill $(pgrep ann) 2>/dev/null
/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o ann_final /tmp/ann_final.cu  && timeout 300 ./ann_final /scratch/ljp5718/