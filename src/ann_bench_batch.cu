#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <string>
#include <cstdlib>
#include <cfloat>
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

// ============================================================================
// Constants
// ============================================================================

#define NB_CLUSTERS   1024
#define VEC_DIM       128
#define NPROBE        10       // search top-10 closest clusters
#define NUM_QUERIES   1000

// Each vector = 128 floats = 512 bytes
// Each 4KB cache page holds 1024 floats = 8 vectors
// Vector i occupies floats [i*128 .. i*128+127], entirely within page i/8
#define FLOATS_PER_PAGE  CL_ELEMS_INT   // 1024 (reusing the int constant, works for float too)
#define VECS_PER_PAGE    (FLOATS_PER_PAGE / VEC_DIM)  // 8

// ============================================================================
// Load binary file into VRAM
// ============================================================================

int load_to_vram(const std::string& filename, void* d_ptr, size_t bytes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    char* h_buf;
    CHECK_CUDA(cudaHostAlloc(&h_buf, bytes, cudaHostAllocDefault));
    file.read(h_buf, bytes);
    if ((size_t)file.gcount() != bytes) {
        std::cerr << "Read error: expected " << bytes << " got " << file.gcount() << std::endl;
        cudaFreeHost(h_buf);
        return 1;
    }
    file.close();
    CHECK_CUDA(cudaMemcpy(d_ptr, h_buf, bytes, cudaMemcpyHostToDevice));
    cudaFreeHost(h_buf);
    return 0;
}

// ============================================================================
// Load binary file into pinned host memory (simulated SSD)
// ============================================================================

int load_to_pinned(const std::string& filename, void** h_ptr, size_t bytes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    CHECK_CUDA(cudaHostAlloc(h_ptr, bytes, cudaHostAllocDefault));
    file.read((char*)*h_ptr, bytes);
    if ((size_t)file.gcount() != bytes) {
        std::cerr << "Read error: expected " << bytes << " got " << file.gcount() << std::endl;
        return 1;
    }
    file.close();
    return 0;
}

// ============================================================================
// Kernel: Find top-NPROBE closest centroids for each query
// ============================================================================

__global__
void find_top_centroids(const float* d_centroids, const float* d_queries,
                        int* d_top_clusters, int num_queries) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_queries) return;

    const float* query = &d_queries[qid * VEC_DIM];

    // Compute distance to all centroids, keep top NPROBE
    float top_dists[NPROBE];
    int   top_ids[NPROBE];
    for (int i = 0; i < NPROBE; i++) {
        top_dists[i] = FLT_MAX;
        top_ids[i] = -1;
    }

    for (int c = 0; c < NB_CLUSTERS; c++) {
        const float* centroid = &d_centroids[c * VEC_DIM];
        float dist = 0.0f;
        for (int d = 0; d < VEC_DIM; d++) {
            float diff = query[d] - centroid[d];
            dist += diff * diff;
        }

        // Insert into top-NPROBE if closer than worst
        if (dist < top_dists[NPROBE - 1]) {
            top_dists[NPROBE - 1] = dist;
            top_ids[NPROBE - 1] = c;
            // Bubble sort to maintain order
            for (int i = NPROBE - 1; i > 0; i--) {
                if (top_dists[i] < top_dists[i-1]) {
                    float td = top_dists[i]; top_dists[i] = top_dists[i-1]; top_dists[i-1] = td;
                    int ti = top_ids[i]; top_ids[i] = top_ids[i-1]; top_ids[i-1] = ti;
                } else break;
            }
        }
    }

    // Write top cluster IDs for this query
    for (int i = 0; i < NPROBE; i++) {
        d_top_clusters[qid * NPROBE + i] = top_ids[i];
    }
}

// ============================================================================
// Kernel: Search clusters using BaM three-tier TLB
//
// Each query searches its NPROBE closest clusters.
// For each cluster, iterate through all vectors in that cluster,
// compute L2 distance, track the closest vector.
//
// The raw vector data lives in pinned DRAM, accessed through BaM cache.
// Access pattern: sequential within each cluster (Tier 1 absorbs most reads)
//
// We use warp-coalesced iteration within each cluster.
// ============================================================================

__global__
void search_clusters_bam(BamCache cache, const float* d_queries,
                          const int* d_top_clusters, const int* d_offsets,
                          int* d_result_ids, float* d_result_dists,
                          int num_queries, const char* backing) {
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    int qid = blockIdx.x;  // one block per query
    if (qid >= num_queries) { t2_fini(&t2, &cache); return; }

    int lane = threadIdx.x;
    const float* query = &d_queries[qid * VEC_DIM];

    // Each thread tracks its own best match
    float best_dist = FLT_MAX;
    int   best_id   = -1;

    Tier1Local t1;
    t1_init(&t1);

    // Search each of the NPROBE clusters
    for (int p = 0; p < NPROBE; p++) {
        int cluster_id = d_top_clusters[qid * NPROBE + p];
        if (cluster_id < 0) continue;

        int vec_start = d_offsets[cluster_id];
        int vec_end   = d_offsets[cluster_id + 1];
        int cluster_size = vec_end - vec_start;

        // Each thread handles a strided subset of vectors in this cluster
        for (int v = lane; v < cluster_size; v += blockDim.x) {
            int vec_id = vec_start + v;

            // Read 128 floats for this vector through BaM cache
            // The raw_vectors array is laid out as float[num_vectors * 128]
            // Vector vec_id starts at float index vec_id * 128
            int float_base = vec_id * VEC_DIM;

            float dist = 0.0f;
            for (int d = 0; d < VEC_DIM; d++) {
                // Read one float at a time through the cache
                // Sequential access within a vector = Tier 1 hits after first float
                int fidx = float_base + d;
                // Read as int from cache, reinterpret as float
                int raw = t1_read(&t1, &t2, &cache, fidx, backing);
                float val = __int_as_float(raw);
                float diff = query[d] - val;
                dist += diff * diff;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_id = vec_id;
            }
        }
    }

    t1_fini(&t1, &t2);

    // Warp reduction to find minimum across threads
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
        int   other_id   = __shfl_down_sync(0xFFFFFFFF, best_id, offset);
        if (other_dist < best_dist) {
            best_dist = other_dist;
            best_id = other_id;
        }
    }

    // Cross-warp reduction using shared memory
    __shared__ float s_dists[32];  // one per warp
    __shared__ int   s_ids[32];
    int warp_id = threadIdx.x / 32;
    int warp_lane = threadIdx.x % 32;

    if (warp_lane == 0) {
        s_dists[warp_id] = best_dist;
        s_ids[warp_id] = best_id;
    }
    __syncthreads();

    // First warp reduces across all warps
    // if (warp_id == 0 && warp_lane < (blockDim.x / 32)) {
    //     best_dist = s_dists[warp_lane];
    //     best_id = s_ids[warp_lane];
    //     for (int offset = 16; offset > 0; offset /= 2) {
    //         float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
    //         int   other_id   = __shfl_down_sync(0xFFFFFFFF, best_id, offset);
    //         if (other_dist < best_dist) {
    //             best_dist = other_dist;
    //             best_id = other_id;
    //         }
    //     }
    //     if (warp_lane == 0) {
    //         d_result_ids[qid] = best_id;
    //         d_result_dists[qid] = best_dist;
    //     }
    // }
    // Let a SINGLE thread finish the job
    if (threadIdx.x == 0) {
        // Start by assuming Warp 0's answer is the best
        float absolute_best_dist = s_dists[0];
        int   absolute_best_id   = s_ids[0];
        
        // Linearly scan the remaining warps (Warp 1, 2, and 3)
        int num_warps = blockDim.x / 32;
        for (int i = 1; i < num_warps; i++) {
            if (s_dists[i] < absolute_best_dist) {
                absolute_best_dist = s_dists[i];
                absolute_best_id   = s_ids[i];
            }
        }
        
        // Write the final answer to global memory
        d_result_ids[qid] = absolute_best_id;
        d_result_dists[qid] = absolute_best_dist;
    }

    t2_fini(&t2, &cache);
}

// ============================================================================
// Kernel: Search clusters using Target T (direct PCIe reads)
// ============================================================================

__global__
void search_clusters_target(const float* h_vectors, const float* d_queries,
                             const int* d_top_clusters, const int* d_offsets,
                             int* d_result_ids, float* d_result_dists,
                             int num_queries) {
    int qid = blockIdx.x;
    if (qid >= num_queries) return;

    int lane = threadIdx.x;
    const float* query = &d_queries[qid * VEC_DIM];

    float best_dist = FLT_MAX;
    int   best_id   = -1;

    for (int p = 0; p < NPROBE; p++) {
        int cluster_id = d_top_clusters[qid * NPROBE + p];
        if (cluster_id < 0) continue;

        int vec_start = d_offsets[cluster_id];
        int vec_end   = d_offsets[cluster_id + 1];
        int cluster_size = vec_end - vec_start;

        for (int v = lane; v < cluster_size; v += blockDim.x) {
            int vec_id = vec_start + v;
            const float* vec = &h_vectors[vec_id * VEC_DIM];

            float dist = 0.0f;
            for (int d = 0; d < VEC_DIM; d++) {
                float diff = query[d] - vec[d];
                dist += diff * diff;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_id = vec_id;
            }
        }
    }

    // Same warp + block reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
        int   other_id   = __shfl_down_sync(0xFFFFFFFF, best_id, offset);
        if (other_dist < best_dist) {
            best_dist = other_dist;
            best_id = other_id;
        }
    }

    __shared__ float s_dists[32];
    __shared__ int   s_ids[32];
    int warp_id = threadIdx.x / 32;
    int warp_lane = threadIdx.x % 32;

    if (warp_lane == 0) {
        s_dists[warp_id] = best_dist;
        s_ids[warp_id] = best_id;
    }
    __syncthreads();

    if (warp_id == 0 && warp_lane < (blockDim.x / 32)) {
        best_dist = s_dists[warp_lane];
        best_id = s_ids[warp_lane];
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
            int   other_id   = __shfl_down_sync(0xFFFFFFFF, best_id, offset);
            if (other_dist < best_dist) {
                best_dist = other_dist;
                best_id = other_id;
            }
        }
        if (warp_lane == 0) {
            d_result_ids[qid] = best_id;
            d_result_dists[qid] = best_dist;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/scratch/ljp5718/";
    if (argc > 1) data_dir = std::string(argv[1]) + "/";

    size_t num_vectors = 1 << 20;  // must match what knn_data generated
    if (argc > 2) num_vectors = 1ULL << atoi(argv[2]);

    float cache_perc = 1.0f;
    if (argc > 3) cache_perc = atof(argv[3]);
    if (cache_perc <= 0.0f || cache_perc > 1.0f) {
        std::cerr << "Error: cache_perc must be between 0.0 and 1.0 (got " << cache_perc << ")\n";
        exit(1);
    }

    printf("=== ANN Benchmark: BaM vs Target T ===\n\n");
    printf("Vectors: %zu, Dim: %d, Clusters: %d, nprobe: %d, Queries: %d\n",
           num_vectors, VEC_DIM, NB_CLUSTERS, NPROBE, NUM_QUERIES);

    size_t raw_vectors_bytes = num_vectors * VEC_DIM * sizeof(float);
    size_t centroid_bytes = NB_CLUSTERS * VEC_DIM * sizeof(float);
    size_t offsets_bytes = (NB_CLUSTERS + 1) * sizeof(int);

    printf("Raw vectors: %.1f MB\n\n", (double)raw_vectors_bytes / (1024*1024));

    // ============================
    // Load data
    // ============================

    // Centroids → VRAM (small, always resident)
    float* d_centroids;
    CHECK_CUDA(cudaMalloc(&d_centroids, centroid_bytes));
    if (load_to_vram(data_dir + "centroids.bin", d_centroids, centroid_bytes))
        return 1;
    printf("Loaded centroids to VRAM (%.1f KB)\n", (double)centroid_bytes/1024);

    // Cluster offsets → VRAM (small)
    int* d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, offsets_bytes));
    if (load_to_vram(data_dir + "cluster_offsets.bin", d_offsets, offsets_bytes))
        return 1;
    printf("Loaded cluster_offsets to VRAM (%.1f KB)\n", (double)offsets_bytes/1024);

    // Raw vectors → pinned host DRAM (simulated SSD)
    float* h_raw_vectors;
    if (load_to_pinned(data_dir + "raw_vectors.bin", (void**)&h_raw_vectors,
                       raw_vectors_bytes))
        return 1;
    printf("Loaded raw_vectors to pinned DRAM (%.1f MB)\n\n",
           (double)raw_vectors_bytes / (1024*1024));

    // ============================
    // Generate query vectors (random sample from the dataset)
    // ============================

    float* d_queries;
    CHECK_CUDA(cudaMalloc(&d_queries, NUM_QUERIES * VEC_DIM * sizeof(float)));
    // Copy first NUM_QUERIES vectors as queries (they're in pinned DRAM)
    CHECK_CUDA(cudaMemcpy(d_queries, h_raw_vectors,
                           NUM_QUERIES * VEC_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));

    // ============================
    // Step 1: Find top-NPROBE clusters per query
    // ============================

    int* d_top_clusters;
    CHECK_CUDA(cudaMalloc(&d_top_clusters, NUM_QUERIES * NPROBE * sizeof(int)));

    printf("Finding top-%d clusters per query...\n", NPROBE);
    {
        int blk = 128, grd = (NUM_QUERIES + blk - 1) / blk;
        find_top_centroids<<<grd, blk>>>(d_centroids, d_queries,
                                          d_top_clusters, NUM_QUERIES);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    printf("Done.\n\n");

    // Result buffers
    int* d_result_ids;
    float* d_result_dists;
    CHECK_CUDA(cudaMalloc(&d_result_ids, NUM_QUERIES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result_dists, NUM_QUERIES * sizeof(float)));

    // ============================
    // Step 2: Search — Target T
    // ============================

    printf("[Target T] Direct PCIe reads...\n");
    {
        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

        int blk = 128;  // threads per block = threads per query
        CHECK_CUDA(cudaEventRecord(t0));
        search_clusters_target<<<NUM_QUERIES, blk>>>(
            h_raw_vectors, d_queries, d_top_clusters, d_offsets,
            d_result_ids, d_result_dists, NUM_QUERIES);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        printf("  Time: %.2f ms\n", ms);

        // Copy results for verification
        int* h_target_ids = (int*)malloc(NUM_QUERIES * sizeof(int));
        float* h_target_dists = (float*)malloc(NUM_QUERIES * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_target_ids, d_result_ids,
                               NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_target_dists, d_result_dists,
                               NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost));

        printf("  Sample results: query[0]->vec %d (dist %.4f), query[1]->vec %d (dist %.4f)\n\n",
               h_target_ids[0], h_target_dists[0],
               h_target_ids[1], h_target_dists[1]);

        free(h_target_ids);
        free(h_target_dists);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    // ============================
    // Step 3: Search — BaM with three-tier TLB
    // ============================

    printf("[BaM] Three-tier TLB cache...\n");
    {
        // Cache for raw vectors
        // raw_vectors is float array, cache works with 4-byte elements
        // Total floats = num_vectors * VEC_DIM
        size_t total_floats = num_vectors * VEC_DIM;
        uint32_t num_pages = (total_floats + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    // 1. Calculate the raw integer number of pages first
    uint32_t num_cache_pages = (uint32_t)(num_pages * cache_perc);

    // 2. Safety check: prevent allocating 0 bytes
    if (num_cache_pages == 0) {
        std::cerr << "Error: Zero cache pages calculated. Cannot begin program.\n";
        exit(1);
    }

    // 3. Calculate bytes strictly using the integer page count
    size_t cache_bytes = (size_t)num_cache_pages * CACHE_LINE_SIZE;

    // 4. Print using the correct types (%u for uint32_t)
    printf("  Cache: %u pages (%.1f MB) for %.1f MB of vectors\n",
           num_cache_pages, 
           (double)cache_bytes / (1024 * 1024),
           (double)raw_vectors_bytes / (1024 * 1024));

        BamCache cache;
        cache_init(&cache, cache_bytes, num_pages);

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

int blk = 128;
        // Launch in small batches to avoid find_slot cold-start contention.
        // With pinned DRAM as backing store, thousands of simultaneous
        // cache misses overwhelm find_slot. Real BaM with SSD latency
        // (11-324us) naturally serializes misses, avoiding this issue.
        // Batch size limits concurrent threads hitting find_slot.
        int batch_size = max(1, (int)(num_cache_pages / (blk * 4)));
        if (batch_size > NUM_QUERIES) batch_size = NUM_QUERIES;
        printf("  Batch size: %d queries per launch\n", batch_size);

        CHECK_CUDA(cudaEventRecord(t0));
        for (int batch = 0; batch < NUM_QUERIES; batch += batch_size) {
            int this_batch = min(batch_size, NUM_QUERIES - batch);
            search_clusters_bam<<<this_batch, blk>>>(
                cache, d_queries + batch * VEC_DIM,
                d_top_clusters + batch * NPROBE,
                d_offsets,
                d_result_ids + batch,
                d_result_dists + batch,
                this_batch,
                (const char*)h_raw_vectors);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));

        CacheStats s = cache_get_stats(&cache);
        printf("  Time: %.2f ms\n", ms);
        printf("  Cache hits: %llu, misses: %llu\n", s.hits, s.misses);
        printf("  PCIe traffic: %.1f MB (%llu pages × 4KB)\n",
               (double)s.misses * CACHE_LINE_SIZE / (1024*1024), s.misses);

        // Verify against Target T
        int* h_bam_ids = (int*)malloc(NUM_QUERIES * sizeof(int));
        float* h_bam_dists = (float*)malloc(NUM_QUERIES * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_bam_ids, d_result_ids,
                               NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_bam_dists, d_result_dists,
                               NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost));

        printf("  Sample results: query[0]->vec %d (dist %.4f), query[1]->vec %d (dist %.4f)\n",
               h_bam_ids[0], h_bam_dists[0],
               h_bam_ids[1], h_bam_dists[1]);

        free(h_bam_ids);
        free(h_bam_dists);
        cache_destroy(&cache);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    // Cleanup
    cudaFreeHost(h_raw_vectors);
    cudaFree(d_centroids);
    cudaFree(d_offsets);
    cudaFree(d_queries);
    cudaFree(d_top_clusters);
    cudaFree(d_result_ids);
    cudaFree(d_result_dists);

    printf("\n=== Done ===\n");
    return 0;
}
