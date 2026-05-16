#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <string>
#include <cstdlib>
#include <cfloat>
#include "cache.cuh"
#include "three_tier_tlb.cuh"
#include "bam_ptr.cuh"

#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define NB_CLUSTERS 1024
#define VEC_DIM 128
#define NPROBE 10 // search top-10 closest clusters
#define NUM_QUERIES 100

// Each vector = 128 floats = 512 bytes
// Each 4KB cache page holds 1024 floats = 8 vectors
// Vector i occupies floats [i*128 .. i*128+127]
#define FLOATS_PER_PAGE CL_ELEMS_INT              // 1024 (reusing the int constant, works for float too)
#define VECS_PER_PAGE (FLOATS_PER_PAGE / VEC_DIM) // 8

int load_to_vram(const std::string &filename, void *d_ptr, size_t bytes)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    char *h_buf;
    CHECK_CUDA(cudaHostAlloc(&h_buf, bytes, cudaHostAllocDefault));
    file.read(h_buf, bytes);
    if ((size_t)file.gcount() != bytes)
    {
        std::cerr << "Read error: expected " << bytes << " got " << file.gcount() << std::endl;
        cudaFreeHost(h_buf);
        return 1;
    }
    file.close();
    CHECK_CUDA(cudaMemcpy(d_ptr, h_buf, bytes, cudaMemcpyHostToDevice));
    cudaFreeHost(h_buf);
    return 0;
}

int load_to_pinned(const std::string &filename, void **h_ptr, size_t bytes)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    CHECK_CUDA(cudaHostAlloc(h_ptr, bytes, cudaHostAllocDefault));
    file.read((char *)*h_ptr, bytes);
    if ((size_t)file.gcount() != bytes)
    {
        std::cerr << "Read error: expected " << bytes << " got " << file.gcount() << std::endl;
        return 1;
    }
    file.close();
    return 0;
}

__global__ void find_top_centroids(const float *d_centroids, const float *d_queries,
                                   int *d_top_clusters, int num_queries)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_queries)
        return;

    const float *query = &d_queries[qid * VEC_DIM];

    // Compute distance to all centroids, keep top NPROBE
    float top_dists[NPROBE];
    int top_ids[NPROBE];
    for (int i = 0; i < NPROBE; i++)
    {
        top_dists[i] = FLT_MAX;
        top_ids[i] = -1;
    }

    for (int c = 0; c < NB_CLUSTERS; c++)
    {
        const float *centroid = &d_centroids[c * VEC_DIM];
        float dist = 0.0f;
        for (int d = 0; d < VEC_DIM; d++)
        {
            float diff = query[d] - centroid[d];
            dist += diff * diff;
        }

        // Insert into top-NPROBE if closer than worst
        if (dist < top_dists[NPROBE - 1])
        {
            top_dists[NPROBE - 1] = dist;
            top_ids[NPROBE - 1] = c;
            // Bubble sort to maintain order
            for (int i = NPROBE - 1; i > 0; i--)
            {
                if (top_dists[i] < top_dists[i - 1])
                {
                    float td = top_dists[i];
                    top_dists[i] = top_dists[i - 1];
                    top_dists[i - 1] = td;
                    int ti = top_ids[i];
                    top_ids[i] = top_ids[i - 1];
                    top_ids[i - 1] = ti;
                }
                else
                    break;
            }
        }
    }

    // Write top cluster IDs for this query
    for (int i = 0; i < NPROBE; i++)
    {
        d_top_clusters[qid * NPROBE + i] = top_ids[i];
    }
}

__global__ void search_clusters_bam(BamCache cache, const float *d_queries,
                                    const int *d_top_clusters, const int *d_offsets,
                                    int *d_result_ids, float *d_result_dists,
                                    int num_queries, const char *backing)
{
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    int qid = blockIdx.x; // one block per query
    if (qid >= num_queries)
    {
        t2_fini(&t2, &cache);
        return;
    }

    int lane = threadIdx.x;
    const float *query = &d_queries[qid * VEC_DIM];

    // Each thread tracks its own best match
    float best_dist = FLT_MAX;
    int best_id = -1;

    Tier1Local t1;
    t1_init(&t1);

    // Search each of the NPROBE clusters
    for (int p = 0; p < NPROBE; p++)
    {
        int cluster_id = d_top_clusters[qid * NPROBE + p];
        if (cluster_id < 0)
            continue;

        int vec_start = d_offsets[cluster_id];
        int vec_end = d_offsets[cluster_id + 1];
        int cluster_size = vec_end - vec_start;

        // Each thread handles a strided subset of vectors in this cluster
        for (int v = lane; v < cluster_size; v += blockDim.x)
        {
            int vec_id = vec_start + v;

            // Read 128 floats for this vector through BaM cache
            // Vector vec_id starts at float index vec_id * 128
            int float_base = vec_id * VEC_DIM;

            float dist = 0.0f;
            for (int d = 0; d < VEC_DIM; d++)
            {
                int fidx = float_base + d;
                int raw = t1_read(&t1, &t2, &cache, fidx, backing);
                float val = __int_as_float(raw);
                float diff = query[d] - val;
                dist += diff * diff;
            }

            if (dist < best_dist)
            {
                best_dist = dist;
                best_id = vec_id;
            }
        }
    }

    t1_fini(&t1, &t2);

    // Warp reduction to find minimum across threads
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
        int other_id = __shfl_down_sync(0xFFFFFFFF, best_id, offset);
        if (other_dist < best_dist)
        {
            best_dist = other_dist;
            best_id = other_id;
        }
    }

    // Cross-warp reduction using shared memory
    __shared__ float s_dists[32]; // one per warp
    __shared__ int s_ids[32];
    int warp_id = threadIdx.x / 32;
    int warp_lane = threadIdx.x % 32;

    if (warp_lane == 0)
    {
        s_dists[warp_id] = best_dist;
        s_ids[warp_id] = best_id;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {

        float absolute_best_dist = s_dists[0];
        int absolute_best_id = s_ids[0];

        int num_warps = blockDim.x / 32;
        for (int i = 1; i < num_warps; i++)
        {
            if (s_dists[i] < absolute_best_dist)
            {
                absolute_best_dist = s_dists[i];
                absolute_best_id = s_ids[i];
            }
        }

        // Write the final answer to global memory
        d_result_ids[qid] = absolute_best_id;
        d_result_dists[qid] = absolute_best_dist;
    }

    t2_fini(&t2, &cache);
}

__global__ void search_clusters_bam_2t(BamCache cache, const float *d_queries,
                                       const int *d_top_clusters, const int *d_offsets,
                                       int *d_result_ids, float *d_result_dists,
                                       int num_queries, const char *backing)
{
    int qid = blockIdx.x;
    if (qid >= num_queries)
        return;

    int lane = threadIdx.x;
    const float *query = &d_queries[qid * VEC_DIM];

    float best_dist = FLT_MAX;
    int best_id = -1;

    BamPtr bp;
    bp_init(&bp);

    for (int p = 0; p < NPROBE; p++)
    {
        int cluster_id = d_top_clusters[qid * NPROBE + p];
        if (cluster_id < 0)
            continue;

        int vec_start = d_offsets[cluster_id];
        int vec_end = d_offsets[cluster_id + 1];
        int cluster_size = vec_end - vec_start;

        for (int v = lane; v < cluster_size; v += blockDim.x)
        {
            int vec_id = vec_start + v;
            int float_base = vec_id * VEC_DIM;

            float dist = 0.0f;
            for (int d = 0; d < VEC_DIM; d++)
            {
                int fidx = float_base + d;
                int raw = bp_read(&bp, &cache, fidx, backing);
                float val = __int_as_float(raw);
                float diff = query[d] - val;
                dist += diff * diff;
            }

            if (dist < best_dist)
            {
                best_dist = dist;
                best_id = vec_id;
            }
        }
    }

    bp_fini(&bp, &cache);

    for (int offset = 16; offset > 0; offset /= 2)
    {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
        int other_id = __shfl_down_sync(0xFFFFFFFF, best_id, offset);
        if (other_dist < best_dist)
        {
            best_dist = other_dist;
            best_id = other_id;
        }
    }

    __shared__ float s_dists[32];
    __shared__ int s_ids[32];
    int warp_id = threadIdx.x / 32;
    int warp_lane = threadIdx.x % 32;

    if (warp_lane == 0)
    {
        s_dists[warp_id] = best_dist;
        s_ids[warp_id] = best_id;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float absolute_best_dist = s_dists[0];
        int absolute_best_id = s_ids[0];
        int num_warps = blockDim.x / 32;
        for (int i = 1; i < num_warps; i++)
        {
            if (s_dists[i] < absolute_best_dist)
            {
                absolute_best_dist = s_dists[i];
                absolute_best_id = s_ids[i];
            }
        }
        d_result_ids[qid] = absolute_best_id;
        d_result_dists[qid] = absolute_best_dist;
    }
}

__global__ void search_clusters_target(const float *h_vectors, const float *d_queries,
                                       const int *d_top_clusters, const int *d_offsets,
                                       int *d_result_ids, float *d_result_dists,
                                       int num_queries)
{
    int qid = blockIdx.x;
    if (qid >= num_queries)
        return;

    // Hardware Identification
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    const float *query = &d_queries[qid * VEC_DIM];

    // Only Lane 0 of each warp will track the local champion for its assigned vectors
    float best_dist = FLT_MAX;
    int best_id = -1;

    for (int p = 0; p < NPROBE; p++)
    {
        int cluster_id = d_top_clusters[qid * NPROBE + p];
        if (cluster_id < 0)
            continue;

        int vec_start = d_offsets[cluster_id];
        int vec_end = d_offsets[cluster_id + 1];
        int cluster_size = vec_end - vec_start;

        // Warp 0 takes Vector 0, Warp 1 takes Vector 1, etc.
        for (int v = warp_id; v < cluster_size; v += num_warps)
        {
            int vec_id = vec_start + v;
            const float *vec = &h_vectors[vec_id * VEC_DIM];

            // Thread 0 reads dims 0, 32, 64, 96.
            float partial_dist = 0.0f;
            for (int d = lane_id; d < VEC_DIM; d += 32)
            {
                float diff = query[d] - vec[d];
                partial_dist += diff * diff;
            }

            // Add up all 32 partial distances into a single total
            float total_dist = partial_dist;
            for (int offset = 16; offset > 0; offset /= 2)
            {
                total_dist += __shfl_down_sync(0xFFFFFFFF, total_dist, offset);
            }

            // Lane 0 compares the vector's total distance to the warp's best
            if (lane_id == 0)
            {
                if (total_dist < best_dist)
                {
                    best_dist = total_dist;
                    best_id = vec_id;
                }
            }
        }
    }

    __shared__ float s_dists[32];
    __shared__ int s_ids[32];

    // Only the Warp Leaders (Lane 0) have valid data to share
    if (lane_id == 0)
    {
        s_dists[warp_id] = best_dist;
        s_ids[warp_id] = best_id;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float abs_best_dist = s_dists[0];
        int abs_best_id = s_ids[0];

        for (int i = 1; i < num_warps; i++)
        {
            if (s_dists[i] < abs_best_dist)
            {
                abs_best_dist = s_dists[i];
                abs_best_id = s_ids[i];
            }
        }

        d_result_ids[qid] = abs_best_id;
        d_result_dists[qid] = abs_best_dist;
    }
}

void print_table_header()
{
    printf("\n=================================================================================================\n");
    printf("%-10s | %-4s | %-9s | %-6s | %-10s | %-12s | %-12s | %-6s\n",
           "Method", "WPQ", "Threads/Q", "Blocks", "Time (ms)", "VRAM Hits", "DRAM Misses", "Status");
    printf("-----------+------+-----------+--------+------------+--------------+--------------+--------\n");
}

void print_result_row(const char *method, int wpq, int threads_per_q, int blocks, float ms,
                      unsigned long long hits, unsigned long long misses, const char *status)
{
    if (strcmp(method, "Target_T") == 0)
    {
        printf("%-10s | %-4d | %-9d | %-6d | %10.2f | %-12s | %-12s | %-6s\n",
               method, wpq, threads_per_q, blocks, ms, "N/A", "N/A", status);
        return;
    }

    printf("%-10s | %-4d | %-9d | %-6d | %10.2f | %12llu | %12llu | %-6s\n",
           method, wpq, threads_per_q, blocks, ms, hits, misses, status);
}

int main(int argc, char **argv)
{
    std::string data_dir = "../data";
    size_t num_vectors = 1 << 20;
    float cache_perc = 1.0f;
    int wpq = 4;      // Warps Per Query (default to 4 warps = 128 threads)
    int tlb_mode = 3; // 3 = three-tier (default), 2 = two-tier

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--data_dir") == 0 && i + 1 < argc)
        {
            data_dir = std::string(argv[++i]);
        }
        else if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc)
        {
            num_vectors = 1ULL << atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--cache") == 0 && i + 1 < argc)
        {
            cache_perc = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--wpq") == 0 && i + 1 < argc)
        {
            wpq = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tlb") == 0 && i + 1 < argc)
        {
            tlb_mode = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --data_dir <path>   : Data directory\n");
            printf("  --scale <N>    : Number of vectors (2^N). Default: 20\n");
            printf("  --cache <F>    : VRAM cache percentage (0.0 to 1.0). Default: 1.0\n");
            printf("  --wpq <N>      : Warps Per Query (1, 2, 3, 4...). Default: 4\n");
            printf("  --tlb <N>      : TLB mode (2 = two-tier, 3 = three-tier). Default: 3\n");
            exit(0);
        }
    }

    if (tlb_mode != 2 && tlb_mode != 3)
    {
        std::cerr << "Error: --tlb must be 2 or 3\n";
        exit(1);
    }

    if (wpq < 1 || wpq > 32)
    {
        std::cerr << "Error: wpq must be between 1 and 32\n";
        exit(1);
    }

    if (data_dir.back() != '/')
        data_dir += "/";

    // Convert warps to threads
    int blk_threads = wpq * 32;

    // Calculate sizes and cache parameters
    size_t raw_vectors_bytes = num_vectors * VEC_DIM * sizeof(float);
    size_t centroid_bytes = NB_CLUSTERS * VEC_DIM * sizeof(float);
    size_t offsets_bytes = (NB_CLUSTERS + 1) * sizeof(int);

    size_t total_floats = num_vectors * VEC_DIM;
    uint32_t num_pages = (total_floats + CL_ELEMS_INT - 1) / CL_ELEMS_INT;
    uint32_t num_cache_pages = (uint32_t)(num_pages * cache_perc);

    if (num_cache_pages == 0)
    {
        std::cerr << "Error: Zero cache pages calculated. Cannot begin program.\n";
        exit(1);
    }
    size_t cache_bytes = (size_t)num_cache_pages * CACHE_LINE_SIZE;

    printf("\n======================================================================\n");
    printf("                  ANN BENCHMARK CONFIGURATION SUMMARY                   \n");
    printf("======================================================================\n");

    printf("[1] IVF-FLAT DATASET TOPOLOGY\n");
    printf("    Total Vectors    : %zu\n", num_vectors);
    printf("    Vector Dimension : %d\n", VEC_DIM);
    printf("    Clusters         : %d\n", NB_CLUSTERS);
    printf("    Target Probes    : %d\n", NPROBE);
    printf("    Total Queries    : %d\n", NUM_QUERIES);

    printf("\n[2] MEMORY & CACHE HIERARCHY\n");
    printf("    Centroids        : %.2f KB (VRAM)\n", (double)centroid_bytes / (1024));
    printf("    Cluster Offsets  : %.2f KB (VRAM)\n", (double)offsets_bytes / (1024));
    printf("    T3 Cache Line    : %d Bytes (%d floats)\n", CACHE_LINE_SIZE, FLOATS_PER_PAGE);
    printf("    Raw Vectors      : %u pages | %.2f MB (Pinned DRAM)\n", num_pages, (double)raw_vectors_bytes / (1024 * 1024));
    printf("    T3 Cache Target  : %.1f%%\n", cache_perc * 100);
    printf("    T3 Capacity      : %u pages | %.2f MB (Global VRAM)\n", num_cache_pages, (double)cache_bytes / (1024 * 1024));
    printf("    T2 Capacity      : %d mapping slots shared per block\n", T2_NUM_ENTRIES);

    printf("\n[3] HARDWARE EXECUTION PARAMETERS\n");
    printf("    Launch Strategy  : 1 Thread Block per Query\n");
    printf("    Warps per Query  : %d\n", wpq);
    printf("    Threads per Block: %d\n", blk_threads);
    printf("    Blocks Spawned   : %d\n", NUM_QUERIES);
    printf("======================================================================\n\n");

    // Load data

    printf("Loading data from disk...\n");
    float *d_centroids;
    CHECK_CUDA(cudaMalloc(&d_centroids, centroid_bytes));
    if (load_to_vram(data_dir + "centroids.bin", d_centroids, centroid_bytes))
        return 1;

    int *d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, offsets_bytes));
    if (load_to_vram(data_dir + "cluster_offsets.bin", d_offsets, offsets_bytes))
        return 1;

    float *h_raw_vectors;
    if (load_to_pinned(data_dir + "raw_vectors.bin", (void **)&h_raw_vectors, raw_vectors_bytes))
        return 1;

    // Setup queries

    float *h_queries = (float *)malloc(NUM_QUERIES * VEC_DIM * sizeof(float));

    // Calculate the stride to evenly sample across the whole dataset
    size_t stride = num_vectors / NUM_QUERIES;

    for (int i = 0; i < NUM_QUERIES; i++)
    {
        size_t source_vec_id = i * stride;
        memcpy(&h_queries[i * VEC_DIM],
               &h_raw_vectors[source_vec_id * VEC_DIM],
               VEC_DIM * sizeof(float));
    }

    float *d_queries;
    CHECK_CUDA(cudaMalloc(&d_queries, NUM_QUERIES * VEC_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries, NUM_QUERIES * VEC_DIM * sizeof(float), cudaMemcpyHostToDevice));
    free(h_queries);

    // Find top-NPROBE clusters per query

    int *d_top_clusters;
    CHECK_CUDA(cudaMalloc(&d_top_clusters, NUM_QUERIES * NPROBE * sizeof(int)));

    int blk = 128, grd = (NUM_QUERIES + blk - 1) / blk;
    find_top_centroids<<<grd, blk>>>(d_centroids, d_queries, d_top_clusters, NUM_QUERIES);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Result buffers
    int *d_result_ids;
    float *d_result_dists;
    CHECK_CUDA(cudaMalloc(&d_result_ids, NUM_QUERIES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result_dists, NUM_QUERIES * sizeof(float)));

    print_table_header();

    // Target T

    float target_ms = 0.0f;
    {
        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

        CHECK_CUDA(cudaEventRecord(t0));
        search_clusters_target<<<NUM_QUERIES, blk_threads>>>(
            h_raw_vectors, d_queries, d_top_clusters, d_offsets,
            d_result_ids, d_result_dists, NUM_QUERIES);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&target_ms, t0, t1));

        print_result_row("Target_T", wpq, blk_threads, NUM_QUERIES, target_ms, 0, 0, "REF");

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    // Target T verification state
    int *h_target_ids = (int *)malloc(NUM_QUERIES * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_target_ids, d_result_ids, NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost));

    // BaM

    float bam_ms = 0.0f;
    {
        BamCache cache;
        cache_init(&cache, cache_bytes, num_pages);

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

        CHECK_CUDA(cudaEventRecord(t0));
        if (tlb_mode == 2)
        {
            search_clusters_bam_2t<<<NUM_QUERIES, blk_threads>>>(
                cache, d_queries, d_top_clusters, d_offsets,
                d_result_ids, d_result_dists, NUM_QUERIES,
                (const char *)h_raw_vectors);
        }
        else
        {
            search_clusters_bam<<<NUM_QUERIES, blk_threads>>>(
                cache, d_queries, d_top_clusters, d_offsets,
                d_result_ids, d_result_dists, NUM_QUERIES,
                (const char *)h_raw_vectors);
        }
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&bam_ms, t0, t1));

        CacheStats s = cache_get_stats(&cache);

        int *h_bam_ids = (int *)malloc(NUM_QUERIES * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_bam_ids, d_result_ids, NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost));

        const char *status = "PASS";
        for (int i = 0; i < NUM_QUERIES; i++)
        {
            if (h_target_ids[i] != h_bam_ids[i])
            {
                status = "FAIL";
                break;
            }
        }

        const char *method = (tlb_mode == 2) ? "BaM_2T" : "BaM";
        print_result_row(method, wpq, blk_threads, NUM_QUERIES, bam_ms, s.hits, s.misses, status);

        free(h_bam_ids);
        cache_destroy(&cache);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    printf("=================================================================================================\n");

    // Cleanup
    free(h_target_ids);
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
