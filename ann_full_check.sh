cat > /tmp/ann_diagnose.cu << 'EOF'
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cfloat>
#include <unistd.h>
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
#define NUM_QUERIES   1000

int load_to_vram(const char* fn, void* d, size_t b) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) { fprintf(stderr, "Can't open %s\n", fn); return 1; }
    char* h; cudaHostAlloc(&h, b, cudaHostAllocDefault);
    f.read(h, b); f.close();
    cudaMemcpy(d, h, b, cudaMemcpyHostToDevice);
    cudaFreeHost(h); return 0;
}

__global__
void find_top_centroids(const float* d_centroids, const float* d_queries,
                        int* d_top_clusters, int num_queries) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_queries) return;
    const float* query = &d_queries[qid * VEC_DIM];
    float top_dists[NPROBE];
    int top_ids[NPROBE];
    for (int i = 0; i < NPROBE; i++) { top_dists[i] = FLT_MAX; top_ids[i] = -1; }
    for (int c = 0; c < NB_CLUSTERS; c++) {
        const float* cent = &d_centroids[c * VEC_DIM];
        float dist = 0;
        for (int d = 0; d < VEC_DIM; d++) { float diff = query[d] - cent[d]; dist += diff*diff; }
        if (dist < top_dists[NPROBE-1]) {
            top_dists[NPROBE-1] = dist; top_ids[NPROBE-1] = c;
            for (int i = NPROBE-1; i > 0; i--) {
                if (top_dists[i] < top_dists[i-1]) {
                    float td=top_dists[i]; top_dists[i]=top_dists[i-1]; top_dists[i-1]=td;
                    int ti=top_ids[i]; top_ids[i]=top_ids[i-1]; top_ids[i-1]=ti;
                } else break;
            }
        }
    }
    for (int i = 0; i < NPROBE; i++) d_top_clusters[qid*NPROBE+i] = top_ids[i];
}

// Diagnostic version of search kernel
// Each thread writes its progress to a status array so we can see
// where it got stuck after a timeout
//
// Status codes:
//   0 = not started
//   1 = entered kernel, past t2_init
//   2 = starting cluster loop
//   10+p = processing cluster p (0-9)
//   20+p = finished cluster p
//   30 = all clusters done, entering t1_fini
//   31 = t1_fini done, entering reduction
//   32 = reduction done, entering t2_fini
//   33 = past t2_fini __syncthreads
//   34 = t2_fini done
//   99 = kernel complete
//
//   Negative = stuck in find_slot, value = -(spin count / 1000)

__global__
void search_diagnose(BamCache cache, const float* d_queries,
                      const int* d_top_clusters, const int* d_offsets,
                      int* d_result_ids, float* d_result_dists,
                      int num_queries, const char* backing,
                      int* d_status,       // [num_queries * blockDim.x]
                      int* d_block_done) { // [num_queries]
    __shared__ Tier2TLB t2;
    t2_init(&t2);

    int qid = blockIdx.x;
    int lane = threadIdx.x;
    int status_idx = qid * blockDim.x + lane;

    if (qid >= num_queries) {
        d_status[status_idx] = 99;
        t2_fini(&t2, &cache);
        return;
    }

    d_status[status_idx] = 1; // past init

    const float* query = &d_queries[qid * VEC_DIM];
    float best_dist = FLT_MAX;
    int best_id = -1;

    Tier1Local t1;
    t1_init(&t1);

    d_status[status_idx] = 2; // starting clusters

    for (int p = 0; p < NPROBE; p++) {
        d_status[status_idx] = 10 + p; // processing cluster p

        int cluster_id = d_top_clusters[qid * NPROBE + p];
        if (cluster_id < 0) continue;

        int vec_start = d_offsets[cluster_id];
        int vec_end   = d_offsets[cluster_id + 1];

        int vec_count = 0;
        for (int v = lane; v < (vec_end - vec_start); v += blockDim.x) {
            int vec_id = vec_start + v;
            int float_base = vec_id * VEC_DIM;
            float dist = 0.0f;

            for (int d = 0; d < VEC_DIM; d++) {
                int raw = t1_read(&t1, &t2, &cache, float_base + d, backing);
                float val = __int_as_float(raw);
                float diff = query[d] - val;
                dist += diff * diff;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_id = vec_id;
            }
            vec_count++;
        }

        d_status[status_idx] = 20 + p; // finished cluster p
    }

    d_status[status_idx] = 30; // all clusters done

    t1_fini(&t1, &t2);
    d_status[status_idx] = 31; // t1_fini done

    // Simple shared memory reduction (no shuffle deadlock)
    __shared__ float sd[256];
    __shared__ int si[256];
    sd[lane] = best_dist;
    si[lane] = best_id;
    __syncthreads();

    if (lane == 0) {
        for (int i = 1; i < blockDim.x; i++) {
            if (sd[i] < best_dist) { best_dist = sd[i]; best_id = si[i]; }
        }
        d_result_ids[qid] = best_id;
        d_result_dists[qid] = best_dist;
    }

    d_status[status_idx] = 32; // reduction done

    t2_fini(&t2, &cache);
    d_status[status_idx] = 99; // complete

    if (lane == 0) atomicAdd(d_block_done, 1);
}

int main(int argc, char** argv) {
    const char* dir = "/scratch/ljp5718/";
    if (argc > 1) dir = argv[1];

    size_t num_vectors = 1 << 20;
    size_t raw_bytes = num_vectors * VEC_DIM * sizeof(float);
    uint32_t num_pages = (num_vectors * VEC_DIM + CL_ELEMS_INT - 1) / CL_ELEMS_INT;

    float cache_perc = 0.5f;
    if (argc > 2) cache_perc = atof(argv[2]);

    printf("=== ANN Diagnostic (cache=%.0f%%) ===\n\n", cache_perc * 100);

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

    float* d_queries;
    cudaMalloc(&d_queries, NUM_QUERIES * VEC_DIM * sizeof(float));
    cudaMemcpy(d_queries, h_raw, NUM_QUERIES * VEC_DIM * sizeof(float), cudaMemcpyHostToDevice);

    int* d_top;
    cudaMalloc(&d_top, NUM_QUERIES * NPROBE * sizeof(int));
    {
        int blk = 128, grd = (NUM_QUERIES + blk - 1) / blk;
        find_top_centroids<<<grd, blk>>>(d_cent, d_queries, d_top, NUM_QUERIES);
        cudaDeviceSynchronize();
    }

    int* d_rid; float* d_rdist;
    cudaMalloc(&d_rid, NUM_QUERIES * sizeof(int));
    cudaMalloc(&d_rdist, NUM_QUERIES * sizeof(float));

    // Status tracking
    int blk = 128;
    int total_threads = NUM_QUERIES * blk;
    int* d_status;
    cudaMalloc(&d_status, total_threads * sizeof(int));
    cudaMemset(d_status, 0, total_threads * sizeof(int));

    int* d_block_done;
    cudaMalloc(&d_block_done, sizeof(int));
    cudaMemset(d_block_done, 0, sizeof(int));

    // Launch with cache
    uint32_t cache_pages = (uint32_t)(num_pages * cache_perc);
    size_t cache_bytes = (size_t)cache_pages * CACHE_LINE_SIZE;
    printf("Cache: %u pages (%.1f MB), launching %d blocks × %d threads\n\n",
           cache_pages, (double)cache_bytes/(1024*1024), NUM_QUERIES, blk);

    BamCache cache;
    cache_init(&cache, cache_bytes, num_pages);

    // Launch ALL blocks at once
    search_diagnose<<<NUM_QUERIES, blk>>>(
        cache, d_queries, d_top, d_off,
        d_rid, d_rdist, NUM_QUERIES,
        (const char*)h_raw, d_status, d_block_done);

    // Poll status every 2 seconds for 30 seconds
    int* h_status = (int*)malloc(total_threads * sizeof(int));
    for (int poll = 0; poll < 15; poll++) {
        sleep(2);

        int blocks_done = 0;
        cudaMemcpy(&blocks_done, d_block_done, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_status, d_status, total_threads * sizeof(int), cudaMemcpyDeviceToHost);

        // Count status distribution
        int counts[100] = {};
        for (int i = 0; i < total_threads; i++) {
            int s = h_status[i];
            if (s >= 0 && s < 100) counts[s]++;
        }

        printf("[%2ds] Blocks done: %d/%d\n", (poll+1)*2, blocks_done, NUM_QUERIES);
        printf("  Status: not_started=%d, init=%d, cluster_loop=%d\n",
               counts[0], counts[1], counts[2]);
        printf("  Processing clusters: ");
        for (int p = 0; p < NPROBE; p++)
            if (counts[10+p] > 0) printf("c%d=%d ", p, counts[10+p]);
        printf("\n");
        printf("  Finished clusters: ");
        for (int p = 0; p < NPROBE; p++)
            if (counts[20+p] > 0) printf("c%d=%d ", p, counts[20+p]);
        printf("\n");
        printf("  Post-work: t1_fini=%d, reduction=%d, t2_fini_sync=%d, t2_fini_done=%d, complete=%d\n",
               counts[30], counts[31], counts[32], counts[33], counts[99]);

        // Find stuck blocks: blocks where threads have mixed status
        int stuck_blocks = 0;
        for (int b = 0; b < NUM_QUERIES && stuck_blocks < 5; b++) {
            int min_s = 100, max_s = -1;
            for (int t = 0; t < blk; t++) {
                int s = h_status[b * blk + t];
                if (s < min_s) min_s = s;
                if (s > max_s) max_s = s;
            }
            // Block is stuck if threads have very different statuses
            if (max_s < 99 && (max_s - min_s) > 5) {
                printf("  STUCK block %d: min_status=%d, max_status=%d\n", b, min_s, max_s);
                // Print first few thread statuses
                printf("    Threads: ");
                for (int t = 0; t < 16; t++)
                    printf("t%d=%d ", t, h_status[b * blk + t]);
                printf("...\n");
                stuck_blocks++;
            }
        }

        if (blocks_done == NUM_QUERIES) {
            printf("\nAll blocks completed!\n");
            break;
        }
        printf("\n");
    }

    // Final check
    int blocks_done = 0;
    cudaMemcpy(&blocks_done, d_block_done, sizeof(int), cudaMemcpyDeviceToHost);
    if (blocks_done < NUM_QUERIES) {
        printf("\nTIMEOUT: %d/%d blocks completed\n", blocks_done, NUM_QUERIES);

        // Detailed analysis of stuck blocks
        cudaMemcpy(h_status, d_status, total_threads * sizeof(int), cudaMemcpyDeviceToHost);
        printf("\nStuck block analysis:\n");
        int analyzed = 0;
        for (int b = 0; b < NUM_QUERIES && analyzed < 10; b++) {
            bool all_done = true;
            for (int t = 0; t < blk; t++) {
                if (h_status[b * blk + t] != 99) { all_done = false; break; }
            }
            if (!all_done) {
                printf("  Block %d:", b);
                // Count per-status
                int bc[100] = {};
                for (int t = 0; t < blk; t++) {
                    int s = h_status[b * blk + t];
                    if (s >= 0 && s < 100) bc[s]++;
                }
                for (int s = 0; s < 100; s++)
                    if (bc[s] > 0) printf(" [%d]=%d", s, bc[s]);
                printf("\n");
                analyzed++;
            }
        }
    }

    free(h_status);
    cudaFreeHost(h_raw);
    cudaFree(d_cent); cudaFree(d_off); cudaFree(d_queries);
    cudaFree(d_top); cudaFree(d_rid); cudaFree(d_rdist);
    cudaFree(d_status); cudaFree(d_block_done);
    cache_destroy(&cache);
    printf("\n=== Done ===\n");
    return 0;
}
EOF

/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o ann_diagnose /tmp/ann_diagnose.cu && ./ann_diagnose /scratch/ljp5718/ 0.5