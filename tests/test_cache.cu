#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cache.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void flush_l2(char* buf, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    volatile char x;
    for (int i = tid; i < size; i += gridDim.x * blockDim.x) x = buf[i];
}

// ============================================================================
// Test kernels
// ============================================================================

__global__
void kernel_cache_no_coalesce(BamCache cache, const int* indices,
                               int* output, int n, const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    output[tid] = cache_read_no_coalesce(&cache, indices[tid], backing);
}

__global__
void kernel_cache_coalesced(BamCache cache, const int* indices,
                             int* output, int n, const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    output[tid] = cache_read_coalesced(&cache, indices[tid], backing);
}

__global__
void kernel_direct_read(const int* backing, const int* indices,
                         int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    output[tid] = backing[indices[tid]];
}

// ============================================================================
// Index generation
// ============================================================================

void gen_uniform(int* h, int n, int num_elems) {
    srand(42);
    for (int i = 0; i < n; i++) h[i] = rand() % num_elems;
}

void gen_zipfian(int* h, int n, int num_elems, double skew) {
    srand(42);
    int num_pages = num_elems / CL_ELEMS_INT;
    double* cdf = (double*)malloc(num_pages * sizeof(double));
    double sum = 0;
    for (int i = 0; i < num_pages; i++) sum += 1.0 / pow(i + 1.0, skew);
    double run = 0;
    for (int i = 0; i < num_pages; i++) {
        run += 1.0 / pow(i + 1.0, skew);
        cdf[i] = run / sum;
    }
    for (int i = 0; i < n; i++) {
        double r = (double)rand() / RAND_MAX;
        int lo = 0, hi = num_pages - 1;
        while (lo < hi) { int mid = (lo+hi)/2; if (cdf[mid] < r) lo=mid+1; else hi=mid; }
        h[i] = lo * CL_ELEMS_INT + (rand() % CL_ELEMS_INT);
    }
    free(cdf);
}

void gen_warp_shared(int* h, int n, int num_elems, int unique_per_warp) {
    srand(42);
    int num_pages = num_elems / CL_ELEMS_INT;
    for (int i = 0; i < n; i++) {
        int warp = i / 32, lane = i % 32;
        int group = lane % unique_per_warp;
        unsigned seed = (unsigned)(warp * unique_per_warp + group) * 2654435761u;
        int page = seed % num_pages;
        h[i] = page * CL_ELEMS_INT + (lane % CL_ELEMS_INT);
    }
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct BenchResult {
    float time_ms;
    CacheStats stats;
    float hit_rate;
    float bw_gbps;
};

BenchResult run_cache(BamCache* cache, const int* d_idx, int* d_out, int n,
                       const char* backing, char* d_flush, int fs, bool coal) {
    BenchResult r = {};
    cache_reset(cache);
    flush_l2<<<256,256>>>(d_flush, fs);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    int blk = 256, grd = (n + blk - 1) / blk;

    CHECK_CUDA(cudaEventRecord(t0));
    if (coal)
        kernel_cache_coalesced<<<grd, blk>>>(*cache, d_idx, d_out, n, backing);
    else
        kernel_cache_no_coalesce<<<grd, blk>>>(*cache, d_idx, d_out, n, backing);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));

    cudaError_t ke = cudaGetLastError();
    if (ke != cudaSuccess) fprintf(stderr, "Kernel err: %s\n", cudaGetErrorString(ke));

    CHECK_CUDA(cudaEventElapsedTime(&r.time_ms, t0, t1));
    r.stats = cache_get_stats(cache);
    unsigned long long total = r.stats.hits + r.stats.misses;
    r.hit_rate = total > 0 ? (float)r.stats.hits / total * 100.0f : 0;
    r.bw_gbps = ((double)n * sizeof(int) / 1e9) / (r.time_ms / 1e3);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return r;
}

BenchResult run_direct(const int* d_backing, const int* d_idx, int* d_out,
                        int n, char* d_flush, int fs) {
    BenchResult r = {};
    flush_l2<<<256,256>>>(d_flush, fs);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    int blk = 256, grd = (n + blk - 1) / blk;

    CHECK_CUDA(cudaEventRecord(t0));
    kernel_direct_read<<<grd, blk>>>(d_backing, d_idx, d_out, n);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));

    CHECK_CUDA(cudaEventElapsedTime(&r.time_ms, t0, t1));
    r.bw_gbps = ((double)n * sizeof(int) / 1e9) / (r.time_ms / 1e3);
    r.hit_rate = -1;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== BaM Cache v3 Benchmark (Real BaM Design) ===\n");
    printf("Cache line: %d bytes (%d ints)\n\n", CACHE_LINE_SIZE, CL_ELEMS_INT);

    const size_t NUM_ELEMS = 128 * 1024 * 1024;  // 128M ints = 512MB
    const uint32_t NUM_PAGES = NUM_ELEMS / CL_ELEMS_INT;
    const int NUM_ACC = 4 * 1024 * 1024;  // 4M accesses

    printf("Dataset: %zu elems (%.0f MB), %u logical pages\n",
           NUM_ELEMS, (double)NUM_ELEMS*sizeof(int)/(1024*1024), NUM_PAGES);
    printf("Accesses: %d (%.1fM)\n\n", NUM_ACC, NUM_ACC/1e6);

    // Backing store in pinned host memory
    int* h_backing;
    CHECK_CUDA(cudaHostAlloc(&h_backing, NUM_ELEMS * sizeof(int),
                              cudaHostAllocDefault));
    printf("Initializing backing store...\n");
    for (size_t i = 0; i < NUM_ELEMS; i++) h_backing[i] = (int)i;

    const char* d_backing = (const char*)h_backing;

    int* h_idx = (int*)malloc(NUM_ACC * sizeof(int));
    int* d_idx; CHECK_CUDA(cudaMalloc(&d_idx, NUM_ACC * sizeof(int)));
    int* d_out; CHECK_CUDA(cudaMalloc(&d_out, NUM_ACC * sizeof(int)));
    char* d_flush; CHECK_CUDA(cudaMalloc(&d_flush, 8*1024*1024));

    // ==== Test 1: Correctness ====
    printf("--- Test 1: Correctness ---\n");
    {
        size_t cache_bytes = 64ULL * 1024 * 1024;  // 64MB cache
        BamCache cache;
        cache_init(&cache, cache_bytes, NUM_PAGES);
        printf("Cache: %u slots (%.0f MB) for %u logical pages\n",
               cache.num_slots,
               (double)cache.num_slots * CACHE_LINE_SIZE / (1024*1024),
               NUM_PAGES);

        gen_uniform(h_idx, NUM_ACC, NUM_ELEMS);
        CHECK_CUDA(cudaMemcpy(d_idx, h_idx, NUM_ACC*sizeof(int),
                               cudaMemcpyHostToDevice));

        int blk = 256, grd = (NUM_ACC + blk - 1) / blk;
        kernel_cache_coalesced<<<grd, blk>>>(cache, d_idx, d_out, NUM_ACC,
                                              d_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        int* h_out = (int*)malloc(NUM_ACC * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_out, d_out, NUM_ACC*sizeof(int),
                               cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < NUM_ACC; i++) {
            if (h_out[i] != h_idx[i]) {
                if (errs < 5)
                    printf("  ERR[%d]: idx=%d expected=%d got=%d\n",
                           i, h_idx[i], h_idx[i], h_out[i]);
                errs++;
            }
        }
        printf("  %s (%d errors / %d)\n",
               errs == 0 ? "PASSED" : "FAILED", errs, NUM_ACC);
        CacheStats s = cache_get_stats(&cache);
        printf("  hits=%llu misses=%llu waits=%llu\n",
               s.hits, s.misses, s.coalesced_waits);
        free(h_out);
        cache_destroy(&cache);
    }

    // ==== Test 2: Coalesced vs Non-Coalesced ====
    printf("\n--- Test 2: Coalesced vs Non-Coalesced ---\n");
    printf("%-6s | %-8s %-8s %-8s %-8s | %-8s %-8s %-8s %-8s\n",
           "Uniq", "NoC ms", "NoC BW", "Hit%", "Miss",
           "BaM ms", "BaM BW", "Hit%", "Miss");
    printf("-------+-------------------------------------+"
           "------------------------------------\n");
    {
        size_t cache_bytes = 64ULL * 1024 * 1024;
        BamCache cache;
        cache_init(&cache, cache_bytes, NUM_PAGES);

        int uniq[] = {1, 4, 8, 16, 32};
        for (int u = 0; u < 5; u++) {
            gen_warp_shared(h_idx, NUM_ACC, NUM_ELEMS, uniq[u]);
            CHECK_CUDA(cudaMemcpy(d_idx, h_idx, NUM_ACC*sizeof(int),
                                   cudaMemcpyHostToDevice));

            BenchResult nc = run_cache(&cache, d_idx, d_out, NUM_ACC,
                                        d_backing, d_flush, 8*1024*1024, false);
            BenchResult co = run_cache(&cache, d_idx, d_out, NUM_ACC,
                                        d_backing, d_flush, 8*1024*1024, true);

            printf("%-6d | %-8.1f %-8.2f %-8.1f %-8llu | %-8.1f %-8.2f %-8.1f %-8llu\n",
                   uniq[u],
                   nc.time_ms, nc.bw_gbps, nc.hit_rate, nc.stats.misses,
                   co.time_ms, co.bw_gbps, co.hit_rate, co.stats.misses);
        }
        cache_destroy(&cache);
    }

    // ==== Test 3: Cache Size Sensitivity ====
    printf("\n--- Test 3: Cache Size Sensitivity (Zipfian skew=1.0) ---\n");
    printf("%-10s | %-8s %-8s %-8s %-10s\n",
           "Cache", "ms", "BW GB/s", "Hit%", "Misses");
    printf("-----------+------------------------------------------\n");
    {
        gen_zipfian(h_idx, NUM_ACC, NUM_ELEMS, 1.0);
        CHECK_CUDA(cudaMemcpy(d_idx, h_idx, NUM_ACC*sizeof(int),
                               cudaMemcpyHostToDevice));

        size_t sizes[] = {4*1024*1024, 16*1024*1024, 64*1024*1024,
                           128*1024*1024, 256*1024*1024};
        const char* labels[] = {"4 MB", "16 MB", "64 MB", "128 MB", "256 MB"};

        for (int c = 0; c < 5; c++) {
            BamCache cache;
            cache_init(&cache, sizes[c], NUM_PAGES);
            BenchResult r = run_cache(&cache, d_idx, d_out, NUM_ACC,
                                       d_backing, d_flush, 8*1024*1024, true);
            printf("%-10s | %-8.1f %-8.2f %-8.1f %-10llu\n",
                   labels[c], r.time_ms, r.bw_gbps, r.hit_rate, r.stats.misses);
            cache_destroy(&cache);
        }
    }

    // ==== Test 4: BaM vs Target T ====
    printf("\n--- Test 4: BaM Cache vs Direct PCIe (Target T) ---\n");
    {
        gen_zipfian(h_idx, NUM_ACC, NUM_ELEMS, 1.0);
        CHECK_CUDA(cudaMemcpy(d_idx, h_idx, NUM_ACC*sizeof(int),
                               cudaMemcpyHostToDevice));

        BenchResult target = run_direct((const int*)h_backing, d_idx, d_out,
                                         NUM_ACC, d_flush, 8*1024*1024);
        printf("  Target T (direct PCIe):  %.1f ms, %.2f GB/s\n",
               target.time_ms, target.bw_gbps);

        BamCache cache;
        cache_init(&cache, 64ULL*1024*1024, NUM_PAGES);
        BenchResult bam = run_cache(&cache, d_idx, d_out, NUM_ACC,
                                     d_backing, d_flush, 8*1024*1024, true);
        printf("  BaM (64MB cache):        %.1f ms, %.2f GB/s, hit=%.1f%%\n",
               bam.time_ms, bam.bw_gbps, bam.hit_rate);
        printf("  Speedup: %.2fx\n", target.time_ms / bam.time_ms);
        printf("  BaM PCIe: %llu misses × %d B = %.1f MB\n",
               bam.stats.misses, CACHE_LINE_SIZE,
               (double)bam.stats.misses * CACHE_LINE_SIZE / (1024*1024));
        printf("  Target PCIe: %d × 4B = %.1f MB\n",
               NUM_ACC, (double)NUM_ACC * sizeof(int) / (1024*1024));

        cache_destroy(&cache);
    }

    free(h_idx);
    cudaFreeHost(h_backing);
    cudaFree(d_idx); cudaFree(d_out); cudaFree(d_flush);
    printf("\n=== Phase 2 Complete ===\n");
    return 0;
}
