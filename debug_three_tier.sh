cat > /tmp/debug_scale.cu << 'EOF'
#include <cstdio>
#include <cuda_runtime.h>
#include "cache.cuh"
#include "three_tier_tlb.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__
void kernel_random(BamCache cache, const int* indices, int* output,
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

int main() {
    printf("=== Scale-up Debug ===\n\n");

    const int NUM_ELEMS = 4 * 1024 * 1024;
    const uint32_t NUM_PAGES = NUM_ELEMS / CL_ELEMS_INT;

    int* h_backing;
    CHECK_CUDA(cudaHostAlloc(&h_backing, NUM_ELEMS * sizeof(int),
                              cudaHostAllocDefault));
    for (int i = 0; i < NUM_ELEMS; i++) h_backing[i] = i;

    int max_n = 1024;
    int* h_idx = (int*)malloc(max_n * sizeof(int));
    srand(42);
    for (int i = 0; i < max_n; i++) h_idx[i] = rand() % NUM_ELEMS;

    int* d_idx; CHECK_CUDA(cudaMalloc(&d_idx, max_n * sizeof(int)));
    int* d_out; CHECK_CUDA(cudaMalloc(&d_out, max_n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_idx, h_idx, max_n * sizeof(int), cudaMemcpyHostToDevice));

    // Test with increasing thread counts
    // Use 100% cache so find_slot has plenty of room
    BamCache cache;
    cache_init(&cache, (size_t)NUM_PAGES * CACHE_LINE_SIZE, NUM_PAGES);

    int counts[] = {32, 64, 128, 256, 512, 768, 1024};
    int num_tests = 7;

    for (int t = 0; t < num_tests; t++) {
        int n = counts[t];
        int blk = 256;
        if (n < 256) blk = n;
        int grd = (n + blk - 1) / blk;

        cache_reset(&cache);
        printf("n=%4d (blk=%d, grd=%d)... ", n, blk, grd);
        fflush(stdout);

        // Use timeout via CUDA events
        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));
        CHECK_CUDA(cudaEventRecord(t0));

        kernel_random<<<grd, blk>>>(cache, d_idx, d_out, n,
                                     (const char*)h_backing);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("FAILED: %s\n", cudaGetErrorString(err));
            continue;
        }

        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));

        int* h_out = (int*)malloc(n * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < n; i++)
            if (h_out[i] != h_idx[i]) errs++;

        CacheStats s = cache_get_stats(&cache);
        printf("%.1f ms, err=%d, hits=%llu, misses=%llu\n",
               ms, errs, s.hits, s.misses);

        free(h_out);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    free(h_idx);
    cudaFreeHost(h_backing);
    cudaFree(d_idx); cudaFree(d_out);
    cache_destroy(&cache);
    printf("\n=== Done ===\n");
    return 0;
}
EOF

/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o debug_scale /tmp/debug_scale.cu 2>/dev/null && timeout 60 ./debug_scale