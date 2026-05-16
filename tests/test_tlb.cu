#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cache.cuh"
#include "tlb.cuh"

// ============================================================================
// Test harness for TLB implementation
//
// Tests:
//   1. Correctness: TLB reads produce same values as direct reads
//   2. Sequential pattern: edges[0], edges[1], edges[2], ... (TLB should
//      hit ~1023 out of every 1024 accesses)
//   3. Random pattern: random indices (TLB misses more, but still correct)
//   4. BFS-like pattern: for each node, iterate its neighbor list
//   5. Performance: TLB vs no-TLB vs direct PCIe timing comparison
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// --- Kernel: TLB sequential read ---
__global__
void kernel_tlb_sequential(BamCache cache, int* output, int n,
                           const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    BamTLB tlb;
    tlb_init(&tlb);

    // Each thread reads a contiguous chunk of 64 elements
    int base = tid * 64;
    for (int i = 0; i < 64; i++) {
        int idx = base + i;
        if (idx < n * 64)
            output[tid * 64 + i] = tlb_read(&cache, idx, backing, &tlb);
    }

    tlb_release(&cache, &tlb);
}

// --- Kernel: TLB random read ---
__global__
void kernel_tlb_random(BamCache cache, const int* indices, int* output,
                       int n, const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    BamTLB tlb;
    tlb_init(&tlb);

    output[tid] = tlb_read(&cache, indices[tid], backing, &tlb);

    tlb_release(&cache, &tlb);
}

// --- Kernel: No-TLB coalesced read (baseline) ---
__global__
void kernel_no_tlb_sequential(BamCache cache, int* output, int n,
                              const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base = tid * 64;
    for (int i = 0; i < 64; i++) {
        int idx = base + i;
        if (idx < n * 64)
            output[tid * 64 + i] = cache_read_no_coalesce(&cache, idx, backing);
    }
}

// --- Kernel: Direct PCIe read (no cache at all) ---
__global__
void kernel_direct_sequential(const int* backing, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base = tid * 64;
    for (int i = 0; i < 64; i++) {
        int idx = base + i;
        if (idx < n * 64)
            output[tid * 64 + i] = backing[idx];
    }
}

// --- Kernel: BFS-like pattern with TLB ---
// Each thread "owns" a node, iterates its neighbor list
__global__
void kernel_tlb_bfs_pattern(BamCache cache, const int* offsets,
                            int* output, int num_nodes,
                            const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    BamTLB tlb;
    tlb_init(&tlb);

    int start = offsets[tid];
    int end   = offsets[tid + 1];
    int sum = 0;

    for (int e = start; e < end; e++) {
        sum += tlb_read(&cache, e, backing, &tlb);
    }

    output[tid] = sum;
    tlb_release(&cache, &tlb);
}

// Same but without TLB
__global__
void kernel_no_tlb_bfs_pattern(BamCache cache, const int* offsets,
                               int* output, int num_nodes,
                               const char* backing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    int start = offsets[tid];
    int end   = offsets[tid + 1];
    int sum = 0;

    for (int e = start; e < end; e++) {
        sum += cache_read_no_coalesce(&cache, e, backing);
    }

    output[tid] = sum;
}

__global__ void flush_l2(char* buf, int sz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    volatile char x;
    for (int i = tid; i < sz; i += gridDim.x * blockDim.x) x = buf[i];
}

int main() {
    printf("=== TLB Implementation Test ===\n\n");

    // Setup: 4M ints = 16MB backing store, 4096 logical pages
    const int NUM_ELEMS = 4 * 1024 * 1024;
    const uint32_t NUM_PAGES = NUM_ELEMS / CL_ELEMS_INT;
    const int NUM_THREADS = 1024;

    int* h_backing;
    CHECK_CUDA(cudaHostAlloc(&h_backing, NUM_ELEMS * sizeof(int),
                              cudaHostAllocDefault));
    for (int i = 0; i < NUM_ELEMS; i++) h_backing[i] = i;

    // Output buffers
    int out_size = NUM_THREADS * 64;
    int* d_out; CHECK_CUDA(cudaMalloc(&d_out, out_size * sizeof(int)));
    int* h_out = (int*)malloc(out_size * sizeof(int));

    char* d_flush; CHECK_CUDA(cudaMalloc(&d_flush, 8*1024*1024));

    // 100% cache to avoid eviction issues during testing
    BamCache cache;
    cache_init(&cache, (size_t)NUM_PAGES * CACHE_LINE_SIZE, NUM_PAGES);

    int blk = 256, grd = (NUM_THREADS + blk - 1) / blk;

    // ================================================================
    // Test 1: Sequential correctness
    // ================================================================
    printf("--- Test 1: Sequential correctness ---\n");
    {
        cache_reset(&cache);
        CHECK_CUDA(cudaMemset(d_out, 0, out_size * sizeof(int)));

        kernel_tlb_sequential<<<grd, blk>>>(cache, d_out, NUM_THREADS,
                                             (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_out, d_out, out_size * sizeof(int),
                               cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < out_size; i++) {
            if (h_out[i] != i) {
                if (errs < 5)
                    printf("  ERR[%d]: expected %d got %d\n", i, i, h_out[i]);
                errs++;
            }
        }
        printf("  %s (%d errors / %d)\n\n",
               errs == 0 ? "PASSED" : "FAILED", errs, out_size);
    }

    // ================================================================
    // Test 2: Random correctness
    // ================================================================
    printf("--- Test 2: Random access correctness ---\n");
    {
        cache_reset(&cache);
        int* h_idx = (int*)malloc(NUM_THREADS * sizeof(int));
        int* d_idx; CHECK_CUDA(cudaMalloc(&d_idx, NUM_THREADS * sizeof(int)));
        int* d_out2; CHECK_CUDA(cudaMalloc(&d_out2, NUM_THREADS * sizeof(int)));

        srand(42);
        for (int i = 0; i < NUM_THREADS; i++)
            h_idx[i] = rand() % NUM_ELEMS;
        CHECK_CUDA(cudaMemcpy(d_idx, h_idx, NUM_THREADS * sizeof(int),
                               cudaMemcpyHostToDevice));

        kernel_tlb_random<<<grd, blk>>>(cache, d_idx, d_out2, NUM_THREADS,
                                         (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        int* h_out2 = (int*)malloc(NUM_THREADS * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_out2, d_out2, NUM_THREADS * sizeof(int),
                               cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            if (h_out2[i] != h_idx[i]) {
                if (errs < 5)
                    printf("  ERR[%d]: idx=%d expected=%d got=%d\n",
                           i, h_idx[i], h_idx[i], h_out2[i]);
                errs++;
            }
        }
        printf("  %s (%d errors / %d)\n\n",
               errs == 0 ? "PASSED" : "FAILED", errs, NUM_THREADS);

        free(h_idx); free(h_out2);
        cudaFree(d_idx); cudaFree(d_out2);
    }

    // ================================================================
    // Test 3: BFS-like pattern correctness
    // ================================================================
    printf("--- Test 3: BFS-like pattern correctness ---\n");
    {
        // Create fake CSR: each node has degree ~32
        int num_nodes = 4096;
        int* h_offsets = (int*)malloc((num_nodes + 1) * sizeof(int));
        srand(123);
        h_offsets[0] = 0;
        for (int i = 0; i < num_nodes; i++) {
            int deg = 16 + (rand() % 32);  // degree 16..47
            h_offsets[i+1] = h_offsets[i] + deg;
            if (h_offsets[i+1] >= NUM_ELEMS)
                h_offsets[i+1] = NUM_ELEMS;
        }

        int* d_offsets;
        CHECK_CUDA(cudaMalloc(&d_offsets, (num_nodes+1) * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, (num_nodes+1) * sizeof(int),
                               cudaMemcpyHostToDevice));

        int* d_out_tlb; CHECK_CUDA(cudaMalloc(&d_out_tlb, num_nodes * sizeof(int)));
        int* d_out_notlb; CHECK_CUDA(cudaMalloc(&d_out_notlb, num_nodes * sizeof(int)));

        cache_reset(&cache);
        int grd2 = (num_nodes + blk - 1) / blk;

        kernel_tlb_bfs_pattern<<<grd2, blk>>>(cache, d_offsets, d_out_tlb,
                                               num_nodes, (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        cache_reset(&cache);
        kernel_no_tlb_bfs_pattern<<<grd2, blk>>>(cache, d_offsets, d_out_notlb,
                                                  num_nodes, (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        int* h_tlb = (int*)malloc(num_nodes * sizeof(int));
        int* h_notlb = (int*)malloc(num_nodes * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_tlb, d_out_tlb, num_nodes * sizeof(int),
                               cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_notlb, d_out_notlb, num_nodes * sizeof(int),
                               cudaMemcpyDeviceToHost));

        int errs = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (h_tlb[i] != h_notlb[i]) {
                if (errs < 5)
                    printf("  ERR node %d: TLB=%d  noTLB=%d\n",
                           i, h_tlb[i], h_notlb[i]);
                errs++;
            }
        }
        printf("  %s (%d mismatches / %d nodes)\n\n",
               errs == 0 ? "PASSED" : "FAILED", errs, num_nodes);

        free(h_offsets); free(h_tlb); free(h_notlb);
        cudaFree(d_offsets); cudaFree(d_out_tlb); cudaFree(d_out_notlb);
    }

    // ================================================================
    // Test 4: Performance comparison
    // ================================================================
    printf("--- Test 4: Performance (sequential, %d threads x 64 reads) ---\n",
           NUM_THREADS);
    {
        cache_reset(&cache);

        // Warmup
        kernel_tlb_sequential<<<grd, blk>>>(cache, d_out, NUM_THREADS,
                                             (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CHECK_CUDA(cudaEventCreate(&t0));
        CHECK_CUDA(cudaEventCreate(&t1));

        // TLB
        flush_l2<<<256,256>>>(d_flush, 8*1024*1024);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaMemset(cache.d_hits, 0, sizeof(unsigned long long));
        cudaMemset(cache.d_misses, 0, sizeof(unsigned long long));

        CHECK_CUDA(cudaEventRecord(t0));
        kernel_tlb_sequential<<<grd, blk>>>(cache, d_out, NUM_THREADS,
                                             (const char*)h_backing);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms_tlb;
        CHECK_CUDA(cudaEventElapsedTime(&ms_tlb, t0, t1));
        CacheStats s_tlb = cache_get_stats(&cache);

        // No TLB
        cache_reset(&cache);
        kernel_no_tlb_sequential<<<grd, blk>>>(cache, d_out, NUM_THREADS,
                                                (const char*)h_backing);
        CHECK_CUDA(cudaDeviceSynchronize());

        flush_l2<<<256,256>>>(d_flush, 8*1024*1024);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaMemset(cache.d_hits, 0, sizeof(unsigned long long));
        cudaMemset(cache.d_misses, 0, sizeof(unsigned long long));

        CHECK_CUDA(cudaEventRecord(t0));
        kernel_no_tlb_sequential<<<grd, blk>>>(cache, d_out, NUM_THREADS,
                                                (const char*)h_backing);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms_notlb;
        CHECK_CUDA(cudaEventElapsedTime(&ms_notlb, t0, t1));
        CacheStats s_notlb = cache_get_stats(&cache);

        // Direct PCIe
        flush_l2<<<256,256>>>(d_flush, 8*1024*1024);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(t0));
        kernel_direct_sequential<<<grd, blk>>>((const int*)h_backing, d_out,
                                                NUM_THREADS);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms_direct;
        CHECK_CUDA(cudaEventElapsedTime(&ms_direct, t0, t1));

        printf("  TLB:        %.3f ms  (hits=%llu misses=%llu)\n",
               ms_tlb, s_tlb.hits, s_tlb.misses);
        printf("  No TLB:     %.3f ms  (hits=%llu misses=%llu)\n",
               ms_notlb, s_notlb.hits, s_notlb.misses);
        printf("  Direct PCIe: %.3f ms\n", ms_direct);
        if (ms_notlb > 0)
            printf("  TLB speedup over no-TLB: %.2fx\n", ms_notlb / ms_tlb);
        printf("\n");

        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    free(h_out);
    cudaFreeHost(h_backing);
    cudaFree(d_out); cudaFree(d_flush);
    cache_destroy(&cache);
    printf("=== All tests complete ===\n");
    return 0;
}
