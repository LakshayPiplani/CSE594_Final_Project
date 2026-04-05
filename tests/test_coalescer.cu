#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "coalescer.cuh"

// ============================================================================
// Phase 1 Microbenchmark: Warp Coalescer
//
// Tests three approaches to warp-level data access:
// 1. Naive: every thread reads independently (no coalescing)
// 2. BaM coalesced: __match_any_sync grouping + leader + broadcast
// 3. Serialized (ActivePointers-style): loop through lanes one by one
//
// We vary the number of unique cache lines per warp to show:
// - When threads share cache lines, BaM coalescing reduces probes
// - When all threads need unique lines, all approaches are similar
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---- Kernels ----

// Kernel 1: Naive reads — every thread reads its element directly
__global__
void kernel_naive(const int* data, const int* indices, int* output,
                  int num_accesses) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_accesses) return;

    int index = indices[tid];
    output[tid] = naive_read(data, index);
}

// Kernel 2: BaM-style coalesced reads
__global__
void kernel_coalesced(const int* cache_data, const FakeSlotMeta* meta,
                      int num_slots, const int* indices, int* output,
                      int num_accesses, int* probe_counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_accesses) return;

    int index = indices[tid];
    output[tid] = coalesced_cache_read(cache_data, meta, num_slots,
                                        index, probe_counter);
}

// Kernel 3: ActivePointers-style serialized coalescing
__global__
void kernel_serialized(const int* cache_data, const FakeSlotMeta* meta,
                       int num_slots, const int* indices, int* output,
                       int num_accesses, int* probe_counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_accesses) return;

    int index = indices[tid];
    output[tid] = serialized_coalesced_read(cache_data, meta, num_slots,
                                             index, probe_counter);
}

// ---- L2 cache flush ----

__global__
void flush_l2(char* buf, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    volatile char x;
    for (int i = tid; i < size; i += gridDim.x * blockDim.x) {
        x = buf[i];
    }
}

// ---- Index generation ----
// Controls how many unique cache lines each warp accesses

void generate_indices(int* h_indices, int num_accesses, int num_elements,
                      int unique_cls_per_warp) {
    // Each warp of 32 threads accesses 'unique_cls_per_warp' unique cache lines
    // Threads within a warp are assigned to cache lines round-robin
    srand(42);

    for (int i = 0; i < num_accesses; i++) {
        int warp_id    = i / 32;
        int lane       = i % 32;
        int cl_group   = lane % unique_cls_per_warp;

        // Each warp picks 'unique_cls_per_warp' random cache lines
        // Use warp_id as seed offset so different warps get different lines
        unsigned seed = (unsigned)(warp_id * unique_cls_per_warp + cl_group);
        seed = seed * 2654435761u;  // hash
        int cl_id = seed % (num_elements / ELEMS_PER_CL_INT);

        // Pick a random offset within that cache line
        int offset = lane % ELEMS_PER_CL_INT;
        h_indices[i] = cl_id * ELEMS_PER_CL_INT + offset;
    }
}

// ---- Benchmark runner ----

struct BenchResult {
    float time_ms;
    int   total_probes;
    float bandwidth_gbps;
};

BenchResult run_naive(const int* d_data, const int* d_indices, int* d_output,
                      int num_accesses, char* d_flush, int flush_size) {
    BenchResult result = {};
    int block = 256;
    int grid  = (num_accesses + block - 1) / block;

    // Warmup
    kernel_naive<<<grid, block>>>(d_data, d_indices, d_output, num_accesses);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Flush L2
    flush_l2<<<256, 256>>>(d_flush, flush_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel_naive<<<grid, block>>>(d_data, d_indices, d_output, num_accesses);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.total_probes = num_accesses;  // every thread probes
    result.bandwidth_gbps = ((double)num_accesses * sizeof(int) / 1e9)
                            / (result.time_ms / 1e3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}

BenchResult run_coalesced(const int* d_cache_data, const FakeSlotMeta* d_meta,
                          int num_slots, const int* d_indices, int* d_output,
                          int num_accesses, char* d_flush, int flush_size,
                          bool use_serialized) {
    BenchResult result = {};
    int block = 256;
    int grid  = (num_accesses + block - 1) / block;

    // Probe counter
    int* d_probes;
    CHECK_CUDA(cudaMalloc(&d_probes, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_probes, 0, sizeof(int)));

    // Warmup
    if (use_serialized) {
        kernel_serialized<<<grid, block>>>(d_cache_data, d_meta, num_slots,
                                           d_indices, d_output, num_accesses,
                                           nullptr);
    } else {
        kernel_coalesced<<<grid, block>>>(d_cache_data, d_meta, num_slots,
                                          d_indices, d_output, num_accesses,
                                          nullptr);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Flush L2
    flush_l2<<<256, 256>>>(d_flush, flush_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset probe counter
    CHECK_CUDA(cudaMemset(d_probes, 0, sizeof(int)));

    // Timed run
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    if (use_serialized) {
        kernel_serialized<<<grid, block>>>(d_cache_data, d_meta, num_slots,
                                           d_indices, d_output, num_accesses,
                                           d_probes);
    } else {
        kernel_coalesced<<<grid, block>>>(d_cache_data, d_meta, num_slots,
                                          d_indices, d_output, num_accesses,
                                          d_probes);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&result.time_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(&result.total_probes, d_probes, sizeof(int),
                           cudaMemcpyDeviceToHost));
    result.bandwidth_gbps = ((double)num_accesses * sizeof(int) / 1e9)
                            / (result.time_ms / 1e3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_probes);
    return result;
}

// ---- Main ----

int main() {
    printf("=== BaM Warp Coalescer Microbenchmark ===\n");
    printf("GPU: RTX A4500, sm_86, 19.7GB VRAM, 56 SMs\n");
    printf("Cache line size: %d bytes (%d int elements)\n\n",
           CACHE_LINE_SIZE, ELEMS_PER_CL_INT);

    // Setup: 1GB data buffer organized as cache lines
    const int NUM_ELEMENTS = 256 * 1024 * 1024;  // 256M ints = 1GB
    const int NUM_SLOTS    = NUM_ELEMENTS / ELEMS_PER_CL_INT;  // 262144 slots
    const int NUM_ACCESSES = 16 * 1024 * 1024;   // 16M accesses per test

    // Allocate data buffer (flat, as if it's a cache data region)
    int* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, (size_t)NUM_ELEMENTS * sizeof(int)));

    // Initialize data: d_data[i] = i (so we can verify correctness)
    {
        int* h_data = (int*)malloc((size_t)NUM_ELEMENTS * sizeof(int));
        for (int i = 0; i < NUM_ELEMENTS; i++) h_data[i] = i;
        CHECK_CUDA(cudaMemcpy(d_data, h_data, (size_t)NUM_ELEMENTS * sizeof(int),
                               cudaMemcpyHostToDevice));
        free(h_data);
    }

    // Allocate fake cache metadata (all slots valid, tag = slot index)
    FakeSlotMeta* h_meta = (FakeSlotMeta*)malloc(NUM_SLOTS * sizeof(FakeSlotMeta));
    for (int i = 0; i < NUM_SLOTS; i++) {
        h_meta[i].tag   = i;
        h_meta[i].valid = 1;
    }
    FakeSlotMeta* d_meta;
    CHECK_CUDA(cudaMalloc(&d_meta, NUM_SLOTS * sizeof(FakeSlotMeta)));
    CHECK_CUDA(cudaMemcpy(d_meta, h_meta, NUM_SLOTS * sizeof(FakeSlotMeta),
                           cudaMemcpyHostToDevice));
    free(h_meta);

    // Allocate index and output arrays
    int* h_indices = (int*)malloc(NUM_ACCESSES * sizeof(int));
    int* d_indices;
    int* d_output;
    CHECK_CUDA(cudaMalloc(&d_indices, NUM_ACCESSES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output,  NUM_ACCESSES * sizeof(int)));

    // L2 flush buffer (larger than 5MB L2)
    char* d_flush;
    int flush_size = 8 * 1024 * 1024;  // 8MB
    CHECK_CUDA(cudaMalloc(&d_flush, flush_size));

    // ---- Run benchmarks ----
    // Vary unique_cls_per_warp: how many distinct cache lines each warp accesses
    // 1  = all 32 threads hit same cache line (maximum coalescing benefit)
    // 4  = 4 unique lines, ~8 threads per line
    // 8  = 8 unique lines, ~4 threads per line
    // 16 = 16 unique lines, ~2 threads per line
    // 32 = all threads hit different lines (no coalescing benefit)

    int unique_counts[] = {1, 2, 4, 8, 16, 32};
    int num_tests = sizeof(unique_counts) / sizeof(unique_counts[0]);

    printf("%-12s | %-10s %-10s %-10s | %-10s %-10s %-10s | %-10s %-10s %-10s\n",
           "Unique CLs", "Naive ms", "Naive BW", "Probes",
           "BaM ms", "BaM BW", "Probes",
           "Serial ms", "Serial BW", "Probes");
    printf("%-12s-+-%-10s-%-10s-%-10s-+-%-10s-%-10s-%-10s-+-%-10s-%-10s-%-10s\n",
           "------------", "----------", "----------", "----------",
           "----------", "----------", "----------",
           "----------", "----------", "----------");

    for (int t = 0; t < num_tests; t++) {
        int unique_cls = unique_counts[t];

        // Generate indices for this configuration
        generate_indices(h_indices, NUM_ACCESSES, NUM_ELEMENTS, unique_cls);
        CHECK_CUDA(cudaMemcpy(d_indices, h_indices, NUM_ACCESSES * sizeof(int),
                               cudaMemcpyHostToDevice));

        // Run naive
        BenchResult naive = run_naive(d_data, d_indices, d_output,
                                      NUM_ACCESSES, d_flush, flush_size);

        // Run BaM coalesced
        BenchResult bam = run_coalesced(d_data, d_meta, NUM_SLOTS,
                                         d_indices, d_output, NUM_ACCESSES,
                                         d_flush, flush_size, false);

        // Run serialized (ActivePointers-style)
        BenchResult serial = run_coalesced(d_data, d_meta, NUM_SLOTS,
                                            d_indices, d_output, NUM_ACCESSES,
                                            d_flush, flush_size, true);

        printf("%-12d | %-10.3f %-10.2f %-10d | %-10.3f %-10.2f %-10d | %-10.3f %-10.2f %-10d\n",
               unique_cls,
               naive.time_ms,  naive.bandwidth_gbps,  naive.total_probes,
               bam.time_ms,    bam.bandwidth_gbps,    bam.total_probes,
               serial.time_ms, serial.bandwidth_gbps, serial.total_probes);
    }

    // ---- Correctness check ----
    printf("\n--- Correctness Check ---\n");
    generate_indices(h_indices, NUM_ACCESSES, NUM_ELEMENTS, 4);
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices, NUM_ACCESSES * sizeof(int),
                           cudaMemcpyHostToDevice));

    // Run naive
    int* h_naive_out = (int*)malloc(NUM_ACCESSES * sizeof(int));
    kernel_naive<<<(NUM_ACCESSES+255)/256, 256>>>(
        d_data, d_indices, d_output, NUM_ACCESSES);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_naive_out, d_output, NUM_ACCESSES * sizeof(int),
                           cudaMemcpyDeviceToHost));

    // Run BaM coalesced
    int* h_bam_out = (int*)malloc(NUM_ACCESSES * sizeof(int));
    kernel_coalesced<<<(NUM_ACCESSES+255)/256, 256>>>(
        d_data, d_meta, NUM_SLOTS, d_indices, d_output, NUM_ACCESSES, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_bam_out, d_output, NUM_ACCESSES * sizeof(int),
                           cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < NUM_ACCESSES; i++) {
        if (h_naive_out[i] != h_bam_out[i]) {
            if (mismatches < 5) {
                printf("MISMATCH at %d: naive=%d, bam=%d (index=%d)\n",
                       i, h_naive_out[i], h_bam_out[i], h_indices[i]);
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASSED: All %d values match between naive and BaM coalesced.\n",
               NUM_ACCESSES);
    } else {
        printf("FAILED: %d mismatches out of %d.\n", mismatches, NUM_ACCESSES);
    }

    // Cleanup
    free(h_indices);
    free(h_naive_out);
    free(h_bam_out);
    cudaFree(d_data);
    cudaFree(d_meta);
    cudaFree(d_indices);
    cudaFree(d_output);
    cudaFree(d_flush);

    return 0;
}
