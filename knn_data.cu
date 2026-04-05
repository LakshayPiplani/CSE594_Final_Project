#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

const int D = 128; // Dimensions
const int C = 1024; // Number of Clusters (Centroids)
const int MAX_ITERS = 20; // K-Means iterations

// ============================================================================
// 1. Fast GPU Dummy Data Generator
// Uses a simple hashing algorithm to generate floats between 0.0 and 1.0
// ============================================================================
__global__ void generate_dummy_vectors(float* data, size_t total_elements, int seed) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_elements) {
        unsigned int hash = tid * 1664525u + 1013904223u + seed;
        hash ^= hash >> 16; hash *= 0x85ebca6b;
        hash ^= hash >> 13; hash *= 0xc2b2ae35;
        hash ^= hash >> 16;
        data[tid] = (float)(hash & 0x00FFFFFF) / (float)0x01000000;
    }
}

// ============================================================================
// 2. K-Means Assignment Phase
// Finds the closest centroid for each vector in the sample
// ============================================================================
__global__ void kmeans_assign(const float* sample_data, const float* centroids, 
                              int* assignments, int num_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;

    const float* my_vec = &sample_data[tid * D];
    float min_dist = 1e30f;
    int best_c = 0;

    for (int c = 0; c < C; c++) {
        const float* c_vec = &centroids[c * D];
        float dist = 0.0f;
        
        // Compute L2 Distance squared
        for (int d = 0; d < D; d++) {
            float diff = my_vec[d] - c_vec[d];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_c = c;
        }
    }
    assignments[tid] = best_c;
}

// ============================================================================
// 3. K-Means Update Phase
// Accumulates the vectors into their assigned centroids
// ============================================================================
__global__ void kmeans_update(const float* sample_data, const int* assignments, 
                              float* new_centroids, int* counts, int num_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;

    int c = assignments[tid];
    atomicAdd(&counts[c], 1);
    
    const float* my_vec = &sample_data[tid * D];
    float* target_centroid = &new_centroids[c * D];

    for (int d = 0; d < D; d++) {
        atomicAdd(&target_centroid[d], my_vec[d]);
    }
}

// ============================================================================
// 4. K-Means Averaging Phase
// Divides the accumulated sums by the cluster counts
// ============================================================================
__global__ void kmeans_average(float* new_centroids, const int* counts) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    int count = counts[c];
    if (count > 0) {
        float* my_centroid = &new_centroids[c * D];
        for (int d = 0; d < D; d++) {
            my_centroid[d] /= (float)count;
        }
    }
}

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv) {
    int scale = 20; // Default: 2^20 = ~1 Million vectors
    if (argc > 1) scale = atoi(argv[1]);

    size_t num_vectors = 1ULL << scale;
    size_t num_samples = num_vectors / 2; // 50% sample
    
    size_t data_bytes = num_vectors * D * sizeof(float);
    size_t centroid_bytes = C * D * sizeof(float);

    printf("=== IVF-Flat Training Phase (Scale %d) ===\n", scale);
    printf("Total Vectors: %zu (%.2f GB)\n", num_vectors, (double)data_bytes / (1024*1024*1024));
    printf("Sample Size:   %zu (50%%)\n", num_samples);
    printf("Centroids:     %d\n\n", C);

    // 1. Allocate VRAM
    float *d_data, *d_centroids, *d_new_centroids;
    int *d_assignments, *d_counts;

    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroid_bytes));
    CHECK_CUDA(cudaMalloc(&d_new_centroids, centroid_bytes));
    CHECK_CUDA(cudaMalloc(&d_assignments, num_samples * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_counts, C * sizeof(int)));

    // 2. Generate Dummy Data on GPU
    printf("Generating dummy vectors...\n");
    int threads = 256;
    size_t total_elements = num_vectors * D;
    size_t blocks = (total_elements + threads - 1) / threads;
    generate_dummy_vectors<<<blocks, threads>>>(d_data, total_elements, 42);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3. Initialize Centroids (Steal the first C vectors from the dataset)
    CHECK_CUDA(cudaMemcpy(d_centroids, d_data, centroid_bytes, cudaMemcpyDeviceToDevice));

    // 4. K-Means Training Loop
    printf("Starting K-Means (%d iterations)...\n", MAX_ITERS);
    int assign_blocks = (num_samples + threads - 1) / threads;
    int avg_blocks = (C + threads - 1) / threads;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        // Zero out the accumulators
        CHECK_CUDA(cudaMemset(d_new_centroids, 0, centroid_bytes));
        CHECK_CUDA(cudaMemset(d_counts, 0, C * sizeof(int)));

        // Assign
        kmeans_assign<<<assign_blocks, threads>>>(d_data, d_centroids, d_assignments, num_samples);
        
        // Update
        kmeans_update<<<assign_blocks, threads>>>(d_data, d_assignments, d_new_centroids, d_counts, num_samples);
        
        // Average
        kmeans_average<<<avg_blocks, threads>>>(d_new_centroids, d_counts);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Swap pointers
        float* temp = d_centroids;
        d_centroids = d_new_centroids;
        d_new_centroids = temp;
        
        printf("  Iteration %d complete.\n", iter + 1);
    }

    // 5. Write to Disk for BaM
    printf("\nWriting data to disk for BaM backing store...\n");
    
    std::vector<float> h_centroids(C * D);
    CHECK_CUDA(cudaMemcpy(h_centroids.data(), d_centroids, centroid_bytes, cudaMemcpyDeviceToHost));
    std::ofstream c_file("centroids.bin", std::ios::binary);
    c_file.write(reinterpret_cast<char*>(h_centroids.data()), centroid_bytes);
    c_file.close();
    printf("  -> Saved centroids.bin\n");

    // We copy the data back in chunks so we don't blow up the Host RAM
    std::ofstream d_file("raw_vectors.bin", std::ios::binary);
    std::vector<float> h_chunk(100000 * D); // 100k vectors at a time
    size_t vectors_written = 0;
    
    while (vectors_written < num_vectors) {
        size_t write_count = std::min((size_t)100000, num_vectors - vectors_written);
        size_t bytes = write_count * D * sizeof(float);
        CHECK_CUDA(cudaMemcpy(h_chunk.data(), d_data + (vectors_written * D), bytes, cudaMemcpyDeviceToHost));
        d_file.write(reinterpret_cast<char*>(h_chunk.data()), bytes);
        vectors_written += write_count;
    }
    d_file.close();
    printf("  -> Saved raw_vectors.bin\n");

    // Cleanup
    cudaFree(d_data); cudaFree(d_centroids); cudaFree(d_new_centroids);
    cudaFree(d_assignments); cudaFree(d_counts);

    printf("Done.\n");
    return 0;
}