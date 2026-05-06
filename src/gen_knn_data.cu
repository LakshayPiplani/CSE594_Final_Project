#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <algorithm>

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
// 5. CSR Offset Builder
// Finds the boundaries where one cluster ends and the next begins
// ============================================================================
__global__ void build_cluster_offsets(const int* sorted_assignments, int* offsets, size_t num_vectors, int C) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handle the very first element and the absolute end boundary
    if (tid == 0) {
        offsets[C] = num_vectors;
    } 
    else if (tid < num_vectors) {
        int my_c = sorted_assignments[tid];
        int prev_c = sorted_assignments[tid - 1];
        
        // If the cluster ID changed, we found a boundary
        if (my_c != prev_c) {
            // Fill in the offsets for all clusters between prev_c and my_c
            // (This handles empty clusters perfectly)
            for (int c = prev_c + 1; c <= my_c; c++) {
                offsets[c] = tid;
            }
        }
    }
}

// ============================================================================
// 6. Physical Reordering Kernel
// Shuffles the raw floats so vectors in the same cluster sit contiguously
// ============================================================================
__global__ void reorder_vectors(const float* old_data, float* new_data, const int* sorted_indices, size_t num_vectors, int D) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vectors) {
        int old_vec_id = sorted_indices[tid];
        for (int d = 0; d < D; d++) {
            new_data[tid * D + d] = old_data[old_vec_id * D + d];
        }
    }
}

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv) {
    int scale = 20; // Default: 2^20 = ~1 Million vectors
    std::string data_dir = "./data";

        for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data_dir") == 0 && i + 1 < argc) {
            data_dir = std::string(argv[++i]);
        } 
        else if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            scale = atoi(argv[++i]);
        } 
 
        else {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --data_dir <path>   : Data directory\n");
            printf("  --scale <N>    : Number of vectors (2^N). Default: 20\n");
            exit(0);
        }
    }

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
// ========================================================================
    // 5. BUILD THE INVERTED INDEX (CSR)
    // ========================================================================
    printf("\nBuilding CSR Inverted Index...\n");

    // 5a. Full Assignment (100% of data)
    int* d_full_assignments;
    CHECK_CUDA(cudaMalloc(&d_full_assignments, num_vectors * sizeof(int)));
    int full_blocks = (num_vectors + threads - 1) / threads;
    kmeans_assign<<<full_blocks, threads>>>(d_data, d_centroids, d_full_assignments, num_vectors);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5b. Generate Sequence of Vector IDs (0, 1, 2, ... N-1)
    int* d_vector_indices;
    CHECK_CUDA(cudaMalloc(&d_vector_indices, num_vectors * sizeof(int)));
    thrust::sequence(thrust::device, d_vector_indices, d_vector_indices + num_vectors);

    // 5c. Sort IDs by Cluster Assignment
    printf("  Sorting vector IDs by cluster...\n");
    thrust::sort_by_key(thrust::device, d_full_assignments, d_full_assignments + num_vectors, d_vector_indices);

    // 5d. Build Offsets
    int* d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, (C + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_offsets, 0, (C + 1) * sizeof(int))); // Default to 0 for empty clusters
    
    build_cluster_offsets<<<full_blocks, threads>>>(d_full_assignments, d_offsets, num_vectors, C);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5e. Physically Reorder the Raw Vectors
    printf("  Physically reordering raw data for BaM alignment...\n");
    float* d_reordered_data;
    CHECK_CUDA(cudaMalloc(&d_reordered_data, data_bytes));
    reorder_vectors<<<full_blocks, threads>>>(d_data, d_reordered_data, d_vector_indices, num_vectors, D);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ========================================================================
    // 6. WRITE BINARY FILES TO DISK
    // ========================================================================
    printf("\nWriting CSR files to disk...\n");
    std::string centroids_file = data_dir + "centroids.bin";

    // Centroids (C * D floats)
    std::vector<float> h_centroids(C * D);
    CHECK_CUDA(cudaMemcpy(h_centroids.data(), d_centroids, centroid_bytes, cudaMemcpyDeviceToHost));
    std::ofstream c_file(centroids_file, std::ios::binary);
    c_file.write(reinterpret_cast<char*>(h_centroids.data()), centroid_bytes);
    c_file.close();
    printf("  -> Saved centroids.bin\n");

    // Offsets (C + 1 ints)
    std::string offsets_file = data_dir + "cluster_offsets.bin";
    std::vector<int> h_offsets(C + 1);
    CHECK_CUDA(cudaMemcpy(h_offsets.data(), d_offsets, (C + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    std::ofstream o_file(offsets_file, std::ios::binary);
    o_file.write(reinterpret_cast<char*>(h_offsets.data()), (C + 1) * sizeof(int));
    o_file.close();
    printf("  -> Saved cluster_offsets.bin\n");

    // Vector Indices (N ints)
    // (Note: Since we physically sorted the data, target_vec_id == its physical index,
    // but we save this file so you know the ORIGINAL vector IDs to return to the user).
    std::string vec_idx_file = data_dir + "vector_indices.bin";
    std::ofstream i_file(vec_idx_file, std::ios::binary);
    std::vector<int> h_idx_chunk(1000000);
    size_t written = 0;
    while (written < num_vectors) {
        size_t chunk_size = std::min((size_t)1000000, num_vectors - written);
        CHECK_CUDA(cudaMemcpy(h_idx_chunk.data(), d_vector_indices + written, chunk_size * sizeof(int), cudaMemcpyDeviceToHost));
        i_file.write(reinterpret_cast<char*>(h_idx_chunk.data()), chunk_size * sizeof(int));
        written += chunk_size;
    }
    i_file.close();
    printf("  -> Saved vector_indices.bin\n");

    // Reordered Raw Vectors (N * D floats)
    std::string vec_file = data_dir + "raw_vectors.bin";
    std::ofstream d_file(vec_file, std::ios::binary);
    std::vector<float> h_data_chunk(100000 * D);
    written = 0;
    while (written < num_vectors) {
        size_t chunk_size = std::min((size_t)100000, num_vectors - written);
        CHECK_CUDA(cudaMemcpy(h_data_chunk.data(), d_reordered_data + (written * D), chunk_size * D * sizeof(float), cudaMemcpyDeviceToHost));
        d_file.write(reinterpret_cast<char*>(h_data_chunk.data()), chunk_size * D * sizeof(float));
        written += chunk_size;
    }
    d_file.close();
    printf("  -> Saved raw_vectors.bin (Physically sorted by cluster)\n");

    // Cleanup
    cudaFree(d_data); cudaFree(d_centroids); cudaFree(d_new_centroids);
    cudaFree(d_assignments); cudaFree(d_counts);
    cudaFree(d_full_assignments); cudaFree(d_vector_indices); 
    cudaFree(d_offsets); cudaFree(d_reordered_data);

    // ========================================================================
    // ANALYZE CLUSTER DISTRIBUTION
    // ========================================================================
    printf("\nAnalyzing Cluster Size Distribution...\n");
    int min_size = num_vectors;
    int max_size = 0;
    int empty_clusters = 0;

    // Create a vector to hold the sizes so we can sort them
    std::vector<int> cluster_sizes(C);

    std::ofstream dist_file("cluster_distribution.csv");
    dist_file << "ClusterID,Size\n";

    for (int c = 0; c < C; c++) {
        int cluster_size = h_offsets[c + 1] - h_offsets[c];
        cluster_sizes[c] = cluster_size;
        
        dist_file << c << "," << cluster_size << "\n";

        if (cluster_size < min_size) min_size = cluster_size;
        if (cluster_size > max_size) max_size = cluster_size;
        if (cluster_size == 0) empty_clusters++;
    }
    dist_file.close();

    // Sort the array from smallest to largest
    std::sort(cluster_sizes.begin(), cluster_sizes.end());

    // Helper lambda to safely calculate median of a sub-array
    auto get_median = [](const std::vector<int>& v, int start, int end) -> double {
        int len = end - start;
        if (len % 2 == 0) {
            return (v[start + len / 2 - 1] + v[start + len / 2]) / 2.0;
        } else {
            return v[start + len / 2];
        }
    };

    // Calculate percentiles
    double median = get_median(cluster_sizes, 0, C);
    double q1     = get_median(cluster_sizes, 0, C / 2);
    double q3     = get_median(cluster_sizes, C - (C / 2), C);

    printf("  Min Cluster Size : %d vectors\n", min_size);
    printf("  Q1 (25th Perc.)  : %.1f vectors\n", q1);
    printf("  Median (50th)    : %.1f vectors\n", median);
    printf("  Q3 (75th Perc.)  : %.1f vectors\n", q3);
    printf("  Max Cluster Size : %d vectors\n", max_size);
    printf("  Avg Cluster Size : %zu vectors\n", num_vectors / C);
    printf("  Empty Clusters   : %d\n", empty_clusters);

    printf("Done.\n");
    return 0;
}