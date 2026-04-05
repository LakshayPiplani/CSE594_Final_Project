#ifndef BAM_COALESCER_CUH
#define BAM_COALESCER_CUH

#include <cstdint>

// ============================================================================
// BaM Warp Coalescer
//
// Groups threads in a warp that need the same cache line, elects one leader
// per group, and lets only leaders interact with the cache. Non-leaders
// receive results via warp shuffle — no memory access needed.
//
// This is the optimization that gives BaM 430 GB/s hot cache bandwidth
// (11.2x over ActivePointers' serialized approach).
// ============================================================================

// Cache line configuration
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 4096
#endif

#define ELEMS_PER_CL_INT (CACHE_LINE_SIZE / sizeof(int))
#define ELEMS_PER_CL_FLOAT (CACHE_LINE_SIZE / sizeof(float))

// ============================================================================
// Coalesced read from a flat buffer (used for testing coalescer in isolation)
// The buffer represents data organized in cache-line-sized blocks.
//
// This version does NOT involve the BaM cache — it's a microbenchmark
// to measure the raw benefit of warp coalescing on buffer access.
// ============================================================================

// Naive: every thread independently reads its element
// No grouping, no leader election — all 32 threads do full work
__device__ __forceinline__
int naive_read(const int* data, int index) {
    return data[index];
}

// Coalesced: group threads by cache line, leader reads, broadcasts
// This tests the coalescing mechanism itself without cache overhead
__device__ __forceinline__
int coalesced_read(const int* data, int index, int* num_probes) {
    int cl_id  = index / ELEMS_PER_CL_INT;
    int offset = index % ELEMS_PER_CL_INT;
    int lane   = threadIdx.x % 32;

    // Step 1: Find all threads in this warp needing the same cache line
    unsigned active = __activemask();
    unsigned match  = __match_any_sync(active, cl_id);

    // Step 2: Elect the lowest-lane thread as leader for each group
    int leader    = __ffs(match) - 1;
    bool is_leader = (lane == leader);

    // Step 3: Only leaders "probe" (in real BaM, this is the cache lookup)
    int val = 0;
    if (is_leader) {
        // Count how many actual probes happen (should equal unique cache lines)
        if (num_probes != nullptr) {
            atomicAdd(num_probes, 1);
        }
        val = data[cl_id * ELEMS_PER_CL_INT + offset];
    }

    // Step 4: Leader broadcasts the cache line base address (slot) to group
    // In real BaM, this would be the cache slot index
    // Here we broadcast cl_id so each thread can compute its own read
    int shared_cl_id = __shfl_sync(match, cl_id, leader);

    // Step 5: Non-leaders read their own element using the shared base
    if (!is_leader) {
        val = data[shared_cl_id * ELEMS_PER_CL_INT + offset];
    }

    return val;
}

// ============================================================================
// Version that simulates a cache slot lookup pattern
// Leader looks up a "slot" from a metadata array, broadcasts slot to group,
// all threads read from the slot's data region.
// This is closer to what real BaM cache access looks like.
// ============================================================================

struct FakeSlotMeta {
    int tag;     // which cl_id is stored here
    int valid;   // 1 if data is present
};

__device__ __forceinline__
int coalesced_cache_read(const int* cache_data, const FakeSlotMeta* meta,
                         int num_slots, int index, int* num_probes) {
    int cl_id  = index / ELEMS_PER_CL_INT;
    int offset = index % ELEMS_PER_CL_INT;
    int lane   = threadIdx.x % 32;

    // Step 1: Group threads by cache line
    unsigned active = __activemask();
    unsigned match  = __match_any_sync(active, cl_id);

    // Step 2: Elect leader
    int leader     = __ffs(match) - 1;
    bool is_leader = (lane == leader);

    // Step 3: Leader probes cache metadata
    int slot = -1;
    if (is_leader) {
        if (num_probes != nullptr) atomicAdd(num_probes, 1);

        // Direct-mapped lookup (same as real BaM)
        int candidate = cl_id % num_slots;
        if (meta[candidate].tag == cl_id && meta[candidate].valid) {
            slot = candidate;
        }
        // In real BaM, a miss here would trigger the I/O path
        // For this test, we assume everything is pre-loaded (all hits)
    }

    // Step 4: Broadcast slot to all threads in the group
    slot = __shfl_sync(match, slot, leader);

    // Step 5: All threads read their own element from the slot's data
    int val = 0;
    if (slot >= 0) {
        val = cache_data[slot * ELEMS_PER_CL_INT + offset];
    }

    return val;
}

// ============================================================================
// ActivePointers-style serialized coalescing (the slow baseline)
// Instead of __match_any_sync, loops through lanes one at a time
// ============================================================================

__device__ __forceinline__
int serialized_coalesced_read(const int* cache_data, const FakeSlotMeta* meta,
                              int num_slots, int index, int* num_probes) {
    int cl_id  = index / ELEMS_PER_CL_INT;
    int offset = index % ELEMS_PER_CL_INT;
    int lane   = threadIdx.x % 32;

    int my_slot = -1;
    unsigned served = 0;  // bitmask of threads already served

    // Loop through all 32 lanes, one at a time
    for (int source = 0; source < 32; source++) {
        // Broadcast source lane's cl_id to entire warp
        int broadcast_cl = __shfl_sync(0xFFFFFFFF, cl_id, source);

        if (source == lane && !(served & (1u << lane))) {
            // My turn to probe
            if (num_probes != nullptr) atomicAdd(num_probes, 1);

            int candidate = broadcast_cl % num_slots;
            if (meta[candidate].tag == broadcast_cl && meta[candidate].valid) {
                my_slot = candidate;
            }
        }

        // If my cl_id matches the one being probed, grab the result
        if (cl_id == broadcast_cl && !(served & (1u << lane))) {
            my_slot = __shfl_sync(0xFFFFFFFF, my_slot, source);
            served |= (1u << lane);
        }
    }

    int val = 0;
    if (my_slot >= 0) {
        val = cache_data[my_slot * ELEMS_PER_CL_INT + offset];
    }
    return val;
}

#endif // BAM_COALESCER_CUH
