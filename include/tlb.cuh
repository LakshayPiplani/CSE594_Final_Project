#ifndef BAM_TLB_CUH
#define BAM_TLB_CUH

#include "cache.cuh"

// ============================================================================
// BaM Thread-Local Buffer (TLB)
//
// This is a per-thread optimization layer that sits ON TOP of the VRAM
// cache (cache.cuh). The VRAM cache is NOT modified at all.
//
// Architecture:
//
//   BFS kernel
//       |
//       v
//   tlb_read()          <-- you implement this
//       |
//       |-- TLB hit?  --> direct VRAM read, zero atomics, return
//       |
//       |-- TLB miss? --> call cache_release_page() for old page
//                         call cache_acquire_page() for new page
//                         update TLB fields
//                         direct VRAM read, return
//
// Why this helps:
//   BFS iterates: for (e = start; e < end; e++) { neighbor = edges[e]; }
//   Each page holds CL_ELEMS_INT (1024) consecutive ints.
//   A node with degree 200 touches ~1 page. Without TLB, that's 200
//   acquire/release pairs = 400 atomic ops. With TLB, it's 1 acquire,
//   199 direct VRAM reads, 1 release = 2 atomic ops.
//
// Important constraints:
//   - While the TLB holds a page, the ref count stays at 1 in the VRAM
//     cache, preventing eviction of that page.
//   - You MUST call tlb_release() when the thread is done, otherwise the
//     ref count stays up permanently and find_slot will livelock.
//   - Cache must have enough slots for all simultaneously-pinned pages.
//
// Usage pattern in a BFS kernel:
//
//   __global__ void bfs_kernel(...) {
//       int tid = blockIdx.x * blockDim.x + threadIdx.x;
//       if (tid >= num_nodes) return;
//       if (d_distances[tid] != level) return;
//
//       BamTLB tlb;
//       tlb_init(&tlb);
//
//       int start = d_offsets[tid];
//       int end   = d_offsets[tid + 1];
//       for (int e = start; e < end; e++) {
//           int neighbor = tlb_read(&cache, e, backing, &tlb);
//           if (d_distances[neighbor] == INF_DIST) {
//               d_distances[neighbor] = level + 1;
//               atomicAdd(d_frontier_size, 1);
//           }
//       }
//
//       tlb_release(&cache, &tlb);   // <-- MUST call this
//   }
//
// ============================================================================

struct BamTLB {
    uint32_t page_id;      // which logical page is currently held
    uint32_t slot;          // which VRAM cache slot holds that page's data
    int      start_elem;   // first element index in this page
    int      end_elem;     // first element index of the NEXT page
    bool     holding;       // true if a page is currently held (ref count > 0)
};

// ============================================================================
// Function 1: tlb_init
//
// Set all fields to indicate "not holding any page".
//
// After init, the first call to tlb_read will always be a TLB miss,
// which triggers a cache_acquire_page.
// ============================================================================
__device__ __forceinline__
void tlb_init(BamTLB* tlb) {
    // YOUR CODE HERE
    //
    // Set holding to false.
    // Set slot to an invalid sentinel (e.g., 0xFFFFFFFF).
    // Set start_elem and end_elem so that any index will be outside
    // the range, guaranteeing the first read is a TLB miss.
    tlb->holding=false;
    tlb->slot=0xFFFFFFFF;
    tlb->start_elem=0;
    tlb->end_elem=0;
}

// ============================================================================
// Function 2: tlb_read
//
// Read the integer at position 'index' from the edge array.
//
// Fast path (TLB hit):
//   If index >= start_elem AND index < end_elem:
//     - Compute offset = index - start_elem
//     - Read directly: cache->d_data[slot * CACHE_LINE_SIZE + offset * 4]
//     - Return the value. No atomics. No cache calls.
//
// Slow path (TLB miss):
//   If index is outside [start_elem, end_elem):
//     1. If currently holding a page (holding == true):
//        a. __threadfence()  -- ensures our previous reads from this slot
//           are committed before we drop the ref count
//        b. cache_release_page(cache, old page_id, 1)
//
//     2. Compute new page_id = index / CL_ELEMS_INT
//
//     3. Acquire the new page:
//        new_slot = cache_acquire_page(cache, new page_id, 1, backing)
//
//     4. Update TLB fields:
//        page_id    = new page_id
//        slot       = new_slot
//        start_elem = new page_id * CL_ELEMS_INT
//        end_elem   = start_elem + CL_ELEMS_INT
//        holding    = true
//
//     5. Compute offset = index - start_elem
//        Read from cache->d_data[slot * CACHE_LINE_SIZE + offset * 4]
//        Return the value.
//
// Constants you will need:
//   CL_ELEMS_INT   -- number of ints per cache line (1024 for 4KB pages)
//   CACHE_LINE_SIZE -- bytes per cache line (4096)
//
// Functions you will call:
//   cache_release_page(cache, page_id, count)  -- from cache.cuh
//   cache_acquire_page(cache, page_id, count, backing) -- from cache.cuh
// ============================================================================
__device__ __forceinline__
int tlb_read(BamCache* cache, int index, const char* backing, BamTLB* tlb) {
    if (index >= tlb->start_elem && index < tlb->end_elem) {
        // TLB hit: read directly from the VRAM cache slot
        int offset = index - tlb->start_elem;
        int* int_cache = (int*)cache->d_data;
        int value = int_cache[tlb->slot * CL_ELEMS_INT + offset]; 
        return value;
    }
    // index not present in TLB, need to acquire the correct page
    if (tlb->holding) {
        __threadfence();
        cache_release_page(cache, tlb->page_id, 1);
    }
    uint32_t new_page_id = index / CL_ELEMS_INT;
    uint32_t new_slot_id = cache_acquire_page(cache, new_page_id, 1, backing);
    tlb->holding=true;
    tlb->page_id = new_page_id;
    tlb->slot=new_slot_id;
    tlb->start_elem=new_page_id*CL_ELEMS_INT;
    tlb->end_elem=new_page_id*CL_ELEMS_INT + CL_ELEMS_INT;
    
    int offset = index - tlb->start_elem;
    int* int_cache = (int*)cache->d_data;
    int value = int_cache[tlb->slot * CL_ELEMS_INT + offset];
    return value;
}

// ============================================================================
// Function 3: tlb_release
//
// Release the currently held page, if any.
//
// This MUST be called when the thread finishes its work (after the
// neighbor-list loop in BFS). If you forget, the ref count stays
// incremented permanently, preventing eviction and causing livelock.
//
// Logic:
//   If holding == true:
//     1. __threadfence()  -- commit any reads from this slot
//     2. cache_release_page(cache, page_id, 1)
//     3. holding = false
// ============================================================================
__device__ __forceinline__
void tlb_release(BamCache* cache, BamTLB* tlb) {
    if (tlb->holding) {

        __threadfence();
        cache_release_page(cache, tlb->page_id, 1);
        tlb->holding=false;

    }
}

#endif // BAM_TLB_CUH
