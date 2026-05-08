#ifndef BAM_PTR_CUH
#define BAM_PTR_CUH

#include "cache.cuh"

// ============================================================================
// Two-Tier BaM Pointer — mirrors real BaM's bam_ptr (page_cache.h)
//
// Tier 1: Per-thread register state (BamPtr) — same as Tier1Local
// Tier 3: BamCache (cache.cuh) — accessed directly on T1 miss
//
// No Tier 2 TLB. On a T1 miss, warp coalescing happens here before
// dropping into cache_acquire_page, so the atomic pressure on T3 is
// the same as the three-tier path for threads landing on the same page.
//
// Refcount accounting:
//   - master acquires with count = __popc(eq_mask) (one unit per thread)
//   - each thread stores holding=true and releases exactly 1 in bp_fini
// ============================================================================

struct BamPtr {
    uint32_t page_id;
    int      start_elem;
    int      end_elem;
    char*    addr;
    bool     holding;
};

// ============================================================================
// bp_init — matches bam_ptr_tlb::init
// ============================================================================

__device__ __forceinline__
void bp_init(BamPtr* bp) {
    bp->page_id    = 0xFFFFFFFF;
    bp->start_elem = 0;
    bp->end_elem   = 0;
    bp->addr       = nullptr;
    bp->holding    = false;
}

// ============================================================================
// bp_fini — releases directly to T3, no T2 intermediary
// Each thread releases exactly 1 refcount unit it acquired in bp_read.
// ============================================================================

__device__ __forceinline__
void bp_fini(BamPtr* bp, BamCache* cache) {
    if (bp->holding) {
        __threadfence();
        cache_release_page(cache, bp->page_id, 1);
        bp->holding    = false;
        bp->addr       = nullptr;
        bp->page_id    = 0xFFFFFFFF;
        bp->start_elem = 0;
        bp->end_elem   = 0;
    }
}

// ============================================================================
// bp_read — T1 fast path + coalesced T3 miss path
//
// Fast path (T1 hit): direct pointer deref, zero atomics.
//
// Slow path (T1 miss):
//   1. Drop current pin via bp_fini
//   2. Warp-coalesce threads landing on the same new page
//   3. Master calls cache_acquire_page once with count = group size
//   4. Slot broadcast to all threads in group via __shfl_sync
//   5. Every thread updates its own BamPtr and marks holding=true
//      so each independently releases 1 unit in bp_fini
// ============================================================================

__device__ __forceinline__
int bp_read(BamPtr* bp, BamCache* cache, int index, const char* backing) {

    // Fast path
    if (index >= bp->start_elem && index < bp->end_elem)
        return ((int*)bp->addr)[index - bp->start_elem];

    // Slow path
    bp_fini(bp, cache);

    uint32_t new_page_id = (uint32_t)index / CL_ELEMS_INT;
    uint32_t lane        = threadIdx.x % 32;

    unsigned active  = __activemask();
    unsigned eq_mask = __match_any_sync(active, new_page_id);
    int      master  = __ffs(eq_mask) - 1;
    uint32_t count   = __popc(eq_mask);

    uint32_t slot;
    if ((int)lane == master)
        slot = cache_acquire_page(cache, new_page_id, count, backing);
    slot = __shfl_sync(eq_mask, slot, master);

    bp->page_id    = new_page_id;
    bp->start_elem = (int)(new_page_id * CL_ELEMS_INT);
    bp->end_elem   = bp->start_elem + CL_ELEMS_INT;
    bp->addr       = cache->d_data + (uint64_t)slot * CACHE_LINE_SIZE;
    bp->holding    = true;

    return ((int*)bp->addr)[index - bp->start_elem];
}

#endif // BAM_PTR_CUH
