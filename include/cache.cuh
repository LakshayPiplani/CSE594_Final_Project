#ifndef BAM_CACHE_CUH
#define BAM_CACHE_CUH

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

/*
 Follows the real BaM page_cache.h design:
 - Per-logical-page state array
 - State + ref count packed into single atomic uint32_t
 - Clock algorithm for cache slot allocation
 - BUSY flag prevents concurrent load/evict races


 State word layout (32 bits):
   Bit 31:    VALID  (data is in cache and ready)
   Bit 30:    BUSY   (someone is loading or evicting)
   Bit 29:    DIRTY  (data has been written, not flushed)
   Bits 28-0: Reference count (up to ~500M concurrent readers)


*/
// State bits
#define ST_VALID 0x80000000U
#define ST_BUSY 0x40000000U
#define ST_DIRTY 0x20000000U
#define ST_CNT_MASK 0x1fffffffU
#define ST_FLAGS_MASK 0xe0000000U

// Combined transitions
#define ST_DISABLE_BUSY_ENABLE_VALID 0xc0000000U // XOR: clears BUSY, sets VALID
#define ST_DISABLE_BUSY_MASK 0xbfffffffU         // AND: clears only BUSY

// State classification (upper 2 bits after shifting)
#define ST_NV_NB 0x00U // not valid, not busy
#define ST_NV_B 0x01U  // not valid, busy: someone else loading, wait
#define ST_V_NB 0x02U  // valid, not busy: HIT
#define ST_V_B 0x03U   // valid, busy: being evicted, wait

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 4096
#endif
#define CL_ELEMS_INT (CACHE_LINE_SIZE / sizeof(int))

// Per-logical-page state (stored in an array indexed by page ID)

struct PageState
{
    uint32_t state;  // atomic: packed VALID|BUSY|DIRTY|refcount
    uint32_t offset; // which cache slot holds this page's data
};

// Per-cache-slot metadata (for eviction tracking)

struct CacheSlotMeta
{
    uint32_t page_take_lock;   // atomic: FREE=2, UNLOCKED=0, LOCKED=1
    uint64_t page_translation; // which logical page is stored here
};

#define SLOT_FREE 2U
#define SLOT_UNLOCKED 0U
#define SLOT_LOCKED 1U

// Cache structure

struct BamCache
{
    // Cache data buffer in VRAM
    char *d_data; // [num_slots * CACHE_LINE_SIZE]

    // Per-logical-page state (in VRAM)
    // Indexed by page_id
    PageState *d_pages;
    uint32_t num_logical_pages;

    // Per-cache-slot metadata (in VRAM)
    CacheSlotMeta *d_slots;
    uint32_t num_slots;

    // Clock eviction counter
    uint32_t *d_clock;

    // Stats
    unsigned long long *d_hits;
    unsigned long long *d_misses;
    unsigned long long *d_coalesced_waits; // threads that waited on BUSY
};

inline void cache_init(BamCache *cache, size_t cache_size_bytes,
                       uint32_t total_logical_pages)
{
    cache->num_slots = cache_size_bytes / CACHE_LINE_SIZE;
    cache->num_logical_pages = total_logical_pages;

    // Data buffer
    cudaMalloc(&cache->d_data, (size_t)cache->num_slots * CACHE_LINE_SIZE);

    // Per-logical-page state array
    PageState *h_pages = (PageState *)malloc(
        total_logical_pages * sizeof(PageState));
    for (uint32_t i = 0; i < total_logical_pages; i++)
    {
        h_pages[i].state = 0; // INVALID: no VALID, no BUSY, count=0
        h_pages[i].offset = 0;
    }
    cudaMalloc(&cache->d_pages, total_logical_pages * sizeof(PageState));
    cudaMemcpy(cache->d_pages, h_pages,
               total_logical_pages * sizeof(PageState),
               cudaMemcpyHostToDevice);
    free(h_pages);

    // Per-cache-slot metadata
    CacheSlotMeta *h_slots = (CacheSlotMeta *)malloc(
        cache->num_slots * sizeof(CacheSlotMeta));
    for (uint32_t i = 0; i < cache->num_slots; i++)
    {
        h_slots[i].page_take_lock = SLOT_FREE;
        h_slots[i].page_translation = 0;
    }
    cudaMalloc(&cache->d_slots, cache->num_slots * sizeof(CacheSlotMeta));
    cudaMemcpy(cache->d_slots, h_slots,
               cache->num_slots * sizeof(CacheSlotMeta),
               cudaMemcpyHostToDevice);
    free(h_slots);

    // Clock counter
    cudaMalloc(&cache->d_clock, sizeof(uint32_t));
    cudaMemset(cache->d_clock, 0, sizeof(uint32_t));

    // Stats
    cudaMalloc(&cache->d_hits, sizeof(unsigned long long));
    cudaMalloc(&cache->d_misses, sizeof(unsigned long long));
    cudaMalloc(&cache->d_coalesced_waits, sizeof(unsigned long long));
    cudaMemset(cache->d_hits, 0, sizeof(unsigned long long));
    cudaMemset(cache->d_misses, 0, sizeof(unsigned long long));
    cudaMemset(cache->d_coalesced_waits, 0, sizeof(unsigned long long));
}

inline void cache_destroy(BamCache *cache)
{
    cudaFree(cache->d_data);
    cudaFree(cache->d_pages);
    cudaFree(cache->d_slots);
    cudaFree(cache->d_clock);
    cudaFree(cache->d_hits);
    cudaFree(cache->d_misses);
    cudaFree(cache->d_coalesced_waits);
}

inline void cache_reset(BamCache *cache)
{
    PageState *h_pages = (PageState *)malloc(
        cache->num_logical_pages * sizeof(PageState));
    for (uint32_t i = 0; i < cache->num_logical_pages; i++)
    {
        h_pages[i].state = 0;
        h_pages[i].offset = 0;
    }
    cudaMemcpy(cache->d_pages, h_pages,
               cache->num_logical_pages * sizeof(PageState),
               cudaMemcpyHostToDevice);
    free(h_pages);

    CacheSlotMeta *h_slots = (CacheSlotMeta *)malloc(
        cache->num_slots * sizeof(CacheSlotMeta));
    for (uint32_t i = 0; i < cache->num_slots; i++)
    {
        h_slots[i].page_take_lock = SLOT_FREE;
        h_slots[i].page_translation = 0;
    }
    cudaMemcpy(cache->d_slots, h_slots,
               cache->num_slots * sizeof(CacheSlotMeta),
               cudaMemcpyHostToDevice);
    free(h_slots);

    cudaMemset(cache->d_clock, 0, sizeof(uint32_t));
    cudaMemset(cache->d_hits, 0, sizeof(unsigned long long));
    cudaMemset(cache->d_misses, 0, sizeof(unsigned long long));
    cudaMemset(cache->d_coalesced_waits, 0, sizeof(unsigned long long));
}

struct CacheStats
{
    unsigned long long hits;
    unsigned long long misses;
    unsigned long long coalesced_waits;
};

inline CacheStats cache_get_stats(BamCache *cache)
{
    CacheStats s;
    cudaMemcpy(&s.hits, cache->d_hits,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s.misses, cache->d_misses,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s.coalesced_waits, cache->d_coalesced_waits,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return s;
}

// Device-side: find_slot (clock eviction)

__device__ __forceinline__
    uint32_t
    cache_find_slot(BamCache *cache, uint32_t page_id)
{
    unsigned int ns = 8;

    while (true)
    {
        uint32_t slot = atomicAdd(cache->d_clock, 1) % cache->num_slots;

        uint32_t v = atomicAdd((unsigned int *)&cache->d_slots[slot].page_take_lock, 0);

        // Slot never used, take it directly
        if (v == SLOT_FREE)
        {
            uint32_t old = atomicCAS(&cache->d_slots[slot].page_take_lock,
                                     SLOT_FREE, SLOT_LOCKED);
            if (old == SLOT_FREE)
            {
                cache->d_slots[slot].page_translation = page_id;
                __threadfence();
                atomicExch(&cache->d_slots[slot].page_take_lock, SLOT_UNLOCKED);
                return slot;
            }
        }
        // Slot previously used, try to evict
        else if (v == SLOT_UNLOCKED)
        {
            uint32_t old = atomicCAS(&cache->d_slots[slot].page_take_lock,
                                     SLOT_UNLOCKED, SLOT_LOCKED);
            if (old == SLOT_UNLOCKED)
            {
                uint32_t prev_page = (uint32_t)cache->d_slots[slot].page_translation;

                // Check if previous occupant can be evicted
                uint32_t prev_state = atomicAdd(
                    (unsigned int *)&cache->d_pages[prev_page].state, 0);
                uint32_t cnt = prev_state & ST_CNT_MASK;
                uint32_t busy = prev_state & ST_BUSY;

                if (cnt == 0 && busy == 0)
                {
                    // Try to set BUSY on the old page
                    uint32_t old_state = atomicOr(
                        (unsigned int *)&cache->d_pages[prev_page].state,
                        ST_BUSY);

                    if ((old_state & ST_BUSY) == 0 &&
                        (old_state & ST_CNT_MASK) == 0)
                    {
                        // Successfully locked old page for eviction
                        // Clear flags (VALID, BUSY, DIRTY) but preserve any ref count that another thread might have added between our check and now.

                        uint32_t cleared = atomicAnd(
                            (unsigned int *)&cache->d_pages[prev_page].state,
                            ST_CNT_MASK);

                        // Verify count is still zero after clearing flags
                        if ((cleared & ST_CNT_MASK) != 0)
                        {
                            // Someone snuck in, restore VALID, remove BUSY
                            atomicOr(
                                (unsigned int *)&cache->d_pages[prev_page].state,
                                ST_VALID);
                            atomicAnd(
                                (unsigned int *)&cache->d_pages[prev_page].state,
                                ST_DISABLE_BUSY_MASK);
                            // Failed to evict, release slot and try another
                            atomicExch(&cache->d_slots[slot].page_take_lock,
                                       SLOT_UNLOCKED);
                            goto next_slot;
                        }

                        // Assign slot to new page
                        cache->d_slots[slot].page_translation = page_id;
                        __threadfence();
                        atomicExch(&cache->d_slots[slot].page_take_lock,
                                   SLOT_UNLOCKED);
                        return slot;
                    }
                    else
                    {
                        // Someone else grabbed it
                        atomicAnd(
                            (unsigned int *)&cache->d_pages[prev_page].state,
                            ST_DISABLE_BUSY_MASK);
                    }
                }

                // Failed to evict, unlock slot and try another
                atomicExch(&cache->d_slots[slot].page_take_lock, SLOT_UNLOCKED);
            }
        }
        // Slot is LOCKED by another thread
    next_slot:

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700)
        __nanosleep(ns);
        if (ns < 256)
            ns *= 2;
#endif
        ;
    }
}

__device__ __forceinline__ void cache_fill(BamCache *cache, uint32_t slot, uint32_t page_id,
                                           const char *backing_store)
{
    const int4 *src = (const int4 *)(backing_store +
                                     (uint64_t)page_id * CACHE_LINE_SIZE);
    int4 *dst = (int4 *)(cache->d_data + (uint64_t)slot * CACHE_LINE_SIZE);
    int num_int4_elements = CL_ELEMS_INT / 4;
    for (int i = 0; i < num_int4_elements; i++)
    {
        dst[i] = src[i];
    }
    __threadfence();
}

__device__ __forceinline__
    uint32_t
    cache_acquire_page(BamCache *cache, uint32_t page_id,
                       uint32_t count, const char *backing_store)
{
    unsigned int ns = 8;

    // Atomically increment ref countt, read previous state
    uint32_t read_state = atomicAdd(
        (unsigned int *)&cache->d_pages[page_id].state, count);

    bool fail = true;
    do
    {
        uint32_t st = (read_state >> 30) & 0x03;

        switch (st)
        {
        // Not valid, not busy: MISS
        case ST_NV_NB:
        {
            uint32_t old = atomicOr(
                (unsigned int *)&cache->d_pages[page_id].state, ST_BUSY);

            if ((old & ST_BUSY) == 0)
            {
                uint32_t old_st = (old >> 30) & 0x03;
                if (old_st == ST_NV_NB)
                {
                    uint32_t slot = cache_find_slot(cache, page_id);

                    atomicAdd(cache->d_misses, 1ULL);
                    cache_fill(cache, slot, page_id, backing_store);

                    cache->d_pages[page_id].offset = slot;

                    __threadfence();
                    atomicXor(
                        (unsigned int *)&cache->d_pages[page_id].state,
                        ST_DISABLE_BUSY_ENABLE_VALID);

                    return slot;
                }
                else
                {
                    // State changed: clear BUSY and retry
                    atomicAnd(
                        (unsigned int *)&cache->d_pages[page_id].state,
                        ST_DISABLE_BUSY_MASK);
                }
            }
            break;
        }

        // Valid, not busy: HIT
        case ST_V_NB:
        {
            atomicAdd(cache->d_hits, 1ULL);
            return cache->d_pages[page_id].offset;
        }

        // Busy
        case ST_NV_B:
        case ST_V_B:
        default:
            atomicAdd(cache->d_coalesced_waits, 1ULL);
            break;
        }

        // Backoff and re-read state (load only, dont add count again)
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700)
        __nanosleep(ns);
        if (ns < 256)
            ns *= 2;
#endif

        read_state = atomicAdd(
            (unsigned int *)&cache->d_pages[page_id].state, 0);

    } while (true);
}

__device__ __forceinline__ void cache_release_page(BamCache *cache, uint32_t page_id, uint32_t count)
{
    atomicSub((unsigned int *)&cache->d_pages[page_id].state, count);
}

__device__ __forceinline__ int cache_read_no_coalesce(BamCache *cache, int index,
                                                      const char *backing_store)
{
    uint32_t page_id = index / CL_ELEMS_INT;
    uint32_t offset = index % CL_ELEMS_INT;

    uint32_t slot = cache_acquire_page(cache, page_id, 1, backing_store);

    int val = ((int *)(cache->d_data + (uint64_t)slot * CACHE_LINE_SIZE))[offset];

    __threadfence();

    cache_release_page(cache, page_id, 1);

    return val;
}

__device__ __forceinline__ int cache_read_coalesced(BamCache *cache, int index,
                                                    const char *backing_store)
{
    uint32_t page_id = index / CL_ELEMS_INT;
    uint32_t offset = index % CL_ELEMS_INT;
    uint32_t lane = threadIdx.x % 32;

    // Step 1: Warp coalescing — group threads by page_id
    unsigned active = __activemask();
    unsigned eq_mask = __match_any_sync(active, page_id);
    int master = __ffs(eq_mask) - 1;
    uint32_t count = __popc(eq_mask);

    // Step 2: Only master acquires the page (with count for whole group)
    uint32_t slot;
    if ((int)lane == master)
    {
        slot = cache_acquire_page(cache, page_id, count, backing_store);
    }

    // Step 3: Broadcast slot to all threads in group
    slot = __shfl_sync(eq_mask, slot, master);

    // Step 4: Every thread reads its own element
    int val = ((int *)(cache->d_data + (uint64_t)slot * CACHE_LINE_SIZE))[offset];

    // Step 5: Ensure all reads complete before releasing ref count
    __threadfence();
    __syncwarp(eq_mask);

    // Step 6: Master releases ref count for entire group
    if ((int)lane == master)
    {
        cache_release_page(cache, page_id, count);
    }

    // Step 7: Sync all active threads
    __syncwarp(active);

    return val;
}

#endif // BAM_CACHE_CUH