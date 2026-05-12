#ifndef BAM_THREE_TIER_TLB_CUH
#define BAM_THREE_TIER_TLB_CUH

#include "cache.cuh"

#define T2_NUM_ENTRIES 32
#define T2_LOCK_BIT 0x80000000U
#define T2_CNT_MASK 0x3fffffffU

struct T2Entry
{
    uint32_t state; // atomic: bit31=LOCK, bits29-0=ref count
    uint32_t page_id;
    uint32_t cache_slot;
};

struct Tier2TLB
{
    T2Entry entries[T2_NUM_ENTRIES];
};

struct Tier1Local
{
    uint32_t page_id;
    uint32_t t2_entry_idx;
    int start_elem;
    int end_elem;
    char *addr;
    bool holding;
};

__device__ __forceinline__ void t2_init(Tier2TLB *t2)
{
    for (int i = threadIdx.x; i < T2_NUM_ENTRIES; i += blockDim.x)
    {
        t2->entries[i].state = 0;
        t2->entries[i].page_id = 0xFFFFFFFF;
        t2->entries[i].cache_slot = 0xFFFFFFFF;
    }
    __syncthreads();
}

__device__ __forceinline__ void t2_fini(Tier2TLB *t2, BamCache *cache)
{
    __syncthreads();
    // Now that everyone in the block is doe, it is safe to unpin the physical Tier 3 VRAM slots.
    for (int i = threadIdx.x; i < T2_NUM_ENTRIES; i += blockDim.x)
    {
        if (t2->entries[i].page_id != 0xFFFFFFFF)
        {
            __threadfence();
            cache_release_page(cache, t2->entries[i].page_id, 1);
        }
    }
}

__device__ __forceinline__ char *t2_acquire(Tier2TLB *t2, BamCache *cache, uint32_t page_id,
                                            uint32_t *out_entry_idx, const char *backing)
{
    uint32_t lane = threadIdx.x % 32;

    // Warp coalescing
    unsigned mask = __activemask();
    unsigned eq_mask = __match_any_sync(mask, page_id);
    int master = __ffs(eq_mask) - 1;
    uint32_t count = __popc(eq_mask);

    uint32_t ent_idx = page_id % T2_NUM_ENTRIES;
    T2Entry *entry = &t2->entries[ent_idx];

    uint64_t base_master = 0;

    if ((int)lane == master)
    {
        uint64_t c = 0;
        uint32_t st;

        do
        {

            do
            {
                st = atomicOr((unsigned int *)&entry->state, T2_LOCK_BIT);
                if ((st & T2_LOCK_BIT) == 0)
                    break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700)
                __nanosleep(100);
#endif
            } while (true);

            // st = state BEFORE lock was set. Lower bits = ref count.

            if (entry->page_id == page_id && entry->page_id != 0xFFFFFFFF)
            {
                // CASE 1: TLB HIT
                // st += count;
                base_master = (uint64_t)(cache->d_data +
                                         (uint64_t)entry->cache_slot * CACHE_LINE_SIZE);
                // Unlock: store st (no LOCK bit since st was pre-lock)
                // atomicExch((unsigned int*)&entry->state, st);
                atomicAdd((unsigned int *)&entry->state, count - T2_LOCK_BIT);
                break;
            }
            else if (entry->page_id == 0xFFFFFFFF ||
                     (st & T2_CNT_MASK) == 0)
            {
                // CASE 2: TLB MISS
                if (entry->page_id != 0xFFFFFFFF)
                {
                    __threadfence();
                    cache_release_page(cache, entry->page_id, 1);
                }

                uint32_t new_slot = cache_acquire_page(
                    cache, page_id, 1, backing);

                entry->page_id = page_id;
                entry->cache_slot = new_slot;
                // st += count;
                base_master = (uint64_t)(cache->d_data +
                                         (uint64_t)new_slot * CACHE_LINE_SIZE);
                __threadfence();
                // atomicExch((unsigned int*)&entry->state, st);
                atomicAdd((unsigned int *)&entry->state, count - T2_LOCK_BIT);
                break;
            }
            else
            {
                // CASE 3: CONFLICT
                if (++c % 100000 == 0)
                {
                    printf("TLB conflict: c=%llu tid=%u wanted=%u got=%u st=0x%x\n",
                           (unsigned long long)c, threadIdx.x,
                           page_id, entry->page_id, st);
                }
                // atomicExch((unsigned int*)&entry->state, st);
                atomicSub((unsigned int *)&entry->state, T2_LOCK_BIT);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700)
                __nanosleep(100);
#endif
            }

        } while (true);
    }

    // Broadcast
    base_master = __shfl_sync(eq_mask, base_master, master);
    ent_idx = __shfl_sync(eq_mask, ent_idx, master);
    *out_entry_idx = ent_idx;

    return (char *)base_master;
}

__device__ __forceinline__ void t2_release(Tier2TLB *t2, uint32_t entry_idx, uint32_t page_id)
{
    atomicSub((unsigned int *)&t2->entries[entry_idx].state, 1);
}

__device__ __forceinline__ void t1_init(Tier1Local *t1)
{
    t1->page_id = 0xFFFFFFFF;
    t1->t2_entry_idx = 0xFFFFFFFF;
    t1->start_elem = 0;
    t1->end_elem = 0;
    t1->addr = nullptr;
    t1->holding = false;
}

__device__ __forceinline__ void t1_fini(Tier1Local *t1, Tier2TLB *t2)
{
    if (t1->holding)
    {
        t2_release(t2, t1->t2_entry_idx, t1->page_id);
        t1->holding = false;
        t1->addr = nullptr;
        t1->start_elem = 0;
        t1->end_elem = 0;
        t1->t2_entry_idx = 0xFFFFFFFF;
        t1->page_id = 0xFFFFFFFF;
    }
}

__device__ __forceinline__ int t1_read(Tier1Local *t1, Tier2TLB *t2, BamCache *cache,
                                       int index, const char *backing)
{

    // Fast path
    if (index >= t1->start_elem && index < t1->end_elem)
    {
        return ((int *)t1->addr)[index - t1->start_elem];
    }

    // Slow path

    t1_fini(t1, t2);

    uint32_t new_page_id = index / CL_ELEMS_INT;
    uint32_t new_entry_idx;

    char *page_addr = t2_acquire(t2, cache, new_page_id,
                                 &new_entry_idx, backing);

    t1->page_id = new_page_id;
    t1->t2_entry_idx = new_entry_idx;
    t1->start_elem = new_page_id * CL_ELEMS_INT;
    t1->end_elem = t1->start_elem + CL_ELEMS_INT;
    t1->addr = page_addr;
    t1->holding = true;

    return ((int *)t1->addr)[index - t1->start_elem];
}

#endif // BAM_THREE_TIER_TLB_CUH
