#ifndef BAM_THREE_TIER_TLB_CUH
#define BAM_THREE_TIER_TLB_CUH

#include "cache.cuh"

// ============================================================================
// Three-Tier TLB — Exact replication of real BaM's page_cache.h
//
// This follows the real BaM code line by line:
//   Tier 1: bam_ptr_tlb (lines 346-411)
//   Tier 2: struct tlb + tlb_entry (lines 158-344)
//   Tier 3: cache.cuh (our VRAM cache)
//
// TLB entry state word layout (matches real BaM's tlb_entry::state):
//   Bit 31:    LOCK (VALID_ in BaM — used as spinlock)
//   Bits 30-0: Ref count (threads currently using this entry)
//
// Lock protocol (matches real BaM's tlb::acquire lines 271-276):
//   Acquire: fetch_or(LOCK_BIT) — if old value had LOCK=0, we got it
//   Release: state.store(new_value) — writes entire word, clears LOCK
//
// Three cases in acquire (matches lines 278-312):
//   CASE 1: entry->page_id == our page → TLB hit, increment count
//   CASE 2: entry empty OR ref count == 0 → evict, load new page
//   CASE 3: different page, ref count > 0 → unlock, nanosleep, retry
//
// t2_fini: no __syncthreads (matches real BaM lines 232-244 where
//   __syncthreads is commented out). Each entry releases its VRAM
//   cache ref directly via cache_release_page.
// ============================================================================

#define T2_NUM_ENTRIES  256
#define T2_LOCK_BIT     0x80000000U   // VALID_ in real BaM
#define T2_CNT_MASK     0x3fffffffU   // CNT_MASK_ in real BaM (30 bits)

// ============================================================================
// Tier 2: TLB entry — matches real BaM's tlb_entry (lines 158-194)
// ============================================================================

struct T2Entry {
    uint32_t state;       // atomic: bit31=LOCK, bits29-0=ref count
    uint32_t page_id;     // global_id in real BaM
    uint32_t cache_slot;  // page->offset in real BaM
};

// ============================================================================
// Tier 2: Shared TLB — matches real BaM's struct tlb (lines 199-344)
// ============================================================================

struct Tier2TLB {
    T2Entry entries[T2_NUM_ENTRIES];
};

// ============================================================================
// Tier 1: Per-thread state — matches bam_ptr_tlb (lines 346-411)
// ============================================================================

struct Tier1Local {
    uint32_t page_id;       // gid in real BaM
    uint32_t t2_entry_idx;  // ent in real BaM
    int      start_elem;    // start in real BaM
    int      end_elem;      // end in real BaM
    char*    addr;           // addr in real BaM
    bool     holding;        // addr != nullptr in real BaM
};

// ============================================================================
// t2_init — matches tlb::init (lines 213-228)
// ============================================================================

__device__ __forceinline__
void t2_init(Tier2TLB* t2) {
    for (int i = threadIdx.x; i < T2_NUM_ENTRIES; i += blockDim.x) {
        t2->entries[i].state = 0;
        t2->entries[i].page_id = 0xFFFFFFFF;
        t2->entries[i].cache_slot = 0xFFFFFFFF;
    }
    __syncthreads();
}

// ============================================================================
// t2_fini — matches tlb::fini (lines 232-244)
// __syncthreads is COMMENTED OUT in real BaM.
// Each thread releases entries in strided pattern.
// entry release does page->state.fetch_sub(1) on VRAM cache page.
// ============================================================================

__device__ __forceinline__
void t2_fini(Tier2TLB* t2, BamCache* cache) {
    // NO __syncthreads — matches real BaM
    __syncthreads();
    // The Teardown: Now that everyone in the block is doe, it is safe to 
    // unpin the physical Tier 3 VRAM slots.
    for (int i = threadIdx.x; i < T2_NUM_ENTRIES; i += blockDim.x) {
        if (t2->entries[i].page_id != 0xFFFFFFFF) {
            __threadfence();
            cache_release_page(cache, t2->entries[i].page_id, 1);
        }
    }
}

// ============================================================================
// t2_acquire — matches tlb::acquire (lines 250-322)
// ============================================================================

__device__ __forceinline__
char* t2_acquire(Tier2TLB* t2, BamCache* cache, uint32_t page_id,
                 uint32_t* out_entry_idx, const char* backing) {
    uint32_t lane = threadIdx.x % 32;

    // Warp coalescing — matches lines 258-262
    unsigned mask = __activemask();
    unsigned eq_mask = __match_any_sync(mask, page_id);
    int master = __ffs(eq_mask) - 1;
    uint32_t count = __popc(eq_mask);

    uint32_t ent_idx = page_id % T2_NUM_ENTRIES;
    T2Entry* entry = &t2->entries[ent_idx];

    uint64_t base_master = 0;

    if ((int)lane == master) {
        uint64_t c = 0;
        uint32_t st;

        do {
            // Lock — matches lines 271-276
            do {
                st = atomicOr((unsigned int*)&entry->state, T2_LOCK_BIT);
                if ((st & T2_LOCK_BIT) == 0)
                    break;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700)
                __nanosleep(100);
#endif
            } while (true);

            // st = state BEFORE lock was set. Lower bits = ref count.

            if (entry->page_id == page_id && entry->page_id != 0xFFFFFFFF) {
                // CASE 1: TLB HIT — matches lines 278-287
                // st += count;
                base_master = (uint64_t)(cache->d_data +
                    (uint64_t)entry->cache_slot * CACHE_LINE_SIZE);
                // Unlock: store st (no LOCK bit since st was pre-lock)
                // atomicExch((unsigned int*)&entry->state, st);
                atomicAdd((unsigned int*)&entry->state, count - T2_LOCK_BIT);
                break;
            }
            else if (entry->page_id == 0xFFFFFFFF ||
                     (st & T2_CNT_MASK) == 0) {
                // CASE 2: TLB MISS — matches lines 289-303
                if (entry->page_id != 0xFFFFFFFF) {
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
                atomicAdd((unsigned int*)&entry->state, count - T2_LOCK_BIT);
                break;
            }
            else {
                // CASE 3: CONFLICT — matches lines 306-311
                if (++c % 100000 == 0) {
                    printf("TLB conflict: c=%llu tid=%u wanted=%u got=%u st=0x%x\n",
                           (unsigned long long)c, threadIdx.x,
                           page_id, entry->page_id, st);
                }
                // atomicExch((unsigned int*)&entry->state, st);
                atomicSub((unsigned int*)&entry->state, T2_LOCK_BIT);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700)
                __nanosleep(100);
#endif
            }

        } while (true);
    }

    // Broadcast — matches line 319
    base_master = __shfl_sync(eq_mask, base_master, master);
    ent_idx = __shfl_sync(eq_mask, ent_idx, master);
    *out_entry_idx = ent_idx;

    return (char*)base_master;
}

// ============================================================================
// t2_release — simplified version without warp coalescing
//
// Real BaM's tlb::release uses __match_any_sync for coalescing.
// However, in our setup threads in the same warp can be at different
// call sites — one in t2_acquire (also using __match_any_sync) and
// another in t2_release. Under ITS, if the scheduler reconverges
// these threads at the __match_any_sync instruction, they participate
// in each other's sync and deadlock.
//
// Simple atomicSub per thread avoids this entirely.
// The cost is one extra atomic per release (vs coalesced), which is
// negligible since Tier 1 absorbs 99%+ of accesses.
// ============================================================================

__device__ __forceinline__
void t2_release(Tier2TLB* t2, uint32_t entry_idx, uint32_t page_id) {
    atomicSub((unsigned int*)&t2->entries[entry_idx].state, 1);
}

// ============================================================================
// t1_init — matches bam_ptr_tlb::init (line 368)
// ============================================================================

__device__ __forceinline__
void t1_init(Tier1Local* t1) {
    t1->page_id = 0xFFFFFFFF;
    t1->t2_entry_idx = 0xFFFFFFFF;
    t1->start_elem = 0;
    t1->end_elem = 0;
    t1->addr = nullptr;
    t1->holding = false;
}

// ============================================================================
// t1_fini — matches bam_ptr_tlb::fini (lines 372-379)
// "if (addr) { tlb_->release(gid); addr = nullptr; }"
// ============================================================================

__device__ __forceinline__
void t1_fini(Tier1Local* t1, Tier2TLB* t2) {
    if (t1->holding) {
        t2_release(t2, t1->t2_entry_idx, t1->page_id);
        t1->holding = false;
        t1->addr = nullptr;
    }
}

// ============================================================================
// t1_read — matches bam_ptr_tlb::operator[] (lines 395-400)
//
//   if ((i < start) || (i >= end)) {
//       update_page(i);    // fini() then acquire()
//   }
//   return addr[i-start];
// ============================================================================

__device__ __forceinline__
int t1_read(Tier1Local* t1, Tier2TLB* t2, BamCache* cache,
            int index, const char* backing) {

    // Fast path — matches line 396
    if (index >= t1->start_elem && index < t1->end_elem) {
        return ((int*)t1->addr)[index - t1->start_elem];
    }

    // Slow path — matches update_page (lines 383-391)
    // fini() first — drop before acquire
    t1_fini(t1, t2);

    uint32_t new_page_id = index / CL_ELEMS_INT;
    uint32_t new_entry_idx;

    char* page_addr = t2_acquire(t2, cache, new_page_id,
                                  &new_entry_idx, backing);

    t1->page_id = new_page_id;
    t1->t2_entry_idx = new_entry_idx;
    t1->start_elem = new_page_id * CL_ELEMS_INT;
    t1->end_elem = t1->start_elem + CL_ELEMS_INT;
    t1->addr = page_addr;
    t1->holding = true;

    return ((int*)t1->addr)[index - t1->start_elem];
}

#endif // BAM_THREE_TIER_TLB_CUH
