NVCC = /usr/local/cuda-12.4/bin/nvcc
NVCC_FLAGS = -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude
DEBUG_FLAGS = -std=c++17 -arch=sm_86 -G -g -Iinclude

# Phase 1: Warp coalescer
test_coalescer: tests/test_coalescer.cu include/coalescer.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_coalescer_debug: tests/test_coalescer.cu include/coalescer.cuh
	$(NVCC) $(DEBUG_FLAGS) -o $@ $<

# Phase 2: Software cache
test_cache: tests/test_cache.cu include/cache.cuh include/coalescer.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_cache_debug: tests/test_cache.cu include/cache.cuh include/coalescer.cuh
	$(NVCC) $(DEBUG_FLAGS) -o $@ $<

# Phase 3: Queues (will add later)
# Phase 4: bam::array (will add later)
# Phase 5: Applications (will add later)

clean:
	rm -f test_coalescer test_coalescer_debug test_cache test_cache_debug

.PHONY: all clean
all: test_coalescer test_cache
