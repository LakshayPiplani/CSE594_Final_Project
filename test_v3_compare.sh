/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o test_v3_compare /tmp/test_v3_compare.cu 2>/dev/null
for i in 1 2 3 4 5; do
    echo "=== Run $i ==="
    ./test_v3_compare 2>&1 | grep "err="
done