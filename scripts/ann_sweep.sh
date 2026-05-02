#!/bin/bash

# 1. Enforce that the user provides exactly one argument
if [ "$#" -ne 3 ]; then
    echo "Error: Missing data directory."
    echo "Usage: ./run_sweep.sh <path_to_data_dir> <scale_of_data> <cache_perc>"
    exit 1
fi

# 2. Store the input argument in a variable
DATA_DIR=$1
SCALE=$2
CACHE_PERC=$3

LOG_FILE="./results/ann_results.log"
CSV_FILE="./results/ann_results.csv"

# 3. Add a trailing slash just in case the user forgot it
# (This ensures paths like /scratch/ljp5718/knn_large work correctly)
if [[ "${DATA_DIR}" != */ ]]; then
    DATA_DIR="${DATA_DIR}/"
fi

echo "=========================================="
echo "Compling ANN Data Generation code"
echo "=========================================="
/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o gen_knn_data ./src/gen_knn_data.cu

echo "=========================================="
echo "Generating ANN Data"
echo "=========================================="
# ./gen_knn_data --data_dir "$DATA_DIR" --scale "$SCALE"

echo "=========================================="
echo "Compling ANN Benchmark code"
echo "=========================================="
/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o ann_benchmark ./src/ann_benchmark.cu

echo "=========================================="
echo " Starting ANN Sweep"
echo " Data Directory: $DATA_DIR"
echo "=========================================="

# 4. Your sweep loops (Example: sweeping WPQ)
for wpq in 1 2 3 4 5 8 16 32; do
    echo "-> Running WPQ = $wpq"
    
    # Pass the DATA_DIR variable directly into your executable's --dir flag
    ./ann_benchmark --data_dir "$DATA_DIR" --scale "$SCALE" --cache "$CACHE_PERC" --wpq $wpq
    
    echo "------------------------------------------"
done 2>&1 | tee $LOG_FILE

echo "Sweep Complete."

# ==========================================
# EXTRACT TO CSV
# ==========================================

echo "Extracting sweep results to $CSV_FILE..."

awk -F'|' '
BEGIN { 
    # Print the exact CSV header matching your ANN table
    print "Method,WPQ,Threads_per_Q,Blocks,Time_ms,VRAM_Hits,DRAM_Misses,Status" 
}
/^Target_T/ || /^BaM/ {
    # Loop through every column and strip leading/trailing whitespace
    for(i=1; i<=NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i);
    }
    # Print the cleaned variables separated by commas
    printf "%s,%s,%s,%s,%s,%s,%s,%s\n", $1, $2, $3, $4, $5, $6, $7, $8
}' $LOG_FILE > $CSV_FILE

echo "CSV extraction complete. Saved to $CSV_FILE"