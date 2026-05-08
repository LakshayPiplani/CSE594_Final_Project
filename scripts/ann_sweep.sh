#!/bin/bash


DATA_DIR=/scratch/ljp5718/knn_large
SCALE=22
CACHE_PERC=0.1
TLB=3

LOG_FILE="./results/ann_results_threetier.log"
CSV_FILE="./results/ann_results_threetier.csv"

# 3. Add a trailing slash just in case the user forgot it
# (This ensures paths like /scratch/ljp5718/knn_large work correctly)
if [[ "${DATA_DIR}" != */ ]]; then
    DATA_DIR="${DATA_DIR}/"
fi

echo "=========================================="
echo "Compling ANN Data Generation code"
echo "=========================================="
/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o gen_knn_data ./src/ann_benchmark.cu

echo "=========================================="
echo "Generating ANN Data"
echo "=========================================="
# ./gen_knn_data --data_dir "$DATA_DIR" --scale "$SCALE"

echo "=========================================="
echo "Compling ANN Benchmnark code"
echo "=========================================="
/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o ann_benchmark ./src/ann_benchmark.cu 

echo "=========================================="
echo " Starting ANN Sweep"
echo " Data Directory: $DATA_DIR"
echo "=========================================="

# 4. Your sweep loops (Example: sweeping WPQ)
for wpq in 1 2 4 8 16 32; do
    echo "-> Running WPQ = $wpq"
    
    # Pass the DATA_DIR variable directly into your executable's --dir flag
    ./ann_benchmark --data_dir "$DATA_DIR" --scale "$SCALE" --cache "$CACHE_PERC" --wpq $wpq --tlb "$TLB"
    
    echo "------------------------------------------"
done 2>&1 | tee "$LOG_FILE"

echo "Sweep Complete. Logs saved to $LOG_FILE"

# ==========================================
# EXTRACT TO CSV
# ==========================================

echo "=========================================="
echo "Extracting sweep results to $CSV_FILE..."
echo "=========================================="

awk -F'|' '
BEGIN { 
    # Print the CSV header
    print "Method,WPQ,Threads_Per_Q,Blocks,Time_ms,VRAM_Hits,DRAM_Misses,Status" 
}
/^Target_T/ || /^BaM/ {
    # Loop through every column and strip leading/trailing whitespace
    for(i=1; i<=NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i);
    }
    # Print the cleaned variables separated by commas
    printf "%s,%s,%s,%s,%s,%s,%s,%s\n", $1, $2, $3, $4, $5, $6, $7, $8
}' "$LOG_FILE" > "$CSV_FILE"

echo "CSV extraction complete."