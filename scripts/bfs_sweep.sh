#!/bin/bash

echo "=========================================="
echo "Compling BFS Data Generation code"
echo "=========================================="
g++ -Iinclude -O3 -fopenmp ./src/gen_bfs_data.cpp -o gen_bfs_data

echo "=========================================="
echo "Generating BFS Data"
echo "=========================================="
mkdir -p /scratch/ljp5718/bfs_data
gen_bfs_data --scale 20 --target_degree 2048 --graph_type uniform --out_dir /scratch/ljp5718/bfs_data

echo "=========================================="
echo "Compling BFS Benchmnark code"
echo "=========================================="
/usr/local/cuda-12.4/bin/nvcc -std=c++17 -arch=sm_86 -O3 --expt-relaxed-constexpr -Iinclude -o bfs_tpn_benchmark ./src/bfs_tpn_benchmark.cu 

LOG_FILE="./results/bfs_tpn_sweep.log"
CSV_FILE="./results/bfs_results.csv"

echo "=========================================="
echo " Starting BFS Sweep"
echo " Cache Target: 10%"
echo "=========================================="

# Sweep exactly the template values compiled in the C++ binary
for tpn in 32 64 96 128 160 192 224 256 288 320; do
    echo "-> Running TPN = $tpn"
    
    # Execute the benchmark
    ./bfs_tpn_benchmark --cache_perc 0.1 --tpn $tpn --data_dir /scratch/ljp5718/bfs_data
    
    echo "------------------------------------------"
# Capture all output, print it to screen, and save to the log file
done 2>&1 | tee $LOG_FILE

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
    print "Method,TPN,Time_ms,Edges_Acc,T1_Hits,T2_T3_Hits,DRAM_Misses,Status" 
}
/^Target_T/ || /^BaM/ {
    # Loop through every column and strip leading/trailing whitespace
    for(i=1; i<=NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i);
    }
    # Print the cleaned variables separated by commas
    printf "%s,%s,%s,%s,%s,%s,%s,%s\n", $1, $2, $3, $4, $5, $6, $7, $8
}' $LOG_FILE > $CSV_FILE

echo "CSV extraction complete."