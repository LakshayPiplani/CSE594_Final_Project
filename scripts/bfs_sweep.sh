#!/bin/bash

LOG_FILE="./results/bfs_tpn_sweep.log"

echo "=========================================="
echo " Starting BFS Sweep"
echo " Cache Target: 10%"
echo "=========================================="

# Sweep exactly the template values compiled in the C++ binary
for tpn in 1 4 8 16 32 64 128; do
    echo "-> Running TPN = $tpn"
    
    # Execute the benchmark
    ./bfs_tpn_benchmark --cache_perc 0.1 --tpn $tpn
    
    echo "------------------------------------------"
# Capture all output, print it to screen, and save to the log file
done 2>&1 | tee $LOG_FILE

echo "Sweep Complete. Logs saved to $LOG_FILE"