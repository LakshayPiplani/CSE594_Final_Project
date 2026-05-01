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

# 3. Add a trailing slash just in case the user forgot it
# (This ensures paths like /scratch/ljp5718/knn_large work correctly)
if [[ "${DATA_DIR}" != */ ]]; then
    DATA_DIR="${DATA_DIR}/"
fi

echo "=========================================="
echo " Starting ANN Sweep"
echo " Data Directory: $DATA_DIR"
echo "=========================================="

# 4. Your sweep loops (Example: sweeping WPQ)
for wpq in 1 2 4 8; do
    echo "-> Running WPQ = $wpq"
    
    # Pass the DATA_DIR variable directly into your executable's --dir flag
    ./ann_benchmark --dir "$DATA_DIR" --scale "$SCALE" --cache "$CACHE_PERC" --wpq $wpq
    
    echo "------------------------------------------"
done 2>&1 | tee ./results/ann_results.log

echo "Sweep Complete."