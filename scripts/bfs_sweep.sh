#!/bin/bash

# Configuration
BFS_EXEC="../bfs_warp_benchmark"
SCALE="20"
LOG_FILE="bfs_sweep_results.log"

# Clear the log file if it already exists
> "$LOG_FILE"
chmod 666 "$LOG_FILE"

echo "=== Starting BFS Cache Sweep ===" | tee -a "$LOG_FILE"

# Loop explicitly through fractional values
for perc in 0.1 0.2 0.3 0.4 0.5; do
    echo "" | tee -a "$LOG_FILE"
    echo "---------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Executing with cache percentage: $perc" | tee -a "$LOG_FILE"
    echo "---------------------------------------------------" | tee -a "$LOG_FILE"
    
    # Execute the command. 
    # 2>&1 merges stderr into stdout so crashes are captured.
    # tee -a appends to the log file while printing to your screen.
    $BFS_EXEC "$SCALE" "$perc" 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"

# ==============================================================================
# Parse the log file and generate the CSV summary block
# ==============================================================================
echo "Generating CSV Summary..." | tee -a "$LOG_FILE"

awk '
BEGIN { 
    print "=== CSV Summary ==="
    print "Cache_Perc,Time_ms,T1_Hits,T2_Hits,DRAM_Access(Misses)" 
}
# Grab the current cache percentage from our bash echo
/Executing with cache percentage:/ { 
    perc = $5 
}
# Parse the specific row for BaM

# Token indexes: 
# $1=Warp-coal, $2=BaM, $3=3T-TLB, $4=|, $5=Time, $6=|, $7=Edges, $8=|, $9=T1_Hits, $10=|, $11=T2_Hits, $12=|, $13=Misses
/Warp-coal BaM 3T-TLB/ { 
    time_ms = $5
    t1_hits = $9
    t2_hits = $11
    dram_access = $13
    
    printf "%s,%.2f,%s,%s,%s\n", perc, time_ms, t1_hits, t2_hits, dram_access
}
' "$LOG_FILE" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Sweep Complete. Log saved to $LOG_FILE ===" | tee -a "$LOG_FILE"