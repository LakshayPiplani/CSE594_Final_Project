#!/bin/bash

# Configuration
ANN_EXEC="../ann_benchmark"
DATA_DIR="/scratch/ljp5718/"
ANN_LOG_FILE="ann_results.log"

# Clear the log file if it already exists from a previous run
> "$ANN_LOG_FILE"
chmod 666 "$ANN_LOG_FILE"

echo "=== Starting BaM Cache Sweep ===" | tee -a "$ANN_LOG_FILE"

# Loop explicitly through your requested fractional values
for perc in 0.1 0.2 0.3 0.4 0.5; do
    echo "" | tee -a "$ANN_LOG_FILE"
    echo "---------------------------------------------------" | tee -a "$ANN_LOG_FILE"
    echo "Executing with cache percentage: $perc" | tee -a "$ANN_LOG_FILE"
    echo "---------------------------------------------------" | tee -a "$ANN_LOG_FILE"
    
    # Execute the command
    $ANN_EXEC "$DATA_DIR" 20 "$perc" 2>&1 | tee -a "$ANN_LOG_FILE"
done

echo "" | tee -a "$ANN_LOG_FILE"

# ==============================================================================
# NEW: Parse the log file and generate the CSV summary block
# ==============================================================================
echo "Generating CSV Summary..."

# Run awk over the log file we just created, and append the output to the same file
awk '
BEGIN { 
    print "=== CSV Summary ==="
    print "Cache_Perc,BaM_Time_ms,PCIe_Traffic_MB" 
}
# Extract the percentage
/Executing with cache percentage:/ { 
    perc = $5 
}
# Set a flag so we only grab the Time from the BaM section, not Target T
/\[BaM\] Three-tier TLB cache/ { 
    in_bam = 1 
}
# Extract the BaM Time
in_bam && /Time:/ { 
    bam_time = $2 
}
# Extract the BaM PCIe traffic, print the CSV row, and reset the flag
in_bam && /PCIe traffic:/ { 
    traffic = $3
    print perc "," bam_time "," traffic
    in_bam = 0 
}
' "$ANN_LOG_FILE" | tee -a "$ANN_LOG_FILE"

echo "" | tee -a "$ANN_LOG_FILE"
echo "=== Sweep Complete. Log saved to $ANN_LOG_FILE ===" | tee -a "$ANN_LOG_FILE"