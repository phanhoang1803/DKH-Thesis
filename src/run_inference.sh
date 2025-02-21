#!/bin/bash

# Configuration
START=0
END=200
STEP=20
MAX_JOBS=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to show current running jobs
show_running_jobs() {
    local running=$(jobs -r | wc -l)
    echo "Current running jobs: $running"
}

# Loop through indices in steps
for ((i=START; i<END; i+=STEP)); do
    j=$((i+STEP))
    echo "Running: start_idx=$i, end_idx=$j"
    
    # Run Python script and redirect output to log file
    {
        python src/inference_use_searched_evidences.py \
            --start_idx "$i" \
            --end_idx "$j" \
            --skip_existing \
            --llm_model "gemini" \
            2>&1 
    } > "logs/log_${i}_${j}.txt" &

    # Wait if max jobs limit is reached
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        show_running_jobs
        sleep 1
    done
done

# Wait for all background jobs to complete
echo "Waiting for all jobs to complete..."
wait
echo "All jobs completed!"

# Optional: Display summary of log files
echo "Checking log files..."
for ((i=START; i<END; i+=STEP)); do
    j=$((i+STEP))
    echo "Log for $i-$j:"
    tail -n 1 "logs/log_${i}_${j}.txt"
done