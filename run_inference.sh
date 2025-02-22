#!/bin/bash

START=30
END=4000
STEP=100
MAX_JOBS=4  # Adjust this to limit the number of parallel processes

for ((i=START; i<END; i+=STEP)); do
    j=$((i+STEP))
    echo "Running: start_idx=$i, end_idx=$j in parallel"
    
    python3 src/inference_use_searched_evidences.py \
            --start_idx "$i" \
            --end_idx "$j" \
            --skip_existing \
            --llm_model "gemini" &

    # Limit the number of parallel jobs
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
done

wait  # Wait for all background jobs to finish
echo "All downloads completed."
