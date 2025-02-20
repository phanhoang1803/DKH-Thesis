#!/bin/bash

API_KEY="AIzaSyBJkAID3urNPfJDJQSzt1jXSmc7mLly4EA"
CSE_ID="94f1749818bdb40a1"
START=1000
END=7000
STEP=500
MAX_JOBS=8  # Adjust this to limit the number of parallel processes

for ((i=START; i<END; i+=STEP)); do
    j=$((i+STEP))
    echo "Running: start_idx=$i, end_idx=$j in parallel"
    
    python3 download_direct_annotations_dirs.py \
        --google_api_key "$API_KEY" \
        --google_cse_id "$CSE_ID" \
        --skip_existing \
        --start_idx "$i" \
        --end_idx "$j" &

    # Limit the number of parallel jobs
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
done

wait  # Wait for all background jobs to finish
echo "All downloads completed."
