#!/bin/bash

# Create directories
mkdir -p "./ablation_results/no_image_evidences"
mkdir -p "./ablation_results/no_text_evidences"
mkdir -p "./ablation_results/no_filters"
mkdir -p "./ablation_results/baseline"

mkdir -p "./ablation_errors/no_image_evidences"
mkdir -p "./ablation_errors/no_text_evidences"
mkdir -p "./ablation_errors/no_filters"
mkdir -p "./ablation_errors/baseline"

# Set common parameters
COMMON_PARAMS="--data_path test_dataset \
               --entities_path test_dataset/links_test.json \
               --image_evidences_path queries_dataset/merged_balanced/inverse_search/test/test.json \
               --text_evidences_path queries_dataset/merged_balanced/direct_search/test/test.json \
               --context_dir_path queries_dataset/merged_balanced/context/test \
               --llm_model gemini \
               --vlm_model gpt \
               --start_idx 0 \
               --end_idx 99 \
               --skip_existing"

# Run baseline (original implementation)
echo "Running baseline test..."
python ablation_studies.py $COMMON_PARAMS --ablation_type baseline \
    --output_dir_path ./ablation_results/ \
    --errors_dir_path ./ablation_errors/

# Run ablation study 1: Without image evidences
echo "Running ablation without image evidences..."
python ablation_studies.py $COMMON_PARAMS --ablation_type no_image_evidences \
    --output_dir_path ./ablation_results/ \
    --errors_dir_path ./ablation_errors/

# Run ablation study 2: Without text evidences
echo "Running ablation without text evidences..."
python ablation_studies.py $COMMON_PARAMS --ablation_type no_text_evidences \
    --output_dir_path ./ablation_results/ \
    --errors_dir_path ./ablation_errors/

# Run ablation study 3: Without filters
echo "Running ablation without filters..."
python ablation_studies.py $COMMON_PARAMS --ablation_type no_filters \
    --output_dir_path ./ablation_results/ \
    --errors_dir_path ./ablation_errors/

echo "All ablation studies completed."