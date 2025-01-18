# DKH-Thesis

## Installation

1. Clone the repository
```bash
git clone https://github.com/phanhoang1803/DKH-Thesis.git
cd DKH-Thesis
```

2. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate   # On Windows
```

3. Install dependencies and project
```bash
pip install -r requirements.txt
```

5. Set up environment variables by creating a .env file in the root directory:
```
GOOGLE_API_KEY=<your_google_api_key>
CX=<your_custom_search_engine_id>
GEMINI_API_KEY=<your_gemini_api_key>
HF_TOKEN=<your_huggingface_token>
```

## Usage

### Running Evidence Retrieval
The evidence retrieval module can be run separately to gather evidence for a dataset of news captions:
```bash
python src/prerun_external_module_script.py \
    --data_path "data/your_test_data.json" \
    --output_path "./cache/external_evidence_cache.json" \
    --start_idx 0
```

Key parameters:

- data_path: Path to NewsClipping Json file
- output_path: Where to save the retrieved evidence cache
- start_idx: Index to start processing from (useful for resuming interrupted runs)

### Running Inference
To run the full inference pipeline which includes internal checking, external checking, and final verification:
```bash
python src/inference_2.py \
    --data_path "data/your_test_data.json" \
    --output_dir_path "./results/" \
    --errors_dir_path "./errors/" \
    --device "cuda" \
    --batch_size 8 \
    --start_idx 0 \
    --ner_model "dslim/bert-large-NER" \
    --blip_model "Salesforce/blip2-opt-2.7b" \
    --llm_model "meta-llama/Llama-3.2-1B-Instruct"
```

Key parameters:

- data_path: Path to JSON dataset
- output_dir_path: Directory to save inference results
- errors_dir_path: Directory to save error logs
- device: Device to run models on ("cuda" or "cpu")
- batch_size: Batch size for processing
- start_idx: Index to start processing from
- ner_model: Model for Named Entity Recognition
- blip_model: Model for image understanding
- llm_model: Language model for verification