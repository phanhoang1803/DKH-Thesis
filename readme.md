# News Multimodal Dataset Creation

This repository contains tools for creating multimodal news datasets that pair images with their corresponding captions and context. The scripts facilitate downloading and processing news content from various sources using both direct search (text-to-image) and inverse search (image-to-text) approaches.

## Overview

The dataset creation pipeline supports two main datasets:
- **COSMOS**
- **NewsCLIPPING**

The tools provide functionality for:
1. **Direct annotation collection** - searching for images using text queries
2. **Inverse annotation collection** - finding web pages for given images
3. **Data extraction** - extracting and storing metadata, captions, and images

## Prerequisites

Before running the scripts, ensure you have the following:

- Python 3.7+
- Required Python packages (install via `pip install -r requirements.txt`):
  - requests
  - tqdm
  - Pillow
  - google-api-python-client
  - filelock
  - concurrent.futures

You also need to structure your dataset in the following format:

For NewsCLIPPING dataset (You can download the dataset from [here](https://huggingface.co/datasets/phanhoang1803/test_dataset/tree/main)):
```
test_dataset/
├── visual_news_test/
├── news_clippings_test.json
├── visual_news_test.json
├── links_test.json
```

For COSMOS dataset (You can download the dataset from [here](https://huggingface.co/datasets/phanhoang1803/test_dataset_cosmos/tree/main)):
```
test_dataset_cosmos/
├── test/
├── public_test_acm.json
```

## Dataset Creation Workflows

### I. Direct Annotation Collection

Direct annotation involves searching for images using text queries from existing datasets.

For Google Custom Search Engine (CSE) functionality:
- Google API Key
- Google Custom Search Engine ID

To get the Google API Key and Google Custom Search Engine ID you can follow the instructions [here](https://developers.google.com/custom-search/v1/introduction)

#### 1. COSMOS Dataset

For the COSMOS dataset, use the following command:

```bash
python download_direct_annotations_dirs_cosmos.py \
    --cosmos_data_path <path_to_public_test_acm.json> \
    --save_folder_path <output_directory> \
    --google_api_key <your_google_api_key> \
    --google_cse_id <your_cse_id> \
    --start_idx <start_idx> \
    --end_idx <end_idx> \
    --skip_existing
```

#### 2. NewsCLIPPING Dataset

For the NewsCLIPPING dataset, use:

```bash
python download_direct_annotations_dirs.py \
    --visual_news_data_path <path_to_visual_news_test.json> \
    --news_clippings_data_path <path_to_news_clippings_test.json> \
    --save_folder_path <output_directory> \
    --google_api_key <your_google_api_key> \
    --google_cse_id <your_cse_id> \
    --start_idx <start_idx> \
    --end_idx <end_idx> \
    --skip_existing
```

### II. Inverse Annotation Collection

Inverse annotation involves finding textual context for images using image search or existing search results.

To run the Google Cloud Vision ([Detect Web](https://cloud.google.com/vision/docs/detecting-web)), you need to set up a service account and download the credentials file.

You can use the `google_cred_json` argument to specify the path to the credentials file.

#### 1. COSMOS Dataset (Using Google Image Search)

```bash
python download_inverse_annotations_dirs_cosmos.py \
    --cosmos_data_path <path_to_public_test_acm.json> \
    --save_folder_path <output_directory> \
    --start_idx <start_idx> \
    --end_idx <end_idx> \
    --skip_existing
```

#### 2. NewsCLIPPING Dataset (Using Google Image Search)

```bash
python download_inverse_annotations_dirs.py \
    --visual_news_data_path <path_to_visual_news_test.json> \
    --news_clippings_data_path <path_to_news_clippings_test.json> \
    --save_folder_path <output_directory> \
    --start_idx <start_idx> \
    --end_idx <end_idx> \
    --skip_existing
```

#### 3. NewsCLIPPING Dataset (Using Existing Search Results)

```bash
python download_inverse_annotations_dirs_from_searched_results.py \
    --existing_results_path <path_to_links_test.json> \
    --save_folder_path <output_directory> \
    --start_idx <start_idx> \
    --end_idx <end_idx> \
    --skip_existing
```

## Common Arguments

| Argument | Description |
|----------|-------------|
| `--save_folder_path` | Directory to save downloaded content |
| `--start_idx` | Starting index for items to process |
| `--end_idx` | Ending index for items to process |
| `--skip_existing` | Skip items that already exist in the output directory |

## Parallelization

For efficient processing of large datasets, you can run multiple instances of the scripts with different index ranges:

```bash
# Terminal 1
python download_direct_annotations_dirs.py --start_idx 0 --end_idx 500 ...

# Terminal 2
python download_direct_annotations_dirs.py --start_idx 500 --end_idx 1000 ...
```

## Output Structure

The scripts create a hierarchical directory structure:

```
save_folder_path/
├── direct_search/
│   └── test/
│       ├── 0/
│       │   ├── 0.jpg
│       │   ├── 0.txt (HTML)
│       │   ├── 1.jpg
│       │   ├── 1.txt (HTML)
│       │   └── direct_annotation.json
│       ├── 1/
│       └── test.json
└── inverse_search/
    └── test/
        ├── 0/
        ├── 1/
        └── test.json
```

Each numbered directory contains:
- Downloaded images
- HTML content from source pages
- A JSON file with metadata and captions

## Error Handling and Resuming

The scripts include mechanisms for:
- Handling network errors
- Resuming interrupted downloads
- Skipping existing items with the `--skip_existing` flag
- Logging progress and errors