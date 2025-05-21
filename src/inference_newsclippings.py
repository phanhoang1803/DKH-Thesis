# inference.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from typing import Optional, Union
import numpy as np
import openai
from modules.entities_module import VisualEntityExtractor
from modules.reasoning_module import GeminiConnector, GPTConnector, GeminiVisionConnector
from mdatasets.newsclipping_datasets import MergedBalancedNewsClippingDataset
from dotenv import load_dotenv
import argparse
import torch
import json
import time
from src.utils.utils import process_results, NumpyJSONEncoder
import google
from modules.evidence_module import ImageEvidencesModule, TextEvidencesModule
from modules.reasoning_module.debate.async_debate import AsyncDebate

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_dataset", 
                       help="")
    parser.add_argument("--entities_path", type=str, default="test_dataset/links_test.json")
    parser.add_argument("--image_evidences_path", type=str, default="queries_dataset/merged_balanced/inverse_search/test/test.json", 
                        help="")
    parser.add_argument("--text_evidences_path", type=str, default="queries_dataset/merged_balanced/direct_search/test/test.json", 
                        help="")
    parser.add_argument("--context_dir_path", type=str, default="queries_dataset/merged_balanced/context/test")
    parser.add_argument("--img_des_dir_path", type=str, default="queries_dataset/merged_balanced/image_description/test")
    parser.add_argument("--random_index_path", type=str, default=None)
    
    parser.add_argument("--gemini_api_key", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default="gemini", choices=["gpt", "gemini"])
    
    parser.add_argument("--vlm_model1", type=str, default="gemini", choices=["gpt", "gemini"])
    parser.add_argument("--vlm_model2", type=str, default="gemini", choices=["gpt", "gemini"])
    parser.add_argument("--vlm_model3", type=str, default="gemini", choices=["gpt", "gemini"])
    parser.add_argument("--vlm_model1_name", type=str, default="gemini-2.0-flash-001")
    parser.add_argument("--vlm_model2_name", type=str, default="gemini-2.0-flash-001")
    parser.add_argument("--vlm_model3_name", type=str, default="gemini-2.0-flash-001")
    parser.add_argument("--vlm_api_key1", type=str, default=None)
    parser.add_argument("--vlm_api_key2", type=str, default=None)
    parser.add_argument("--vlm_api_key3", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--output_dir_path", type=str, default="./result_debate_with_newspaper_2.0/")
    parser.add_argument("--errors_dir_path", type=str, default="./errors_debate_with_newspaper_2.0/")
    
    # Integrated similarity weights
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for visual similarity (S_visual)")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for textual similarity (S_textual)")
    parser.add_argument("--gamma", type=float, default=0.2, help="Weight for interaction term (S_visual * S_textual)")
    
    # Dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_shuffle", action='store_false')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    
    # Model configs
    parser.add_argument("--max_debate_rounds", type=int, default=3, help="Maximum number of debate rounds")
    
    return parser.parse_args()

def inference(
             async_debate: AsyncDebate,
             data: dict,
             idx: int,
             context_dir_path: str,
             img_des_dir_path: str,
             alpha: float = 0.5,
             beta: float = 0.5,
             gamma: float = 0.2,
             ):
    start_time = time.time()
    
    # Get base64 encoded image and caption
    image_base64 = data["image_base64"]
    caption = data["caption"]
    news_content = data["content"]
    
    # Run the debate
    result = async_debate.run_debate(idx, image_base64, caption, news_content)
    
    # Add metadata and timing information
    result["metadata"] = {
        "idx": idx,
        "timestamp": datetime.now().isoformat(),
        "processing_time": time.time() - start_time,
        "parameters": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
    }
    
    # Add ground truth information from data if available
    if "label" in data:
        result["ground_truth"] = {
            "label": data["label"]
        }
    
    return process_results(result)

def get_transform():
    return None

def main():
    args = arg_parser()
    
    # Setup environment
    load_dotenv()
    
    # Make results and error folders
    if not os.path.exists(args.output_dir_path):
        os.makedirs(args.output_dir_path)
    if not os.path.exists(args.errors_dir_path):
        os.makedirs(args.errors_dir_path)
    
    print(args.gemini_api_key)
    
    print("Connecting to LLM Model...")
    if args.llm_model == "gpt":
        llm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.llm_model == "gemini":
        llm_connector = GeminiConnector(
            api_key=args.gemini_api_key if args.gemini_api_key else os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.0-flash-001"
        )
    else:
        raise ValueError(f"Invalid LLM model: {args.llm_model}")
    print("LLM Model Connected")
        
    print("Connecting to VLM Model 1...")
    if args.vlm_model1 == "gpt":
        vlm_connector1 = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.vlm_model1 == "gemini":
        vlm_connector1 = GeminiConnector(
            api_key=args.vlm_api_key1 if args.vlm_api_key1 else os.environ["GEMINI_API_KEY"],
            model_name=args.vlm_model1_name,
            # connector_name="VLM 1"
        )
    else:
        raise ValueError(f"Invalid VLM model: {args.vlm_model1  }")
    print("VLM Model 1 Connected")
        
    print("Connecting to VLM Model 2...")
    if args.vlm_model2 == "gpt":
        vlm_connector2 = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.vlm_model2 == "gemini":
        vlm_connector2 = GeminiConnector(
            api_key=args.vlm_api_key2 if args.vlm_api_key2 else os.environ["GEMINI_API_KEY"],
            model_name=args.vlm_model2_name,
            # connector_name="VLM 2"
        )
    else:
        raise ValueError(f"Invalid VLM model: {args.vlm_model2}")
    print("VLM Model 2 Connected")
        
    print("Connecting to VLM Model 3...")
    if args.vlm_model3 == "gpt":
        vlm_connector3 = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.vlm_model3 == "gemini":
        vlm_connector3 = GeminiConnector(
            api_key=args.vlm_api_key3 if args.vlm_api_key3 else os.environ["GEMINI_API_KEY"],
            model_name=args.vlm_model3_name,
            # connector_name="VLM 3"
        )
    else:
        raise ValueError(f"Invalid VLM model: {args.vlm_model3}")
    print("VLM Model 3 Connected")

    # Initialize modules
    print("Initializing modules...")
    entities_module = VisualEntityExtractor(args.entities_path)
    image_evidences_module = ImageEvidencesModule(args.image_evidences_path)
    text_evidences_module = TextEvidencesModule(args.text_evidences_path)
    print("Modules initialized")
    
    # Initialize AsyncDebate
    async_debate = AsyncDebate(
        image_evidences_module=image_evidences_module,
        text_evidences_module=text_evidences_module,
        max_rounds=args.max_debate_rounds,
        vlm_connector1=vlm_connector1,
        vlm_connector2=vlm_connector2,
        vlm_connector3=vlm_connector3,
        image_information_save_dir=args.img_des_dir_path
    )
    # Load dataset
    dataset = MergedBalancedNewsClippingDataset(args.data_path)
    
    # Determine indices to process
    start_idx = args.start_idx if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else len(dataset) - 1
    
    # Load random indices if specified
    if args.random_index_path is not None:    
        try:
            with open(args.random_index_path, "r") as f:
                random_index = [int(line.strip()) for line in f.readlines()]
        except Exception as e:
            raise e
    else:
        random_index = list(range(start_idx, end_idx + 1))
        
    # Select indices between start_idx and end_idx
    indices = [idx for idx in random_index if start_idx <= idx <= end_idx]
    
    # Validate indices
    if start_idx >= len(dataset):
        raise ValueError(f"Start index {start_idx} is out of range for dataset of length {len(dataset)}")
    if end_idx >= len(dataset):
        end_idx = len(dataset) - 1
    if start_idx > end_idx:
        raise ValueError(f"Start index {start_idx} is greater than end index {end_idx}")
    
    # Process data and save results
    results = []
    error_items = []
    total_start_time = time.time()
    
    print(f"Processing {len(indices)} items from index {start_idx} to {end_idx}")

    for i in range(len(indices)):
        idx = indices[i]
        retry_count = 0
        max_retries = 3
    
        while retry_count <= max_retries:
            try:
                print(f"\n--- Processing item {idx} (Attempt {retry_count + 1}/{max_retries + 1}) ---")
                res_path = os.path.join(args.output_dir_path, f"result_{idx}.json")
                
                # Skip if result already exists and skip_existing is set
                if args.skip_existing and os.path.exists(res_path):
                    print(f"Skipping existing result for index {idx}")
                    break
                
                # Get dataset item
                item = dataset[idx]
                
                # Run inference
                result = inference(
                    async_debate=async_debate,
                    data=item,
                    idx=idx,
                    context_dir_path=args.context_dir_path,
                    img_des_dir_path=args.img_des_dir_path,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                )
                
                # Save result
                with open(res_path, "w", encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
                
                results.append(result)
                print(f"Saved result to {res_path}")
                
                # Print progress
                progress = (i + 1) / len(indices) * 100
                elapsed_time = time.time() - total_start_time
                estimated_total = elapsed_time / (i + 1) * len(indices)
                estimated_remaining = estimated_total - elapsed_time
                
                print(f"Progress: {progress:.2f}% ({i+1}/{len(indices)})")
                print(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining: {estimated_remaining:.2f}s")
                
                break  # Success - exit the retry loop
                
            except KeyError as e:
                print(f"KeyError processing item {idx}: {e}")
                break  # Don't retry for these errors
            except openai.BadRequestError as e:
                print(f"BadRequestError processing item {idx}: {e}")
                break  # Don't retry for these errors
            except json.decoder.JSONDecodeError as e:
                print(f"JSONDecodeError processing item {idx}: {e}")
                # break  # Don't retry for these errors
                raise e
            except UnicodeEncodeError as e:
                print(f"UnicodeEncodeError processing item {idx}: {e}")
                break  # Don't retry for these errors
            except google.api_core.exceptions.ResourceExhausted as e:
                retry_count += 1
                wait_time = 30
                print(f"Gemini quota exceeded for item {idx} (Attempt {retry_count}/{max_retries + 1})")
                print(f"Waiting {wait_time} seconds before retry...")
                
                if retry_count <= max_retries:
                    time.sleep(wait_time)  # Wait before retry
                else:
                    print(f"Max retries ({max_retries}) exceeded for item {idx}, moving to next item")
                    error_item = {
                        "idx": idx,
                        "error": "Gemini quota exceeded after max retries",
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(os.path.join(args.errors_dir_path, f"error_{idx}.json"), "w") as f:
                        json.dump(error_item, f, indent=2, ensure_ascii=False)
                    error_items.append(error_item)
                    
            except Exception as e:
                error_item = {
                    "idx": idx,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                with open(os.path.join(args.errors_dir_path, f"error_{idx}.json"), "w") as f:
                    json.dump(error_item, f, indent=2, ensure_ascii=False)
                error_items.append(error_item)
                print(f"Error processing item {idx}: {e}")
                # raise e
                break  # Move to next item
                
    total_time = time.time() - total_start_time
    print(f"\nProcessing complete. Total time: {total_time:.2f}s")
    print(f"Processed {len(indices)} items with {len(error_items)} errors")
    
    # Save summary
    summary = {
        "total_items": len(indices),
        "successful_items": len(indices) - len(error_items),
        "error_items": len(error_items),
        "total_time": total_time,
        "average_time_per_item": total_time / len(indices) if len(indices) > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    main()