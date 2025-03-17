# advanced_ablation_studies.py

import argparse
import json
import os
import time
import torch
from typing import Optional, Union, List

import numpy as np
import openai
from dotenv import load_dotenv

from modules import EntitiesModule, GPTConnector, GeminiConnector, ImageEvidencesModule, TextEvidencesModule
from mdatasets.newsclipping_datasets import MergedBalancedNewsClippingDataset
from templates import get_visual_prompt, get_final_prompt
from templates_for_generating_context import (
    CAPTION_CONTEXT_CHECKING_RESPONSE_SCHEMA, 
    CONTEXT_RESPONSE_SCHEMA, 
    get_context_prompt, 
    get_caption_context_checking_prompt, 
    SYSTEM_PROMPT_FOR_VLM_GENERATED_CONTEXT,
    SYSTEM_PROMPT_FOR_CAPTION_CONTEXT_CHECKING
)
from src.modules.reasoning_module.connector.gpt import VISUAL_RESPONSE_SCHEMA, FINAL_RESPONSE_SCHEMA
from src.utils.utils import process_results

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_dataset")
    parser.add_argument("--entities_path", type=str, default="test_dataset/links_test.json")
    parser.add_argument("--image_evidences_path", type=str, default="queries_dataset/merged_balanced/inverse_search/test/test.json")
    parser.add_argument("--text_evidences_path", type=str, default="queries_dataset/merged_balanced/direct_search/test/test.json")
    parser.add_argument("--context_dir_path", type=str, default="queries_dataset/merged_balanced/context/test")
    parser.add_argument("--random_index_path", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default="gemini", choices=["gpt", "gemini", "fireworks"])
    parser.add_argument("--vlm_model", type=str, default="gpt", choices=["gpt", "gemini", "fireworks"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_shuffle", action='store_false')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--ner_model", type=str, default="dslim/bert-large-NER")
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip2-opt-2.7b")
    
    # Advanced ablation study parameters
    parser.add_argument("--rank", type=int, required=True, choices=[1, 2, 3],
                       help="Maximum number of evidences to use")
    parser.add_argument("--filter_type", type=str, required=True, 
                       choices=["similarity_only", "domain_only", "both"],
                       help="Type of filtering to apply")
    parser.add_argument("--output_dir_path", type=str, default="./ablation_filter_module_results/")
    parser.add_argument("--errors_dir_path", type=str, default="./ablation_filter_module_errors/")
    
    return parser.parse_args()

def get_evidences_by_filter_type(
    image_evidences_module: ImageEvidencesModule, 
    text_evidences_module: TextEvidencesModule, 
    idx: int, 
    query: str, 
    image_base64: str, 
    rank: int, 
    filter_type: str
) -> List:
    """
    Get evidences based on filter type and rank
    
    Args:
        image_evidences_module: Module to get image evidences
        text_evidences_module: Module to get text evidences
        idx: Index of the data point
        query: Query text (caption)
        image_base64: Base64 encoded image
        rank: Maximum number of evidences to retrieve
        filter_type: Type of filtering to apply ('similarity_only', 'domain_only', 'both')
        
    Returns:
        List of evidences
    """
    # Set parameters based on filter type
    if filter_type == "similarity_only":
        # Only use similarity threshold, no domain filtering
        image_evidences = image_evidences_module.get_evidence_by_index(
            idx, 
            query=query, 
            threshold=0.0, 
            reference_image=image_base64, 
            image_similarity_threshold=0.7, 
            max_results=rank, 
            min_results=0, 
            use_filter_by_domain=False,
            use_filter_by_excluding_domains=False,
            use_filter_by_unique_domain_title=False
        )
        
        text_evidences = text_evidences_module.get_evidence_by_index(
            idx, 
            query=query, 
            threshold=0.7, 
            reference_image=image_base64, 
            image_similarity_threshold=0.0, 
            max_results=rank, 
            min_results=0,
            use_filter_by_domain=False,
            use_filter_by_excluding_domains=False,
            use_filter_by_unique_domain_title=False,
            sort_by_text_score=False
        )
        
    elif filter_type == "domain_only":
        # Only use domain filtering, no similarity threshold
        image_evidences = image_evidences_module.get_evidence_by_index(
            idx, 
            query=None,  # No query filtering
            threshold=0.0, 
            reference_image=None,  # No reference image
            image_similarity_threshold=0.0, 
            max_results=rank, 
            min_results=0, 
            use_filter_by_domain=True,
            use_filter_by_excluding_domains=True,
            use_filter_by_unique_domain_title=True
        )
        
        text_evidences = text_evidences_module.get_evidence_by_index(
            idx, 
            query=None,  # No query filtering
            threshold=0.0, 
            reference_image=None,  # No reference image
            image_similarity_threshold=0.0, 
            max_results=rank, 
            min_results=0,
            use_filter_by_domain=True,
            use_filter_by_excluding_domains=True,
            use_filter_by_unique_domain_title=True,
            sort_by_text_score=False
        )
        
    else:  # both
        # Use both similarity and domain filtering
        image_evidences = image_evidences_module.get_evidence_by_index(
            idx, 
            query=query, 
            threshold=0.0, 
            reference_image=image_base64, 
            image_similarity_threshold=0.7, 
            max_results=rank, 
            min_results=0, 
            use_filter_by_domain=True,
            use_filter_by_excluding_domains=True,
            use_filter_by_unique_domain_title=True
        )
        
        text_evidences = text_evidences_module.get_evidence_by_index(
            idx, 
            query=query, 
            threshold=0.7, 
            reference_image=image_base64, 
            image_similarity_threshold=0.0, 
            max_results=rank, 
            min_results=0,
            use_filter_by_domain=True,
            use_filter_by_excluding_domains=True,
            use_filter_by_unique_domain_title=True,
            sort_by_text_score=False
        )
    
    # Return combined evidences, limited by rank
    all_evidences = image_evidences + text_evidences
    # # Sort by combined similarity (text + image similarity)
    # all_evidences = sorted(all_evidences, 
    #                        key=lambda x: (x.text_similarity_score or 0) + (x.image_similarity_score or 0), 
    #                        reverse=True)
    
    # Return only top 'rank' evidences
    # return all_evidences[:rank]
    return all_evidences

def inference_advanced_ablation(
    entities_module: EntitiesModule,
    image_evidences_module: ImageEvidencesModule, 
    text_evidences_module: TextEvidencesModule,
    llm_connector: Union[GPTConnector, GeminiConnector],
    vlm_connector: Optional[Union[GPTConnector, GeminiConnector]],
    data: dict,
    idx: int,
    context_dir_path: str,
    rank: int,
    filter_type: str
):
    """Run inference with the specified ablation parameters"""
    start_time = time.time()
    image_base64 = data["image_base64"]
    
    visual_entities = image_evidences_module.get_entities_by_index(idx)
    
    # Get evidences according to ablation parameters
    evidences = get_evidences_by_filter_type(
        image_evidences_module=image_evidences_module,
        text_evidences_module=text_evidences_module,
        idx=idx,
        query=data["caption"],
        image_base64=image_base64,
        rank=rank,
        filter_type=filter_type
    )
    
    print(f"Number of evidences: {len(evidences)}")
    
    check_info = {
        "entities": visual_entities,
        "ablation_params": {
            "rank": rank,
            "filter_type": filter_type
        }
    }
    
    if len(evidences) > 0:
        # Process with available evidences
        visual_prompt = get_visual_prompt(
            caption=data["caption"],
            content=data["content"],
            visual_entities=visual_entities,
            visual_candidates=evidences,
            pre_check=True
        )
        visual_check_result = llm_connector.call_with_structured_output(
            prompt=visual_prompt,
            schema=VISUAL_RESPONSE_SCHEMA,
        )
        check_info['evidences'] = [ev.to_dict() for ev in evidences]        
        check_info['result'] = visual_check_result
        check_info["check_type"] = "high quality evidences"
    else:
        # Fallback to context-based checking if no evidences available
        if os.path.exists(os.path.join(context_dir_path, f"{idx}.json")):
            with open(os.path.join(context_dir_path, f"{idx}.json"), "r") as f:
                context_result = json.load(f)
        else:
            # Generate context from image
            print("Generating Context")
            context_prompt = get_context_prompt(
                entities=visual_entities, 
                caption=data["caption"], 
                news_content=data["content"]
            )
            context_result = vlm_connector.call_with_structured_output(
                prompt=context_prompt,
                schema=CONTEXT_RESPONSE_SCHEMA,
                image_base64=image_base64,
                system_prompt=SYSTEM_PROMPT_FOR_VLM_GENERATED_CONTEXT
            )
            print("Generated Context")
            
            # Save the context result to a file
            os.makedirs(context_dir_path, exist_ok=True)
            file_path = os.path.join(context_dir_path, f"{idx}.json")
            with open(file_path, "w") as f:
                json.dump(context_result, f, indent=2, ensure_ascii=False)
                
        caption_context_checking_prompt = get_caption_context_checking_prompt(
            caption=data["caption"],
            context=context_result
        )
        
        visual_check_result = llm_connector.call_with_structured_output(
            prompt=caption_context_checking_prompt,
            schema=CAPTION_CONTEXT_CHECKING_RESPONSE_SCHEMA, 
            system_prompt=SYSTEM_PROMPT_FOR_CAPTION_CONTEXT_CHECKING
        )
        
        check_info["evidences"] = []
        check_info["result"] = visual_check_result
        check_info["context"] = context_result
        check_info["check_type"] = "context"
    
    # Final checking
    final_prompt = get_final_prompt(
        caption=data["caption"],
        content=data["content"],
        visual_check_result=visual_check_result,
    )
    final_result = vlm_connector.call_with_structured_output(
        prompt=final_prompt,
        schema=FINAL_RESPONSE_SCHEMA, 
        image_base64=image_base64
    )
    
    inference_time = time.time() - start_time
    
    result = {
        "caption": data["caption"],
        "ground_truth": data["label"],
        "check_result": check_info,
        "final_result": final_result,
        "inference_time": float(inference_time),
        "ablation_params": {
            "rank": rank,
            "filter_type": filter_type
        }
    }
    
    return process_results(result)

def main():
    args = arg_parser()
    
    # Setup environment
    load_dotenv()
    
    # Create ablation type identifier
    ablation_id = f"rank_{args.rank}_{args.filter_type}"
    
    # Create output directories
    output_dir = os.path.join(args.output_dir_path, ablation_id)
    errors_dir = os.path.join(args.errors_dir_path, ablation_id)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(errors_dir):
        os.makedirs(errors_dir)
    
    # Connect to LLM and VLM models
    print(f"Connecting to LLM Model: {args.llm_model}")
    if args.llm_model == "gpt":
        llm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.llm_model == "gemini":
        llm_connector = GeminiConnector(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.0-flash-001"
        )
    else:
        raise ValueError(f"Invalid LLM model: {args.llm_model}")
    
    print(f"Connecting to VLM Model: {args.vlm_model}")
    if args.vlm_model == "gpt":
        vlm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.vlm_model == "gemini":
        vlm_connector = GeminiConnector(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.0-flash-001"
        )
    else:
        raise ValueError(f"Invalid VLM model: {args.vlm_model}")
    
    # Initialize modules
    print("Loading evidence modules...")
    entities_module = EntitiesModule(args.entities_path)
    image_evidences_module = ImageEvidencesModule(args.image_evidences_path)
    text_evidences_module = TextEvidencesModule(args.text_evidences_path)
    
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
        
    # Filter indices to be within range
    indices = [idx for idx in random_index if start_idx <= idx <= end_idx]
    
    # Validate indices
    if start_idx >= len(dataset):
        raise ValueError(f"Start index {start_idx} is out of range for dataset of length {len(dataset)}")
    if end_idx >= len(dataset):
        end_idx = len(dataset) - 1
    if start_idx > end_idx:
        raise ValueError(f"Start index {start_idx} is greater than end index {end_idx}")
    
    print(f"Running ablation study: rank={args.rank}, filter_type={args.filter_type}")
    print(f"Processing {len(indices)} items from index {start_idx} to {end_idx}")

    # Process each item
    results = []
    error_items = []
    total_start_time = time.time()

    for idx in indices:
        try:
            print(f"Processing item {idx}")
            res_path = os.path.join(output_dir, f"result_{idx}.json")
            
            if args.skip_existing and os.path.exists(res_path):
                print(f"Skipping existing result for item {idx}")
                continue
            
            item = dataset[idx]
            
            result = inference_advanced_ablation(
                entities_module=entities_module,
                image_evidences_module=image_evidences_module,
                text_evidences_module=text_evidences_module,
                llm_connector=llm_connector,
                vlm_connector=vlm_connector,
                data=item,
                idx=idx,
                context_dir_path=args.context_dir_path,
                rank=args.rank,
                filter_type=args.filter_type
            )
            
            with open(res_path, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            results.append(result)
            
        except KeyError as e:
            print(f"KeyError processing item {idx}: {e}")
            continue
        except openai.BadRequestError as e:
            print(f"BadRequestError processing item {idx}: {e}")
            continue
        except json.decoder.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            continue
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
            continue
        except Exception as e:
            error_path = os.path.join(errors_dir, f"error_{idx}.json")
            with open(error_path, "w") as f:
                error_item = {
                    "error": str(e),
                    "ablation_params": {
                        "rank": args.rank,
                        "filter_type": args.filter_type
                    }
                }
                json.dump(error_item, f, indent=2, ensure_ascii=False)
            
            error_items.append(error_item)
            print(f"Error processing item {idx}: {e}")
                
    total_time = time.time() - total_start_time
    print(f"Ablation study rank={args.rank}, filter_type={args.filter_type} completed in {total_time:.2f} seconds")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()