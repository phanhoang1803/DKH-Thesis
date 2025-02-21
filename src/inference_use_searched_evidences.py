# inference_use_retrieved_evidences.py

from datetime import datetime
import numpy as np
from modules import NERConnector, EntitiesModule, BLIP2Connector, GPTConnector, GeminiConnector, ExternalRetrievalModule, EvidencesModule, Evidence, GPTVisionConnector, GeminiVisionConnector
from dataloaders import cosmos_dataloader
from datasets.newsclipping_datasets import MergedBalancedNewsClippingDataset
from src.modules.evidence_retrieval_module.scraper.scraper import Article
from templates import get_internal_prompt, get_final_prompt, get_external_prompt, get_vision_prompt
import os
from dotenv import load_dotenv
import argparse
from huggingface_hub import login
from torchvision import transforms
from typing_extensions import TypedDict
import torch
import json
import time
from src.config import NEWS_SITES, FACT_CHECKING_SITES
from src.utils.utils import process_results, NumpyJSONEncoder, EvidenceCache
from src.modules.reasoning_module.connector.gpt import INTERNAL_RESPONSE_SCHEMA, EXTERNAL_RESPONSE_SCHEMA, FINAL_RESPONSE_SCHEMA
from src.modules.reasoning_module.connector.gpt_vision import VISION_FINAL_SCHEMA

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_dataset", 
                       help="")
    parser.add_argument("--internal_path", type=str, default="test_dataset/links_test.json", 
                        help="")
    parser.add_argument("--external_path", type=str, default="queries_dataset\merged_balanced\direct_search/test/test.json")
    parser.add_argument("--llm_model", type=str, default="gemini", choices=["gpt", "gemini"])
    parser.add_argument("--vision_model", type=str, default="gpt", choices=["gpt", "gemini"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--output_dir_path", type=str, default="./result/")
    parser.add_argument("--errors_dir_path", type=str, default="./errors/")
    
    # Dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_shuffle", action='store_false')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    
    # Model configs
    parser.add_argument("--ner_model", type=str, default="dslim/bert-large-NER")
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip2-opt-2.7b")
    # parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    
    return parser.parse_args()

def inference(entities_module: EntitiesModule, 
             llm_connector: GPTConnector,
             vision_connector: GPTVisionConnector, 
             evidences_module: EvidencesModule,
             data: dict,
             idx: int):
    
    class InternalResponse(TypedDict, total=True):
        verdict: bool
        explanation: str
        confidence_score: int

    class VisionResponse(TypedDict, total=True):
        verdict: bool
        alignment_score: int  # 0-100
        confidence_score: int  # 0-10
        explanation: str  # up to 1000 words
        key_observations: str

    class ExternalResponse(TypedDict, total=True):
        verdict: bool
        explanation: str
        confidence_score: int
        supporting_points: str

    class FinalResponse(TypedDict, total=True):
        OOC: bool
        validation_summary: str
        explanation: str
        confidence_score: int

    start_time = time.time()
    
    # Extract text entities
    visual_entities = entities_module.get_entities_by_index(idx)
    
    image_base64 = data["image_base64"]

    # Get external evidence
    web_results = evidences_module.get_evidence_by_index(idx)
    
    # Get extracted reference images from web results
    reference_images = [result.image_data for result in web_results if result.image_data]
    
    # 1: Internal Checking
    internal_prompt = get_internal_prompt(
        caption=data["caption"],
        content=data["content"],
        visual_entities=visual_entities
    )
    internal_result = llm_connector.call_with_structured_output(
        prompt=internal_prompt,
        schema=InternalResponse if isinstance(llm_connector, GeminiConnector) else INTERNAL_RESPONSE_SCHEMA,
        image_base64=image_base64
    )
    
    # 2: External Checking
    vision_result = None
    vision_prompt = get_vision_prompt()
    if reference_images:
        vision_result = vision_connector.call_with_structured_output(
            prompt=vision_prompt,
            schema=VisionResponse if isinstance(vision_connector, GeminiVisionConnector) else VISION_FINAL_SCHEMA,
            # schema=VISION_FINAL_SCHEMA,
            image_base64=image_base64,
            ref_images_base64=reference_images
        )
    
    # external_prompt = get_external_prompt(
    #     caption=data["caption"],
    #     web_results=web_results,
    #     vision_result=vision_result
    # )
    # external_result = llm_connector.call_with_structured_output(
    #     prompt=external_prompt,
    #     schema=ExternalResponse if isinstance(llm_connector, GeminiConnector) else EXTERNAL_RESPONSE_SCHEMA
    # )

    # 3: Final Checking
    final_prompt = get_final_prompt(
        caption=data["caption"],
        internal_result=internal_result,
        external_result=vision_result,
    )
    final_result = llm_connector.call_with_structured_output(
        prompt=final_prompt,
        schema=FinalResponse if isinstance(llm_connector, GeminiConnector) else FINAL_RESPONSE_SCHEMA, 
        image_base64=image_base64
    )
    
    inference_time = time.time() - start_time
    
    result = {
        "caption": data["caption"],
        "ground_truth": data["label"],
        "internal_check": {
            "visual_entities": visual_entities,
            "result": internal_result
        },
        "external_check": {
            "web_results": [web_result.to_dict() for web_result in web_results],
            "result": vision_result
        },
        "final_result": final_result,
        "inference_time": float(inference_time)
    }
    
    return process_results(result)

def get_transform():
    return None

def main():
    args = arg_parser()
    
    # Setup environment
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])
    
    # Make res folder
    if not os.path.exists(args.output_dir_path):
        os.makedirs(args.output_dir_path)
    if not os.path.exists(args.errors_dir_path):
        os.makedirs(args.errors_dir_path)
    
    # Initialize models
    entities_module = EntitiesModule(args.internal_path)
    
    print("Connecting to LLM Model...")
    if args.llm_model == "gpt":
        llm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o"
        )
    elif args.llm_model == "gemini":
        llm_connector = GeminiConnector(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.0-pro-exp-02-05"
        )
    else:
        raise ValueError(f"Invalid LLM model: {args.llm_model}")
    print("LLM Model Connected")
    
    print("Connecting to Vision Model...")
    if args.vision_model == "gpt":
        vlm_connetor = GPTVisionConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o"
        )
    elif args.vision_model == "gemini":
        vlm_connetor = GeminiVisionConnector(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-1.5-flash-latest"
        )
    else:
        raise ValueError(f"Invalid Vision model: {args.vision_model}")
    print("Vision Model Connected")
        
    
    # Initialize external retrieval module
    # print("Connecting to External Retrieval Module...")
    evidences_module = EvidencesModule(args.external_path)
    
    # Process data and save results
    results = []
    error_items = []
    total_start_time = time.time()

    dataset = MergedBalancedNewsClippingDataset(args.data_path)
    
    start_idx = args.start_idx if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else len(dataset) - 1
    
    # Validate indices
    if start_idx >= len(dataset):
        raise ValueError(f"Start index {start_idx} is out of range for dataset of length {len(dataset)}")
    if end_idx >= len(dataset):
        end_idx = len(dataset) - 1
    if start_idx > end_idx:
        raise ValueError(f"Start index {start_idx} is greater than end index {end_idx}")
    
    print(f"Processing items from index {start_idx} to {end_idx}")
    for idx in range(start_idx, end_idx + 1):
        try:
            print(f"Processing item {idx}")
            item = dataset[idx]
        
            result = inference(
                entities_module=entities_module,
                llm_connector=llm_connector,
                vision_connector=vlm_connetor,
                evidences_module=evidences_module,
                data=item,
                idx=idx
            )
            
            with open(os.path.join(args.output_dir_path, f"result_{idx}.json"), "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            results.append(result)
        except Exception as e:
            with open(os.path.join(args.errors_dir_path, f"error_{idx}.json"), "w") as f:
                error_item = {
                    "error": str(e),
                    # "caption": item["caption"],
                }
                json.dump(error_item, f, indent=2, ensure_ascii=False)
            error_items.append(error_item)
            print(f"Error processing item {idx}: {e}")
            # raise Exception(e)
                
        # break
    
    total_time = time.time() - total_start_time
    
    # Add total processing time to results
    final_results = {
        "results": results,
        "total_processing_time": total_time,
        "average_inference_time": total_time / len(results)
    }
    
    # Save results
    with open(os.path.join(args.output_dir_path, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2, cls=NumpyJSONEncoder, ensure_ascii=False)
    with open(os.path.join(args.errors_dir_path, "error_items.json"), "w") as f:
        json.dump(error_items, f, indent=2, cls=NumpyJSONEncoder, ensure_ascii=False)

if __name__ == "__main__":
    main()