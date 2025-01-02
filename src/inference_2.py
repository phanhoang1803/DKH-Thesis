from datetime import datetime
import numpy as np
from modules import NERConnector, BLIP2Connector, GeminiConnector, ExternalRetrievalModule
from dataloaders import cosmos_dataloader
from src.modules.evidence_retrieval_module.scraper.scraper import Article
from templates_2 import get_internal_prompt, get_final_prompt, get_external_prompt
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

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/public_test_acm.json", 
                       help="Path to the json file. The json file should in the same directory as dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_dir_path", type=str, default="./result/")
    parser.add_argument("--errors_dir_path", type=str, default="./errors/")
    
    # Dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_shuffle", action='store_false')
    parser.add_argument("--num_workers", type=int, default=2)
    
    # Model configs
    parser.add_argument("--ner_model", type=str, default="dslim/bert-large-NER")
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    
    return parser.parse_args()

def inference(ner_connector: NERConnector, 
             blip2_connector: BLIP2Connector, 
             llm_connector: GeminiConnector, 
             external_retrieval_module: ExternalRetrievalModule, 
             data: dict):
    
    class InternalResponse(TypedDict, total=True):
        verdict: bool
        explanation: str
        confidence_score: int

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
    textual_entities = ner_connector.extract_text_entities(data["caption"])
    textual_entities = [{"entity": item["entity"], "word": item["word"]} for item in textual_entities]

    image_base64 = data["image_base64"]

    # Get external evidence
    candidates = external_retrieval_module.retrieve(data["caption"], num_results=10, threshold=0.7, news_factcheck_ratio=0.5, min_result_number=0)
    
    # 1: Internal Checking
    internal_prompt = get_internal_prompt(
        caption=data["caption"],
        textual_entities=textual_entities
    )
    internal_result = llm_connector.call_with_structured_output(
        prompt=internal_prompt,
        schema=InternalResponse,
        image_base64=image_base64
    )
    
    # 2: External Checking
    external_prompt = get_external_prompt(
        caption=data["caption"],
        candidates=candidates
    )
    external_result = llm_connector.call_with_structured_output(
        prompt=external_prompt,
        schema=ExternalResponse
    )

    # 3: Final Checking
    final_prompt = get_final_prompt(
        caption=data["caption"],
        internal_result=internal_result,
        external_result=external_result,
    )
    final_result = llm_connector.call_with_structured_output(
        prompt=final_prompt,
        schema=FinalResponse,
        image_base64=image_base64
    )
    
    inference_time = time.time() - start_time
    
    result = {
        "caption": data["caption"],
        "ground_truth": data["label"],
        "internal_check": {
            "textual_entities": textual_entities,
            "result": internal_result
        },
        "external_check": {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "result": external_result
        },
        "final_result": final_result,
        "inference_time": float(inference_time)
    }
    
    return process_results(result)

def get_transform():
    return None

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Article):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def process_results(results_dict):
    """
    Recursively process dictionary to convert all NumPy types and Article objects to JSON-serializable types.
    
    Args:
        results_dict (dict): Dictionary containing results
        
    Returns:
        dict: Processed dictionary with all values converted to JSON-serializable types
    """
    try:
        if isinstance(results_dict, dict):
            return {key: process_results(value) for key, value in results_dict.items()}
        elif isinstance(results_dict, (list, tuple)):
            return [process_results(item) for item in results_dict]
        elif isinstance(results_dict, np.integer):
            return int(results_dict)
        elif isinstance(results_dict, np.floating):
            return float(results_dict)
        elif isinstance(results_dict, np.ndarray):
            return results_dict.tolist()
        elif isinstance(results_dict, Article):
            return results_dict.to_dict()
        elif isinstance(results_dict, datetime):
            return results_dict.isoformat()
        return results_dict
    except Exception as e:
        print(f"Error processing value {type(results_dict)}: {str(e)}")
        return str(results_dict)

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

    # Initialize dataloader
    dataloader = cosmos_dataloader.get_cosmos_dataloader(
        args.data_path, 
        args.batch_size, 
        args.no_shuffle, 
        args.num_workers,
        transform=get_transform()
    )
    
    # Initialize models
    ner_connector = NERConnector(
        model_name=args.ner_model,
        tokenizer_name=args.ner_model,
        device=args.device
    )
    ner_connector.connect()
    
    llm_connector = GeminiConnector(
        api_key=os.environ["GEMINI_API_KEY"],
    )
    
    # print("Connecting to External Retrieval Module...")
    external_retrieval_module = ExternalRetrievalModule(
        text_api_key=os.environ["GOOGLE_API_KEY"],
        cx=os.environ["CX"],
        news_sites=NEWS_SITES,
        fact_checking_sites=FACT_CHECKING_SITES
    )
    
    # Process data and save results
    results = []
    error_items = []
    total_start_time = time.time()
    count = 0

    for batch_idx, batch in enumerate(dataloader):
        for item in batch:
            count += 1
            
            if count <= args.start_idx:
                continue
            
            try:
                result = inference(
                    ner_connector,
                    None,
                    llm_connector,
                    external_retrieval_module,
                    item
                )
                
                with open(os.path.join(args.output_dir_path, f"result_{count}.json"), "w", encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                results.append(result)
            except Exception as e:
                with open(os.path.join(args.errors_dir_path, f"error_{count}.json"), "w") as f:
                    error_item = {
                        "error": str(e),
                        "caption": item["caption"],
                    }
                    json.dump(error_item, f, indent=2, ensure_ascii=False)
                error_items.append(error_item)
                print(f"Error processing item {count}: {e}")
                # raise Exception(e)
    
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