import numpy as np
from modules import NERConnector, BLIP2Connector, GeminiConnector, ExternalRetrievalModule
from dataloaders import cosmos_dataloader
from templates import get_internal_prompt, get_external_prompt, get_final_prompt
import os
from dotenv import load_dotenv
import argparse
from huggingface_hub import login
from torchvision import transforms
from typing_extensions import TypedDict
import torch
import json
import time


class FinalResponse(TypedDict):
    OOC: bool
    confidence_score: int
    validation_summary: str
    explanation: str

class InternalResponse(TypedDict):
    verdict: bool
    confidence_score: int
    explanation: str

class ExternalResponse(TypedDict):
    verdict: bool
    confidence_score: int
    explanation: str

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/public_test_acm.json", 
                       help="Path to the json file. The json file should in the same directory as dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default="results.json")
    
    # Dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=True)
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
    start_time = time.time()
    # Extract text entities
    textual_entities = ner_connector.extract_text_entities(data["content"])
    
    image_base64 = data["image_base64"]

    # Get external evidence
    candidates = external_retrieval_module.retrieve(data["caption"], num_results=10)
    # # print(candidates)

    # # 1: Internal Checking
    internal_prompt = get_internal_prompt(
        caption=data["caption"],
        textual_entities=textual_entities
    )
    # # print(internal_prompt)
    internal_result = llm_connector.call_with_structured_output(
        prompt=internal_prompt,
        schema=InternalResponse
    )
    
    # # 2: External Checking
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
        textual_entities=textual_entities,
        candidates=candidates,
        internal_result=internal_result,
        external_result=external_result,
    )
    final_result = llm_connector.call_with_structured_output(
        prompt=final_prompt,
        schema=FinalResponse,
        image_base64=image_base64
    )

    # json_output = json.loads(final_result.choices[0].text)
    
    inference_time = time.time() - start_time
    
    result = {
        "caption": data["caption"],
        "internal_check": {
            "textual_entities": textual_entities,
            "result": internal_result
        },
        "external_check": {
            "candidates": candidates,
            "result": external_result
        },
        "final_result": final_result,
        "inference_time": float(inference_time)  # Explicitly convert to Python float
    }
    
    # Process the result to ensure JSON serialization compatibility
    return process_results(result)

def get_transform():
    return None

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def process_results(results_dict):
    """
    Recursively process dictionary to convert all NumPy types to Python native types.
    
    Args:
        results_dict (dict): Dictionary containing results
        
    Returns:
        dict: Processed dictionary with all values converted to JSON-serializable types
    """
    if isinstance(results_dict, dict):
        return {key: process_results(value) for key, value in results_dict.items()}
    elif isinstance(results_dict, list):
        return [process_results(item) for item in results_dict]
    elif isinstance(results_dict, np.integer):
        return int(results_dict)
    elif isinstance(results_dict, np.floating):
        return float(results_dict)
    elif isinstance(results_dict, np.ndarray):
        return results_dict.tolist()
    return results_dict

def main():
    args = arg_parser()
    
    # Setup environment
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])
    
    # Initialize dataloader
    dataloader = cosmos_dataloader.get_cosmos_dataloader(
        args.data_path, 
        args.batch_size, 
        args.shuffle, 
        args.num_workers,
        transform=get_transform()
    )
    
    # Initialize models
    # print("Connecting to NER model...")
    ner_connector = NERConnector(
        model_name=args.ner_model,
        tokenizer_name=args.ner_model,
        device=args.device
    )
    ner_connector.connect()
    
    # print("Connecting to LLM model...")
    llm_connector = GeminiConnector(
        api_key=os.environ["GEMINI_API_KEY"],
    )
    
    # print("Connecting to External Retrieval Module...")
    external_retrieval_module = ExternalRetrievalModule(
        text_api_key=os.environ["GOOGLE_API_KEY"],
        cx=os.environ["CX"],
        news_sites=None
    )
    
    # Process data and save results
    results = []
    total_start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        for item in batch:
            result = inference(
                ner_connector,
                None,
                llm_connector,
                external_retrieval_module,
                item
            )
            results.append(result)
            
            # # print progress and current inference time
            # print(f"Processed item {len(results)}, inference time: {result['inference_time']:.2f}s")
            break
        break
    
    total_time = time.time() - total_start_time
    
    # Add total processing time to results
    final_results = {
        "results": results,
        "total_processing_time": total_time,
        "average_inference_time": total_time / len(results)
    }
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
        
    # print(f"\nTotal processing time: {total_time:.2f}s")
    # print(f"Average inference time per item: {total_time/len(results):.2f}s")

if __name__ == "__main__":
    main()