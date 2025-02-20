# inference_use_retrieved_evidences.py

from datetime import datetime
import numpy as np
from modules import NERConnector, EntitiesModule, BLIP2Connector, GPTConnector, GeminiConnector, ExternalRetrievalModule
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
from src.utils.utils import process_results, NumpyJSONEncoder, EvidenceCache
from src.modules.reasoning_module.connector.gpt import INTERNAL_RESPONSE_SCHEMA, EXTERNAL_RESPONSE_SCHEMA, FINAL_RESPONSE_SCHEMA

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_dataset", 
                       help="")
    parser.add_argument("--internal_path", type=str, default="queries_dataset/links_test.json", 
                        help="")
    parser.add_argument("--external_path", type=str, default="queries_dataset")
    parser.add_argument("--llm_model", type=str, default="gpt", choices=["gpt", "gemini"])
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=0)
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
             external_retrieval_module: ExternalRetrievalModule,
             evidence_cache: EvidenceCache,
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
    if evidence_cache.get(data["caption"]):
        candidates = evidence_cache.get(data["caption"]).evidences
    else:
        candidates = external_retrieval_module.retrieve(data["caption"], num_results=10, threshold=0.7, news_factcheck_ratio=0.5, min_result_number=0)
    
    # 1: Internal Checking
    internal_prompt = get_internal_prompt(
        caption=data["caption"],
        textual_entities=textual_entities
    )
    internal_result = llm_connector.call_with_structured_output(
        prompt=internal_prompt,
        schema=InternalResponse if isinstance(llm_connector, GeminiConnector) else INTERNAL_RESPONSE_SCHEMA,
        image_base64=image_base64
    )
    
    # 2: External Checking
    external_prompt = get_external_prompt(
        caption=data["caption"],
        candidates=candidates
    )
    external_result = llm_connector.call_with_structured_output(
        prompt=external_prompt,
        schema=ExternalResponse if isinstance(llm_connector, GeminiConnector) else EXTERNAL_RESPONSE_SCHEMA
    )

    # 3: Final Checking
    final_prompt = get_final_prompt(
        caption=data["caption"],
        internal_result=internal_result,
        external_result=external_result,
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
            "textual_entities": textual_entities,
            "result": internal_result
        },
        "external_check": {
            "candidates": [candidate.to_dict() if isinstance(candidate, Article) else candidate for candidate in candidates],
            "result": external_result
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
    
    # Initialize evidence cache
    evidence_cache = EvidenceCache(args.external_cache_path)

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
    
    print("Connecting to LLM Model...")
    if args.llm_model == "gpt":
        llm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.llm_model == "gemini":
        llm_connector = GeminiConnector(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-1.5-flash-latest"
        )
    else:
        raise ValueError(f"Invalid LLM model: {args.llm_model}")
    print("LLM Model Connected")
    
    # Initialize external retrieval module
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

    for batch in dataloader:
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
                    evidence_cache,
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