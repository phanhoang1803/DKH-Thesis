# cosmos_exmodule_script.py

import argparse
from src.modules.parallel_external_retrieval_module import ParallelExternalRetrievalModule
from dotenv import load_dotenv
import os
from src.dataloaders.cosmos_dataloader import get_cosmos_dataloader

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    
    load_dotenv()
    
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")
    
    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")
    
    
    retriever = ParallelExternalRetrievalModule(API_KEY, CX)
    
    dataloader = get_cosmos_dataloader(args.data_path)
    
    for batch in dataloader:
        print(batch)
        
        queries = []
        for item in batch:
            caption = item['caption']
            if caption and isinstance(caption, str):
                queries.append({"text_query": caption.strip()})
            else:
                print(f"Warning: Skipping invalid caption: {caption}")
        
        if not queries:
            print("Warning: No valid queries in batch, skipping...")
            continue
        
        articles = retriever.batch_retrieve(queries, num_results=10, threshold=0.8)
        
        print(articles)
        break
    

