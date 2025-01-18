# src/prerun_external_module_script.py

# This script is used to run ExternalRetrievalModule to get evidences of NewsClipping test set

from datetime import datetime
import argparse
import json
import time
import os
from typing import Dict, List, Any
import numpy as np
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from src.modules.evidence_retrieval_module import ExternalRetrievalModule
from src.config import NEWS_SITES, FACT_CHECKING_SITES
from src.modules.evidence_retrieval_module.scraper.scraper import Article
from src.dataloaders.newsclipping_dataloader import get_newsclipping_dataloader
from src.utils.utils import NumpyJSONEncoder, process_results, EvidenceCache, ExternalEvidence
from multiprocessing import cpu_count

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/short_newsclipping_test.json",
                       help="Path to the json file containing captions")
    parser.add_argument("--output_path", type=str, default="./cache/external_evidence_cache_newsclippingtest.json",
                       help="Path to save the external evidence results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_shuffle", action='store_false')
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--start_idx", type=int, default=0)
    return parser.parse_args()

def main():
    args = arg_parser()
    load_dotenv()

    # Initialize cache with saving every 50 items
    evidence_cache = EvidenceCache(args.output_path, save_frequency=1)
    
    # Initialize External Retrieval Module
    external_retrieval_module = ExternalRetrievalModule(
        text_api_key=os.environ["GOOGLE_API_KEY"],
        cx=os.environ["CX"],
        news_sites=NEWS_SITES,
        fact_checking_sites=FACT_CHECKING_SITES
    )

    # Initialize dataloader
    dataloader = get_newsclipping_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=None
    )

    count = 0
    for batch_idx, batch in enumerate(dataloader):
        for item in batch:
            count += 1
            
            if count <= args.start_idx:
                continue

            caption = item["caption"]
            
            # Check cache using the new cache class
            if evidence_cache.get(caption):
                print(f"Skipping caption {count} (already in cache)")
                continue

            try:
                print(f"Processing caption {count}: {caption[:100]}...")
                start_time = time.time()
                
                # Get external evidence
                candidates = external_retrieval_module.retrieve(
                    caption,
                    num_results=10,
                    threshold=0.7,
                    news_factcheck_ratio=0.5,
                    min_result_number=0
                )
                
                inference_time = time.time() - start_time
                
                # Create evidence object with hash
                evidence = ExternalEvidence(
                    query=caption,
                    query_hash=evidence_cache._generate_hash(caption),
                    evidences=[candidate.to_dict() for candidate in candidates],
                    inference_time=inference_time,
                    timestamp=datetime.now().isoformat()
                )
                
                # Add to cache (saving handled by cache class)
                evidence_cache.add(evidence)
                
                print(f"Successfully processed caption {count}")
                
            except Exception as e:
                print(f"Error processing caption {count}: {str(e)}")
                continue

    # Force final save
    evidence_cache.save_if_needed(force=True)
    print(f"Completed processing {len(evidence_cache.cache)} captions")

if __name__ == "__main__":
    main()