from dataclasses import asdict, dataclass
import json
import os
import numpy as np
from datetime import datetime
from src.modules.evidence_retrieval_module.scraper.scraper import Article
from typing import Dict, List, Any
from hashlib import sha256

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
    """Process dictionary to convert all NumPy types and Article objects to JSON-serializable types."""
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
    
    
@dataclass
class ExternalEvidence:
    query: str
    query_hash: str  # Added hash field
    evidences: List[Dict[str, Any]]
    inference_time: float
    timestamp: str

class EvidenceCache:
    def __init__(self, cache_path: str, save_frequency: int = 50):
        """
        Initialize the evidence cache with improved saving strategy
        
        Args:
            cache_path: Path to the cache file
            save_frequency: How often to save to disk (in number of items)
        """
        self.cache_path = cache_path
        self.save_frequency = save_frequency
        self.cache: Dict[str, ExternalEvidence] = {}
        self.items_since_save = 0
        
        # Create directory if it doesn't exist
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:  # Only create if there's actually a directory path
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create cache directory {cache_dir}: {e}")
        
        self._load_cache()

    def _generate_hash(self, text: str) -> str:
        """Generate a consistent hash for the text."""
        return sha256(text.encode('utf-8')).hexdigest()[:16]

    def _load_cache(self):
        """Load the existing cache from disk if it exists."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                existing_cache = json.load(f)
                self.cache = {
                    key: ExternalEvidence(**data)
                    for key, data in existing_cache.items()
                }

    def save_if_needed(self, force: bool = False):
        """Save cache to disk if enough items have been processed or if forced."""
        self.items_since_save += 1
        if force or self.items_since_save >= self.save_frequency:
            self._save_cache()
            self.items_since_save = 0

    def _save_cache(self):
        """Save the current cache to disk."""
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json_cache = {
                query_hash: asdict(data)
                for query_hash, data in self.cache.items()
            }
            json.dump(
                json_cache,
                f,
                indent=2,
                cls=NumpyJSONEncoder,
                ensure_ascii=False
            )

    def get(self, query: str) -> ExternalEvidence:
        """Get evidence for a query if it exists in cache."""
        query_hash = self._generate_hash(query)
        return self.cache.get(query_hash)

    def add(self, evidence: ExternalEvidence):
        """Add new evidence to the cache."""
        self.cache[evidence.query_hash] = evidence
        self.save_if_needed()