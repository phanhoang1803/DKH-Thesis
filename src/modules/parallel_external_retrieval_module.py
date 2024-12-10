from concurrent.futures import ThreadPoolExecutor, as_completed
from ..scraper.scraper import Scraper, Article
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Union, Tuple
import numpy as np
from dotenv import load_dotenv
import os
from functools import lru_cache
import time
import random

class ParallelExternalRetrievalModule:
    def __init__(self, text_api_key: str, cx: str, news_sites: List[str] = None, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_workers: int = 4,
                 cache_size: int = 1000,
                 retry_delay: float = 1.0,
                 max_retries: int = 3):
        """
        Initialize the Optimized External Retrieval Module with retry logic.
        
        Args:
            retry_delay (float): Delay between retries in seconds
            max_retries (int): Maximum number of retries for failed requests
        """
        self.scraper = Scraper(text_api_key, cx, news_sites)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.retry_delay = retry_delay
        self.max_retries = max_retries

    def batch_retrieve(self, queries: List[Dict[str, str]], num_results: int = 10, 
                      threshold: float = 0.5) -> Dict[str, List[Article]]:
        """Batch retrieve with rate limiting"""
        # Validate queries
        valid_queries = []
        for query in queries:
            if self._validate_query(query):
                valid_queries.append(query)
            else:
                print(f"[WARNING] Skipping invalid query: {query}")
        
        if not valid_queries:
            print("[WARNING] No valid queries to process")
            return {}

        # Process queries with rate limiting
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(valid_queries), self.max_workers):
                batch = valid_queries[i:i + self.max_workers]
                
                # Add random delay between batches to avoid rate limits
                if i > 0:
                    time.sleep(random.uniform(1.0, 2.0))
                
                future = executor.submit(self._process_query_batch, batch, num_results, threshold)
                futures.append((future, batch))

            results = {}
            for future, batch in futures:
                try:
                    batch_results = future.result()
                    results.update(batch_results)
                except Exception as e:
                    print(f"[ERROR] Batch processing failed: {str(e)}")
                    # Add empty results for failed queries
                    for query in batch:
                        query_key = query.get('text_query') or query.get('image_query') or 'unknown'
                        results[query_key] = []

        return results

    def _validate_query(self, query: Dict[str, str]) -> bool:
        """Validate query format and content"""
        if not isinstance(query, dict):
            return False
            
        text_query = query.get('text_query')
        image_query = query.get('image_query')
        
        # Check if at least one query type is present and valid
        if not text_query and not image_query:
            return False
            
        # Validate text query if present
        if text_query and not isinstance(text_query, str):
            return False
            
        # Validate image query if present
        if image_query and not isinstance(image_query, str):
            return False
            
        return True

    def _process_query_batch(self, queries: List[Dict[str, str]], num_results: int, 
                           threshold: float) -> Dict[str, List[Article]]:
        """Process a batch of queries with retry logic"""
        results = {}
        for query in queries:
            text_query = query.get('text_query')
            image_query = query.get('image_query')
            
            for retry in range(self.max_retries):
                try:
                    articles = self._retrieve_single(text_query, image_query, num_results, threshold)
                    query_key = text_query or image_query or 'unknown'
                    results[query_key] = articles
                    print(f"[INFO] Processed query: {query_key} with {len(articles)} results")
                    break
                except Exception as e:
                    if retry < self.max_retries - 1:
                        delay = self.retry_delay * (retry + 1)
                        print(f"[WARNING] Retry {retry + 1} for query after {delay}s: {str(e)}")
                        time.sleep(delay)
                    else:
                        print(f"[ERROR] Query failed after {self.max_retries} retries: {str(e)}")
                        results[text_query or image_query or 'unknown'] = []
                
        return results

    def _retrieve_single(self, text_query: Optional[str], image_query: Optional[str], 
                        num_results: int, threshold: float) -> List[Article]:
        """Process a single query with error handling"""
        if not text_query and not image_query:
            return []
            
        try:
            raw_articles = self.scraper.search_and_scrape(text_query=text_query, 
                                                         image_query=image_query, 
                                                         num_results=num_results)
        except Exception as e:
            print(f"[ERROR] Scraping failed: {str(e)}")
            return []

        if not raw_articles or not text_query:
            return [article for article in raw_articles if self.exist_image(article)]

        # Batch process all titles together with the query
        articles_with_images = [article for article in raw_articles if self.exist_image(article)]
        if not articles_with_images:
            return []

        titles = [article.title for article in articles_with_images]
        
        try:
            similarities = self._batch_similarity(text_query, titles)
        except Exception as e:
            print(f"[ERROR] Similarity calculation failed: {str(e)}")
            return articles_with_images  # Return all articles if similarity fails

        # Filter articles based on similarity scores
        filtered_articles = []
        for article, similarity in zip(articles_with_images, similarities):
            if similarity >= threshold:
                filtered_articles.append(article)
                print(f"[INFO] Article '{article.title}' matches with similarity {similarity:.3f}")
            else:
                print(f"[INFO] Article '{article.title}' below threshold with similarity {similarity:.3f}")

        return filtered_articles

    def _batch_similarity(self, query: str, texts: List[str]) -> List[float]:
        """Calculate similarities between one query and multiple texts efficiently"""
        # Get embeddings for all texts including query at once
        all_texts = [query] + texts
        embeddings = self._get_batch_embeddings(all_texts)
        
        # Split query embedding and text embeddings
        query_embedding = embeddings[0].unsqueeze(0)  # [1, dim]
        text_embeddings = embeddings[1:]  # [n, dim]
        
        # Calculate similarities all at once
        similarities = torch.mm(query_embedding, text_embeddings.t())
        return similarities[0].cpu().numpy().tolist()

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> torch.Tensor:
        """Get cached embedding for a single text"""
        return self._get_batch_embeddings([text])[0]

    def _get_batch_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for a batch of texts efficiently"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            return F.normalize(embeddings, p=2, dim=1)

    def mean_pooling(self, model_output, attention_mask):
        """Calculate mean pooling of token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def exist_image(self, article: Article) -> bool:
        """Check if article has an image"""
        return bool(article.image_url)

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")

    # Initialize with optimized settings
    retriever = ParallelExternalRetrievalModule(
        API_KEY, 
        CX,
        news_sites=["nytimes.com", "cnn.com", "washingtonpost.com", "foxnews.com"],
        max_workers=4,
        cache_size=1000
    )
    
    # Example batch search
    queries = [
        {"text_query": "Trump Names Karoline Leavitt as Press Secretary"},
        {"text_query": "Biden Infrastructure Bill Progress"},
        {"text_query": "NASA Mars Mission Updates"}
    ]
    
    results = retriever.batch_retrieve(queries, num_results=10, threshold=0.8)
    
    # Print results
    for query, articles in results.items():
        print(f"\nResults for query: {query}")
        print("-" * 50)
        for article in articles:
            print(f"Title: {article.title}")
            print(f"URL: {article.url}")
            print(f"Image URL: {article.image_url}")
            print("---")