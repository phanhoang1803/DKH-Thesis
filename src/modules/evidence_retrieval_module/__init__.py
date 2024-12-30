# evidence_retrieval_module/__init__.py

from multiprocessing import Pool
from .scraper import Scraper, Article
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
import os
from functools import lru_cache, partial
from src.utils.logger import Logger

class ExternalRetrievalModule:
    def __init__(self, text_api_key: str, cx: str, news_sites: List[str] = None, fact_checking_sites: List[str] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the External Retrieval Module.
        
        Args:
            text_api_key (str): API key for text search
            cx (str): Custom search engine ID
            news_sites (List[str]): List of news sites to scrape from
            model_name (str): Name of the pre-trained model to use for text embeddings
            
            
        Example usage:
        ```python
        retriever = ExternalRetrievalModule(API_KEY, CX)
        articles = retriever.retrieve(text_query="Trump Names Karoline Leavitt as His White House Press Secretary", num_results=10, threshold=0.8)
        ```
        """
        self.scraper = Scraper(text_api_key, cx, news_sites, fact_checking_sites)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger(name="ExternalRetrievalModule")
        self.model.to(self.device)
        
    # def retrieve(self, 
    #             text_query: Optional[str] = None, 
    #             image_query: Optional[str] = None, 
    #             num_results: int = 10, 
    #             threshold: float = 0.7,
    #             news_factcheck_ratio: float = 0.7):
    #     """
    #     Retrieve and filter articles based on text and image queries.
        
    #     Args:
    #         text_query (str, optional): Text query for article search
    #         image_query (str, optional): Image query for article search
    #         num_results (int): Number of results to retrieve. By default, 10, with settings max at 20.
    #         threshold (float): Similarity threshold for filtering articles
    #         news_factcheck_ratio (float): Ratio of news sites to fact-checking sites. Defaults to 0.7.

    #     Returns:
    #         List[Article]: List of filtered articles
    #     """
    #     assert text_query is not None or image_query is not None, "Either text_query or image_query must be provided"
        
    #     self.logger.info(f"[INFO] Retrieving articles with text query: {text_query}")
    #     raw_articles = self.scraper.search_and_scrape(text_query=text_query, 
    #                                                   image_query=image_query, 
    #                                                   num_results=num_results,
    #                                                   news_factcheck_ratio=news_factcheck_ratio)
    #     filtered_articles = []
    #     for article in raw_articles:
    #         if self.exist_image(article):
    #             if text_query:
    #                 cosine_similarity = self.cosine_similarity(text1=text_query, text2=article.title)
    #                 if cosine_similarity >= threshold:
    #                     filtered_articles.append(article)
    #                     # self.logger.info(f"Article {article.title} MATCHES text query with cosine similarity = {cosine_similarity}")
    #                 # else:
    #                     # self.logger.info(f"Article {article.title} does NOT match text query with cosine similarity = {cosine_similarity}")
    #             else:
    #                 filtered_articles.append(article)
    #         # else:
    #             # self.logger.info(f"Article {article.title} does NOT have an image")
                
    #     self.logger.info(f"[INFO] Filtered to {len(filtered_articles)} articles")
    #     return filtered_articles
    
    def process_article(self, article, text_query: Optional[str], threshold: float, exist_image_func, cosine_similarity_func) -> Tuple[bool, Optional[float]]:
        """
        Process a single article in parallel.
        
        Returns:
            Tuple[bool, Optional[float]]: (should_include, similarity_score)
        """
        if not exist_image_func(article):
            return False, None
            
        if text_query:
            similarity = cosine_similarity_func(text1=text_query, text2=article.title)
            return similarity >= threshold, similarity
        
        return True, None

    def retrieve(self, 
                text_query: Optional[str] = None, 
                image_query: Optional[str] = None, 
                num_results: int = 10, 
                threshold: float = 0.7,
                news_factcheck_ratio: float = 0.7,
                max_workers: Optional[int] = None):
        """
        Retrieve and filter articles based on text and image queries using parallel processing.
        
        Args:
            text_query (str, optional): Text query for article search
            image_query (str, optional): Image query for article search
            num_results (int): Number of results to retrieve. By default, 10, with settings max at 20.
            threshold (float): Similarity threshold for filtering articles
            news_factcheck_ratio (float): Ratio of news sites to fact-checking sites. Defaults to 0.7.
            max_workers (int, optional): Maximum number of worker processes. Defaults to CPU count - 1.

        Returns:
            List[Article]: List of filtered articles
        """
        assert text_query is not None or image_query is not None, "Either text_query or image_query must be provided"
        
        self.logger.info(f"[INFO] Retrieving articles with text query: {text_query}")
        raw_articles = self.scraper.search_and_scrape(
            text_query=text_query, 
            image_query=image_query, 
            num_results=num_results,
            news_factcheck_ratio=news_factcheck_ratio
        )
        
        if not raw_articles:
            self.logger.info("[INFO] No articles found")
            return []
            
        # Determine number of workers
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)  # Leave one CPU for system tasks
        
        # Create partial function with fixed arguments
        process_func = partial(
            self.process_article,
            text_query=text_query,
            threshold=threshold,
            exist_image_func=self.exist_image,
            cosine_similarity_func=self.cosine_similarity
        )
        
        filtered_articles = []
        filtered_scores = []
        
        # Process articles in parallel
        with Pool(processes=max_workers) as pool:
            results = pool.map(process_func, raw_articles)
            
            # Filter articles based on results
            for article, (should_include, similarity) in zip(raw_articles, results):
                if should_include:
                    filtered_articles.append(article)
                    filtered_scores.append(similarity)
                    
                    # if similarity is not None:
                    #     self.logger.debug(
                    #         f"Article '{article.title[:50]}...' MATCHES text query with "
                    #         f"cosine similarity = {similarity:.3f}"
                    #     )
        
        self.logger.info(f"[INFO] Filtered to {len(filtered_articles)} articles from {len(raw_articles)} total")
        return filtered_articles
    
    def exist_image(self, article: Article):
        """
        Check if an article has an associated image.
        """
        return article.image_url != ""
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str):
        """
        Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        Get embedding for a piece of text using the transformer model.
        
        Args:
            text (str): Input text
            
        Returns:
            torch.Tensor: Text embedding
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(text=text, 
                              padding=True, 
                              truncation=True, 
                              max_length=512, 
                              return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)


        # Use mean pooling to get text embedding
        embedding = self.mean_pooling(outputs, inputs['attention_mask'])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def cosine_similarity(self, text1: str, text2: str):
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score
        """
        if text1 is None or text2 is None:
            return 0
        
        # Get embeddings for both texts
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Normalize embeddings
        embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=1)
        embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.mm(embedding1, embedding2.transpose(0, 1))
        
        return float(similarity[0][0].cpu().numpy())

# Example usage
if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")

    from src.config import NEWS_SITES, FACT_CHECKING_SITES

    # Initialize the module
    retriever = ExternalRetrievalModule(API_KEY, CX, news_sites=NEWS_SITES, fact_checking_sites=FACT_CHECKING_SITES)
    
    # Example search
    articles = retriever.retrieve(
        text_query="Trump Names Karoline Leavitt as His White House Press Secretary",
        num_results=10,
        threshold=0.8,
        news_factcheck_ratio=0.7
    )
    
    # Print results
    for article in articles:
        print(f"Title: {article.title}")
        print(f"URL: {article.url}")
        print(f"Image URL: {article.image_url}")
        print("---")