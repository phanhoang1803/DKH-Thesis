# search_engine/google_text_search.py

import math
from typing import Any, Dict, List
import requests
import os
from dotenv import load_dotenv

class GoogleTextSearch:
    def __init__(self, api_key, cx, news_sites=None):
        """
        Initialize the GoogleTextSearch class.
        
        Args:
            api_key (str): Google API key
            cx (str): Custom search engine ID
            news_sites (List[str]): List of news sites to filter results by. If None, all news sites are included.
        """
        self.api_key = api_key
        self.cx = cx

        # Default US news websites if none are provided
        self.news_sites = news_sites
        self.search_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 10, **kwargs) -> List[Dict[Any, Any]]:
        """
        Alternative implementation using requests library instead of google-api-python-client.
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return (max 100)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            List[Dict]: List of search result items
        """
        if num_results > 100:
            raise ValueError('Google Custom Search API supports max of 100 results')
            
        calls_needed = math.ceil(num_results / 10)
        all_results = []
        start_index = 1
        
        for call in range(calls_needed):
            # Calculate items for this call
            items_this_call = min(10, num_results - len(all_results))
            
            params = {
                'key': self.api_key,
                'cx': self.cx,
                'q': query,
                'num': items_this_call,
                'start': start_index,
                **kwargs
            }
            
            response = requests.get(self.search_url, params=params)
            
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code}, {response.text}")
            
            data = response.json()
            if 'items' in data:
                all_results.extend(data['items'])
            
            start_index += items_this_call
            
        return all_results
    
    def _filter_news_sites(self, search_results):
        if self.news_sites is None or len(self.news_sites) == 0:
            return search_results.get('items', [])
        
        filtered_items = []
        
        for item in search_results.get('items', []):
            url = item.get('link')
            if any(news_site in url for news_site in self.news_sites):
                filtered_items.append(item)
        
        return filtered_items

# Example Usage:
if __name__ == "__main__":
    load_dotenv()

    # Load API key and CX from env
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")

    # Instantiate the GoogleTextSearch class
    google_news = GoogleTextSearch(api_key=API_KEY, 
                                   cx=CX, 
                                   news_sites= ["nytimes.com", "cnn.com", "washingtonpost.com", "foxnews.com", "usatoday.com", "wsj.com"])

    # Search for a query
    query = "latest US election news"
    
    # Get filtered news URLs
    filtered_news = google_news.search(query, num_results=10)
    
    # Output the results
    print("Filtered News:")
    for news in filtered_news:
        print(news)
