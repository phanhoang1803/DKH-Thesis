# search_engine/google_text_search.py

import requests
import os
from dotenv import load_dotenv

class GoogleTextSearch:
    def __init__(self, api_key, cx, news_sites=None):
        self.api_key = api_key
        self.cx = cx

        # Default US news websites if none are provided
        self.news_sites = news_sites
        self.search_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query, num_results=10):
        results = self._search(query, num_results)
        return self._filter_news_sites(results)

    def _search(self, query, num_results=10):
        params = {
            'key': self.api_key,
            'cx': self.cx,
            'q': query,
            'num': num_results
        }
        
        response = requests.get(self.search_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code}, {response.text}")
        
        return response.json()
    
    def _filter_news_sites(self, search_results):
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
