# search_engine/google_image_search.py

import requests
import dotenv

class GoogleImageSearch:
    def __init__(self, news_sites=None):
        self.news_sites=news_sites

    def search(self, query, num_results=10):
        results = self._search(query, num_results)
        return self._filter_news_sites(results)
    
    def _search(self, image, num_results=10):
        #TODO: need to implement
        return []
    
    def _filter_news_sites(self, search_results):
        filtered_items = []
        
        for item in search_results.get('items', []):
            url = item.get('link')
            if any(news_site in url for news_site in self.news_sites):
                filtered_items.append(item)
        
        return filtered_items
    
