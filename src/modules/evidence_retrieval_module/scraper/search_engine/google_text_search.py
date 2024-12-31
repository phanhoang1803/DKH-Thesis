import math
from typing import Any, Dict, List, Optional
import requests
import os
from dotenv import load_dotenv

class GoogleTextSearch:
    def __init__(self, api_key: str, cx: str, news_sites: Optional[List[str]] = None, fact_checking_sites: Optional[List[str]] = None):
        """
        Initialize the GoogleTextSearch class.
        
        Args:
            api_key (str): Google API key
            cx (str): Custom search engine ID
            news_sites (List[str], optional): List of trusted news sites
            fact_checking_sites (List[str], optional): List of fact checking sites
        """
        self.api_key = api_key
        self.cx = cx
        self.news_sites = news_sites
        self.fact_checking_sites = fact_checking_sites
        self.search_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(
        self, 
        query: str, 
        num_results: int = 10,
        filter: str = '1',
        date_restrict: Optional[str] = None,
        news_factcheck_ratio: float = 0.7  # 70% news sites, 30% fact-checking by default
    ) -> List[Dict[Any, Any]]:
        """
        Perform a Google Custom Search.
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return (1-100)
            filter (str): '1' to enable duplicate filtering, '0' to disable
            date_restrict (str, optional): Date restriction (e.g., 'd[number]', 'w[number]', 'm[number]', 'y[number]')
            news_factcheck_ratio (float): Ratio of results to get from news sites vs fact-checking sites (0.7 means 70% news)
            
        Returns:
            List[Dict]: Search results
        """
        if not query or not isinstance(query, str):
            return []
            
        if not isinstance(num_results, int) or num_results < 1 or num_results > 100:
            return []
        
        
        # Calculate number of results for each type
        news_results = int(num_results * news_factcheck_ratio)
        factcheck_results = num_results - news_results
        
        all_results = []
        
        # Get news site results
        if news_results > 0 and self.news_sites:
            news_site_query = ' OR '.join(f'site:{site}' for site in self.news_sites)
            news_query = f'{query} {news_site_query}'
            all_results.extend(self._execute_search(
                news_query, 
                news_results,
                filter,
                date_restrict
            ))
            
        # Get fact-checking site results
        if factcheck_results > 0 and self.fact_checking_sites:
            factcheck_site_query = ' OR '.join(f'site:{site}' for site in self.fact_checking_sites)
            factcheck_query = f'{query} {factcheck_site_query}'
            all_results.extend(self._execute_search(
                factcheck_query,
                factcheck_results,
                filter,
                date_restrict
            ))
            
        return all_results
    
    def _execute_search(
        self,
        query: str,
        num_results: int,
        filter: str,
        date_restrict: Optional[str]
    ) -> List[Dict[Any, Any]]:
        """
        Execute a single search request, handling pagination if needed.
        """
        calls_needed = math.ceil(num_results / 10)
        results = []
        start_index = 1
        try:
            for call in range(calls_needed):
                items_this_call = min(10, num_results - len(results))
                
                params = {
                    'key': self.api_key,
                    'cx': self.cx,
                    'q': query,
                    'num': items_this_call,
                    'start': start_index,
                    'filter': filter
                }
                
                if date_restrict:
                    params['dateRestrict'] = date_restrict
                
                response = requests.get(self.search_url, params=params)
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                
                if 'items' not in data:
                    return results
                
                results.extend(data['items'])
                start_index += items_this_call
            
            return results[:num_results]
            
        except requests.RequestException as e:
            return []
        
        except Exception as e:
            print(f"Error: {e}")
            return []

if __name__ == "__main__":
    load_dotenv()
    
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")
    
    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")
    
    from src.config import NEWS_SITES, FACT_CHECKING_SITES
    
    
    google_news = GoogleTextSearch(
        api_key=API_KEY,
        cx=CX,
        news_sites=NEWS_SITES,
        fact_checking_sites=FACT_CHECKING_SITES
    )
    
    query = "A man lost his testicles while attempting to fill a scuba tank with marijuana smoke."
    results = google_news.search(
        query=query,
        num_results=10,
        filter='1', # Enable duplicate filtering
        news_factcheck_ratio=1  # 100% news sites, 0% fact-checking sites
    )
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"\nTitle: {result.get('title')}")
        print(f"URL: {result.get('link')}")
        print(f"Snippet: {result.get('snippet')}\n")