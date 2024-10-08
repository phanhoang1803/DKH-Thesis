import requests
import os

class GoogleNewsSearch:
    def __init__(self, api_key, cx, news_sites=None):
        """
        Initialize the GoogleNewsSearch class with the API key and CX (Custom Search Engine ID).
        
        Parameters:
        api_key (str): The Google Custom Search API Key.
        cx (str): The Custom Search Engine ID.
        news_sites (list): A list of specific news websites to filter (optional).
        """
        self.api_key = api_key
        self.cx = cx
        # Default US news websites if none are provided
        self.news_sites = news_sites or [
            "nytimes.com", "cnn.com", "washingtonpost.com", 
            "foxnews.com", "usatoday.com", "wsj.com"
        ]
        self.search_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query, num_results=10):
        """
        Perform a Google search using the Custom Search JSON API.
        
        Parameters:
        query (str): The search query.
        num_results (int): The number of search results to retrieve.
        
        Returns:
        dict: The raw JSON response from the Google API.
        """
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
    
    def filter_news_sites(self, search_results):
        """
        Filter the search results to include only specific news websites.
        
        Parameters:
        search_results (dict): The raw search results from Google API.
        
        Returns:
        list: A list of URLs filtered to match the specified news websites.
        """
        filtered_urls = []
        
        for item in search_results.get('items', []):
            url = item.get('link')
            if any(news_site in url for news_site in self.news_sites):
                filtered_urls.append(url)
        
        return filtered_urls

    def search_news(self, query, num_results=10):
        """
        Perform a news search and filter results based on the defined news websites.
        
        Parameters:
        query (str): The search query.
        num_results (int): The number of search results to retrieve.
        
        Returns:
        list: A list of filtered news URLs from the search results.
        """
        results = self.search(query, num_results)
        return self.filter_news_sites(results)

# Example Usage:
if __name__ == "__main__":
    # Replace with your actual API_KEY and CX
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    # Instantiate the GoogleNewsSearch class
    google_news = GoogleNewsSearch(api_key=API_KEY, cx=CX)

    # Search for a query
    query = "latest US election news"
    
    # Get filtered news URLs
    filtered_news = google_news.search_news(query, num_results=10)
    
    # Output the results
    print("Filtered News URLs:")
    for url in filtered_news:
        print(url)
