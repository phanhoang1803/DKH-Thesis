# scraper.py

import os
from news_scraper.news_scraper import NewsPleaseScraper
from search_engine.google_image_search import GoogleImageSearch
from search_engine.google_text_search import GoogleTextSearch

from dotenv import load_dotenv

class Scraper:
    def __init__(self, text_api_key, cx, news_sites=None):
        self.google_text_search = GoogleTextSearch(api_key=text_api_key, cx=cx, news_sites=news_sites)
        self.google_image_search = GoogleImageSearch(news_sites=news_sites)
        self.news_scraper = NewsPleaseScraper()

    def search_and_scrape(self, text_query=None, image_query=None, num_results=10):
        if not text_query and not image_query:
            raise ValueError("At least one of 'text_query' or 'image_query' must be provided.")
        
        # Initialize an empty list to collect URLs from both search methods
        all_urls = []

        # Perform text search if a text query is provided
        if text_query:
            text_results = self.google_text_search.search(text_query, num_results)
            text_urls = [item['link'] for item in text_results]
            all_urls.extend(text_urls)

        # Perform image search if an image query is provided
        if image_query:
            image_results = self.google_image_search.search(image_query, num_results)
            image_urls = [item['link'] for item in image_results]
            all_urls.extend(image_urls)

        # Remove duplicate URLs, if any
        all_urls = list(set(all_urls))

        # Use NewsPleaseScraper to scrape the news content from the URLs
        scraped_articles = self.news_scraper.scrape(all_urls)
        return scraped_articles
    

if __name__ == "__main__":
    load_dotenv()

    # Load API key and CX from env
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")

    # Default news websites
    news_sites = ["nytimes.com", "cnn.com", "washingtonpost.com", "foxnews.com", "usatoday.com", "wsj.com"]

    # Instantiate the Scraper class
    scraper = Scraper(text_api_key=API_KEY, cx=CX, news_sites=news_sites)

    # Example usage with a text query
    text_query = "latest US election news"
    scraped_articles = scraper.search_and_scrape(text_query=text_query, image_query=None, num_results=10)
    print("Scraped articles from text query:")
    for article in scraped_articles:
        print(article)
