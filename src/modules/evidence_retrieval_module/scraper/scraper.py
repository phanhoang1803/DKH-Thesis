# scraper.py
import os
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
from .news_scraper.news_scraper import NewsPleaseScraper
from .search_engine.google_image_search import GoogleImageSearch
from .search_engine.google_text_search import GoogleTextSearch
from src.utils.logger import Logger

@dataclass
class Article:
    """Represents a scraped news article."""
    authors: list = field(default_factory=list)
    date_download: datetime = field(default_factory=datetime.now)
    date_modify: datetime = field(default_factory=datetime.now)
    title: str = ""
    date: datetime = field(default_factory=datetime.now)
    content: str = ""
    url: str = ""
    description: str = ""
    image_url: str = ""
    language: str = ""
    source_domain: str = ""

class Scraper:
    def __init__(self, text_api_key, cx, news_sites=None):
        self.google_text_search = GoogleTextSearch(api_key=text_api_key, cx=cx, news_sites=news_sites)
        self.google_image_search = GoogleImageSearch(news_sites=news_sites)
        self.news_scraper = NewsPleaseScraper()
        self.logger = Logger(name="Scraper")
        
    def search_and_scrape(self, text_query=None, image_query=None, num_results=10):
        """
        Searches for news articles using a text query and/or an image query
        and scrapes the content of the articles using the NewsPleaseScraper.

        Parameters
        ----------
        text_query : str, optional
            The text query to search for. If not provided, no text search is
            performed.
        image_query : str, optional
            The image query to search for. If not provided, no image search is
            performed.
        num_results : int, optional
            The number of search results to retrieve. Defaults to 10.

        Returns
        -------
        list
            A list of `Article` objects, each representing a scraped news article.
        """
        if not text_query and not image_query:
            raise ValueError("At least one of 'text_query' or 'image_query' must be provided.")

        all_urls = []

        if text_query:
            text_results = self.google_text_search.search(text_query, num_results)
            text_urls = [item['link'] for item in text_results]
            all_urls.extend(text_urls)

        if image_query:
            image_results = self.google_image_search.search(image_query, num_results)
            image_urls = [item['link'] for item in image_results]
            all_urls.extend(image_urls)

        self.logger.info(f"[INFO] Retrieved {len(all_urls)} URLs from search engine to scrape")
        all_urls = list(set(all_urls))
        
        scraped_articles = self.news_scraper.scrape(all_urls)
        self.logger.info(f"[INFO] Scraped {len(scraped_articles)} articles")
        
        return [Article(**article) for article in scraped_articles]

if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")

    news_sites = ["nytimes.com", "cnn.com", "washingtonpost.com", "foxnews.com", "usatoday.com", "wsj.com"]
    scraper = Scraper(text_api_key=API_KEY, cx=CX, news_sites=news_sites)

    text_query = "latest US election news"
    scraped_articles = scraper.search_and_scrape(text_query=text_query, image_query=None, num_results=10)
    print("Scraped articles from text query:")
    for article in scraped_articles:
        print(article)