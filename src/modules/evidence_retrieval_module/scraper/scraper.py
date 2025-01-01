# scraper.py
import os
from dataclasses import dataclass, field
from datetime import datetime
import re
from dotenv import load_dotenv
from src.modules.evidence_retrieval_module.scraper.news_scraper.news_scraper import NewsPleaseScraper
from src.modules.evidence_retrieval_module.scraper.search_engine.google_image_search import GoogleImageSearch
from src.modules.evidence_retrieval_module.scraper.search_engine.google_text_search import GoogleTextSearch
from src.utils.logger import Logger
from src.config import is_valid_url

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

    def _clean_text(self, text: str) -> str:
        """Clean text by removing/replacing problematic characters."""
        if not text:
            return ""
            
        # List of problematic Unicode characters to remove
        chars_to_remove = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\u202a',  # Left-to-right embedding
            '\ufeff',  # Zero-width no-break space
            '\u2011',  # Non-breaking hyphen
            '\u2033',  # Double prime
            '\u0107',  # Latin small letter c with acute
            '\u0219',  # Latin small letter s with comma below
            '\u010d',  # Latin small letter c with caron
            '\u0101',  # Latin small letter a with macron
            '\u014d',  # Latin small letter o with macron
            '\u2665',  # Black heart suit
            '\U0001f61b'  # Face with stuck-out tongue
        ]
        
        # Remove problematic characters
        for char in chars_to_remove:
            text = text.replace(char, '')
        
        # Remove or fix other special characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    def to_dict(self):
        try:
            return {
                "title": self._clean_text(self.title),
                "date": self.date.isoformat() if self.date else None,
                "content": self._clean_text(self.content),
                "description": self._clean_text(self.description),
                "source_domain": self._clean_text(self.source_domain)
            }
        except Exception as e:
            return {
                "error": f"Serialization failed: {str(e)}",
                "url": self._clean_text(self.url),
                "title": self._clean_text(self.title)
            }

class Scraper:
    def __init__(self, text_api_key, cx, news_sites=None, fact_checking_sites=None):
        self.google_text_search = GoogleTextSearch(api_key=text_api_key, cx=cx, news_sites=news_sites, fact_checking_sites=fact_checking_sites)
        self.google_image_search = GoogleImageSearch(news_sites=news_sites)
        self.news_scraper = NewsPleaseScraper()
        self.logger = Logger(name="Scraper")
        
    def search_and_scrape(self, text_query=None, image_query=None, num_results=10, news_factcheck_ratio=0.7):
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
        news_factcheck_ratio : float, optional
            The ratio of news sites to fact-checking sites. Defaults to 0.7. 
            If 1, only news sites are searched. If 0, only fact-checking sites are searched.

        Returns
        -------
        list
            A list of `Article` objects, each representing a scraped news article.
        """
        if not text_query and not image_query:
            raise ValueError("At least one of 'text_query' or 'image_query' must be provided.")

        all_urls = []

        if text_query:
            text_results = self.google_text_search.search(query=text_query, num_results=num_results, news_factcheck_ratio=news_factcheck_ratio)
            text_urls = [item['link'] for item in text_results]
            all_urls.extend(text_urls)

        if image_query:
            image_results = self.google_image_search.search(image_query, num_results)
            image_urls = [item['link'] for item in image_results]
            all_urls.extend(image_urls)

        self.logger.info(f"[INFO] Retrieved {len(all_urls)} URLs from search engine to scrape")
        all_urls = list(set(all_urls))
        
        # Filter out URLs including video, or pdf.
        all_urls = [url for url in all_urls if is_valid_url(url)]
        self.logger.info(f"[INFO] Filtered {len(all_urls)} URLs to scrape")
        
        scraped_articles = self.news_scraper.scrape(all_urls)
        self.logger.info(f"[INFO] Scraped {len(scraped_articles)} articles")
        
        return [Article(**article) for article in scraped_articles]

if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    from src.config import NEWS_SITES, FACT_CHECKING_SITES
    
    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")
    
    scraper = Scraper(text_api_key=API_KEY, cx=CX, news_sites=NEWS_SITES, fact_checking_sites=FACT_CHECKING_SITES)

    text_query = "latest US election news"
    scraped_articles = scraper.search_and_scrape(text_query=text_query, image_query=None, num_results=100, news_factcheck_ratio=0.7)
    print("Scraped articles from text query:")
    for article in scraped_articles:
        print(article)