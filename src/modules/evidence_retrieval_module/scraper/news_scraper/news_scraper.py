# news_scraper/news_scraper.py

from newsplease import NewsPlease
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

class NewsPleaseScraper:
    def __init__(self, user_agent=None):
        self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
    def scrape(self, urls) -> List[Dict]:
        """
        Scrape multiple URLs with timeout control.
        
        Args:
            urls: Single URL string or list of URLs
            total_timeout: Maximum total time (in seconds) for all scraping
                
        Returns:
            List of successfully scraped article data
        """
        # Ensure urls is a list
        if isinstance(urls, str):
            urls = [urls]
                
        if not urls:
            return []

        total_timeout = 6 * len(urls)
        scraped_articles = []
        
        # Create a ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor() as executor:
            try:
                # Submit the scraping task
                future = executor.submit(
                    NewsPlease.from_urls,
                    urls=urls,
                    timeout=6,
                    user_agent=self.user_agent
                )
                
                # Wait for result with timeout
                articles = future.result(timeout=total_timeout)
                
                # Process successful results
                for url, article in articles.items():
                    try:
                        if article is None:
                            continue
                            
                        article_data = {
                            'authors': article.authors if hasattr(article, 'authors') else [],
                            'date_download': article.date_download if hasattr(article, 'date_download') else None,
                            'date_modify': article.date_modify if hasattr(article, 'date_modify') else None,
                            'title': article.title if hasattr(article, 'title') else '',
                            'date': article.date_publish if hasattr(article, 'date_publish') else None,
                            'content': article.maintext if hasattr(article, 'maintext') else '',
                            'url': url,
                            'description': article.description if hasattr(article, 'description') else '',
                            'image_url': article.image_url if hasattr(article, 'image_url') else '',
                            'language': article.language if hasattr(article, 'language') else '',
                            'source_domain': article.source_domain if hasattr(article, 'source_domain') else ''
                        }
                        scraped_articles.append(article_data)
                        
                    except AttributeError as e:
                        print(f"Article object missing attributes for {url}: {e}")
                        print(article)
                    except Exception as e:
                        print(f"Error processing article from {url}: {e}")
                        
            except TimeoutError:
                print(f"Scraping timed out after {total_timeout} seconds")
            except Exception as e:
                print(f"Failed to scrape URLs with error: {str(e)}")
            finally:
                # Always attempt to cancel the future
                if not future.done():
                    future.cancel()
                
        return scraped_articles

# Example usage with your filtered URLs
if __name__ == "__main__":
    # Instantiate the scraper
    scraper = NewsPleaseScraper()

    news_urls = [
        "https://www.factcheck.org/2024/12/factchecking-trumps-meet-the-press-interview/",
    ]

    scraped_articles = scraper.scrape(news_urls)

    # Scrape each article and print the results
    if scraped_articles:
        for article in scraped_articles:
            print(f"Scraped Article:\n{article}\n")
