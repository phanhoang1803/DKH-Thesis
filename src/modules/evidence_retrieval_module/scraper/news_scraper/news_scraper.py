# news_scraper/news_scraper.py

from newsplease import NewsPlease

class NewsPleaseScraper:
    def __init__(self, user_agent=None):
        self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
    def scrape(self, urls):
        # Ensure urls is a list for uniform processing
        if isinstance(urls, str):
            urls = [urls]
        
        try:
            # Scrape multiple URLs
            articles = NewsPlease.from_urls(urls, user_agent=self.user_agent, timeout=10)
            
            # Collect article data for each URL
            scraped_articles = []
            for url, article in articles.items():
                if article:
                    article_data = {
                        'authors': article.authors,
                        'date_download': article.date_download,
                        'date_modify': article.date_modify,
                        'title': article.title,
                        'date': article.date_publish,
                        'content': article.maintext,
                        'url': url,
                        'description': article.description,
                        'image_url': article.image_url,
                        'language': article.language,
                        'source_domain': article.source_domain
                    }
                    scraped_articles.append(article_data)
            return scraped_articles
        except Exception as e:
            print(f"Failed to scrape URLs: {e}")
            return None

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
