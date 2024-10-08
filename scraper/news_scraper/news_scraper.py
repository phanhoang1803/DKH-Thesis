from newsplease import NewsPlease

class NewsPleaseScraper:
    def __init__(self, user_agent=None):
        """
        Initialize the NewsPleaseScraper with an optional User-Agent.

        Parameters:
        user_agent (str): Custom User-Agent to be used while scraping.
        """
        self.user_agent = user_agent or 'NewsPleaseBot/0.1 (+https://github.com/fhamborg/news-please)'
    
    def scrape(self, urls):
        """
        Scrape news articles using news-please. Handles both single URL and multiple URLs.

        Parameters:
        urls (str or list): A single URL (str) or a list of URLs (list) to scrape.

        Returns:
        list: A list of dictionaries containing article metadata for each scraped article.
        """
        # Ensure urls is a list for uniform processing
        if isinstance(urls, str):
            urls = [urls]
        
        try:
            # Scrape multiple URLs
            if len(urls) > 1:
                articles = NewsPlease.from_urls(urls, user_agent=self.user_agent)
            else:
                articles = {urls[0]: NewsPlease.from_url(urls[0], user_agent=self.user_agent)}
            
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

    # Example filtered URLs from the GoogleNewsSearch results
    filtered_news = [
        "https://edition.cnn.com/2024/10/07/politics/debt-harris-trump-proposals/index.html",
    ]

    scraped_articles = scraper.scrape(filtered_news)

    # Scrape each article and print the results
    if scraped_articles:
        for article in scraped_articles:
            print(f"Scraped Article:\n{article}\n")
