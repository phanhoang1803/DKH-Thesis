# # # news_scraper/news_scraper.py

# # import os
# # from newsplease import NewsPlease
# # from typing import List, Dict
# # from concurrent.futures import ThreadPoolExecutor
# # import concurrent.futures as cf

# # import psutil

# # class NewsPleaseScraper:
# #     def __init__(self, user_agent=None):
# #         self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
# #     def kill_child_processes(self, parent_pid=None, timeout=3):
# #         """Forcefully kill all child processes and their descendants"""
# #         if parent_pid is None:
# #             parent_pid = os.getpid()
        
# #         try:
# #             parent = psutil.Process(parent_pid)
# #         except psutil.NoSuchProcess:
# #             return

# #         children = parent.children(recursive=True)
        
# #         for child in children:
# #             try:
# #                 # Check if process is a NewsPlease or related process
# #                 if "news" in child.name().lower() or "selenium" in child.name().lower():
# #                     child.terminate()
# #             except (psutil.NoSuchProcess, psutil.AccessDenied):
# #                 pass

# #         _, alive = psutil.wait_procs(children, timeout=timeout)
        
# #         for child in alive:
# #             try:
# #                 if "news" in child.name().lower() or "selenium" in child.name().lower():
# #                     child.kill()
# #             except (psutil.NoSuchProcess, psutil.AccessDenied):
# #                 pass
    
# #     def scrape(self, urls) -> List[Dict]:
# #         """
# #         Scrape multiple URLs with timeout control.
        
# #         Args:
# #             urls: Single URL string or list of URLs
# #             total_timeout: Maximum total time (in seconds) for all scraping
                
# #         Returns:
# #             List of successfully scraped article data
# #         """
# #         # Ensure urls is a list
# #         if isinstance(urls, str):
# #             urls = [urls]
                
# #         if not urls:
# #             return []

# #         total_timeout = 6 * len(urls)
# #         scraped_articles = []
        
# #         # Create a ThreadPoolExecutor for timeout control
# #         with ThreadPoolExecutor() as executor:
# #             try:
# #                 # Submit the scraping task
# #                 future = executor.submit(
# #                     NewsPlease.from_urls,
# #                     urls=urls,
# #                     timeout=6,
# #                     user_agent=self.user_agent
# #                 )
                
# #                 try:
# #                     # Wait for result with timeout
# #                     articles = future.result(timeout=total_timeout)
                    
# #                     if not articles:
# #                         print("No articles returned from NewsPlease")
# #                         return scraped_articles
                    
# #                     # Process successful results
# #                     for url, article in articles.items():
# #                         try:
# #                             if article is None:
# #                                 continue
                                
# #                             article_data = {
# #                                 'authors': article.authors if hasattr(article, 'authors') else [],
# #                                 'date_download': article.date_download if hasattr(article, 'date_download') else None,
# #                                 'date_modify': article.date_modify if hasattr(article, 'date_modify') else None,
# #                                 'title': article.title if hasattr(article, 'title') else '',
# #                                 'date': article.date_publish if hasattr(article, 'date_publish') else None,
# #                                 'content': article.maintext if hasattr(article, 'maintext') else '',
# #                                 'url': url,
# #                                 'description': article.description if hasattr(article, 'description') else '',
# #                                 'image_url': article.image_url if hasattr(article, 'image_url') else '',
# #                                 'language': article.language if hasattr(article, 'language') else '',
# #                                 'source_domain': article.source_domain if hasattr(article, 'source_domain') else ''
# #                             }
# #                             scraped_articles.append(article_data)
                            
# #                         except AttributeError as e:
# #                             print(f"Article object missing attributes for {url}: {e}")
# #                             print(article)
# #                         except Exception as e:
# #                             print(f"Error processing article from {url}: {e}")
                            
# #                 except cf._base.TimeoutError:
# #                     print(f"Scraping timed out after {total_timeout} seconds")
# #                     if future:
# #                         future.cancel()
# #                     if executor:
# #                         executor.shutdown(wait=False, cancel_futures=True)
# #                     self.kill_child_processes()
# #             except Exception as e:
# #                 error_msg = str(e) if str(e) else "Unknown error occurred"
# #                 print(f"Failed to scrape URLs with error: {error_msg}")
# #             finally:
# #                 # Always attempt to cancel the future
# #                 if 'future' in locals():
# #                     future.cancel()
# #                 executor.shutdown(wait=False, cancel_futures=True)
# #                 self.kill_child_processes()
# #         return scraped_articles

# # # Example usage with your filtered URLs
# # if __name__ == "__main__":
# #     # Instantiate the scraper
# #     scraper = NewsPleaseScraper()

# #     news_urls = [
# #         "https://www.factcheck.org/2024/12/factchecking-trumps-meet-the-press-interview/",
# #     ]

# #     scraped_articles = scraper.scrape(news_urls)

# #     # Scrape each article and print the results
# #     if scraped_articles:
# #         for article in scraped_articles:
# #             print(f"Scraped Article:\n{article}\n")


# import os
# from newsplease import NewsPlease
# from typing import List, Dict, Optional, Union
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import concurrent.futures as cf
# import psutil
# import signal
# import time

# class NewsPleaseScraper:
#     def __init__(self, user_agent: Optional[str] = None, timeout_per_url: int = 6):
#         self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
#         self.timeout_per_url = timeout_per_url
#         # Store process IDs to clean up
#         self.child_processes = set()
        
#     def _setup_timeout_handler(self):
#         """Set up signal handler for timeout"""
#         def timeout_handler(signum, frame):
#             raise TimeoutError("Scraping operation timed out")
#         signal.signal(signal.SIGALRM, timeout_handler)

#     def kill_child_processes(self, parent_pid: Optional[int] = None, timeout: int = 3) -> None:
#         """Forcefully kill all child processes and their descendants"""
#         if parent_pid is None:
#             parent_pid = os.getpid()
        
#         try:
#             parent = psutil.Process(parent_pid)
#             children = parent.children(recursive=True)
            
#             # First try graceful termination
#             for child in children:
#                 try:
#                     if "news" in child.name().lower() or "selenium" in child.name().lower():
#                         self.child_processes.add(child.pid)
#                         child.terminate()
#                 except (psutil.NoSuchProcess, psutil.AccessDenied):
#                     continue

#             # Wait for processes to terminate
#             _, alive = psutil.wait_procs(children, timeout=timeout)
            
#             # Force kill remaining processes
#             for child in alive:
#                 try:
#                     if child.pid in self.child_processes:
#                         child.kill()
#                 except (psutil.NoSuchProcess, psutil.AccessDenied):
#                     continue
                
#             # Clear the set of tracked processes
#             self.child_processes.clear()
            
#         except psutil.NoSuchProcess:
#             pass
#         except Exception as e:
#             print(f"Error during process cleanup: {e}")

#     def _scrape_single_url(self, url: str) -> Dict:
#         """Scrape a single URL with timeout"""
#         try:
#             articles = NewsPlease.from_urls(
#                 urls=[url],
#                 timeout=self.timeout_per_url,
#                 user_agent=self.user_agent
#             )
            
#             if not articles or url not in articles or articles[url] is None:
#                 raise ValueError(f"No article data retrieved for {url}")
                
#             article = articles[url]
#             return {
#                 'authors': getattr(article, 'authors', []),
#                 'date_download': getattr(article, 'date_download', None),
#                 'date_modify': getattr(article, 'date_modify', None),
#                 'title': getattr(article, 'title', ''),
#                 'date': getattr(article, 'date_publish', None),
#                 'content': getattr(article, 'maintext', ''),
#                 'url': url,
#                 'description': getattr(article, 'description', ''),
#                 'image_url': getattr(article, 'image_url', ''),
#                 'language': getattr(article, 'language', ''),
#                 'source_domain': getattr(article, 'source_domain', '')
#             }
#         except Exception as e:
#             print(f"Error scraping {url}: {e}")
#             return None

#     def scrape(self, urls: Union[str, List[str]], max_workers: int = 3) -> List[Dict]:
#         """
#         Scrape multiple URLs with improved timeout control and error handling.
        
#         Args:
#             urls: Single URL string or list of URLs
#             max_workers: Maximum number of concurrent threads
                
#         Returns:
#             List of successfully scraped article data
#         """
#         if isinstance(urls, str):
#             urls = [urls]
                
#         if not urls:
#             return []

#         scraped_articles = []
        
#         # Calculate total timeout based on number of URLs and workers
#         total_timeout = min(self.timeout_per_url * len(urls), 300)  # Cap at 5 minutes
        
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             future_to_url = {
#                 executor.submit(self._scrape_single_url, url): url 
#                 for url in urls
#             }
            
#             try:
#                 # Use as_completed with timeout
#                 for future in as_completed(future_to_url, timeout=total_timeout):
#                     url = future_to_url[future]
#                     try:
#                         article_data = future.result()
#                         if article_data:
#                             scraped_articles.append(article_data)
#                     except Exception as e:
#                         print(f"Failed to scrape {url}: {e}")
                        
#             except cf._base.TimeoutError:
#                 print(f"Scraping timed out after {total_timeout} seconds")
#             except Exception as e:
#                 print(f"Unexpected error during scraping: {e}")
#             finally:
#                 # Cancel all pending futures
#                 for future in future_to_url:
#                     future.cancel()
#                 # Clean up processes
#                 self.kill_child_processes()
                
#         return scraped_articles

# if __name__ == "__main__":
#     # Example usage
#     scraper = NewsPleaseScraper(timeout_per_url=10)
    
#     news_urls = [
#         "https://www.factcheck.org/2024/12/factchecking-trumps-meet-the-press-interview/",
#     ]

#     try:
#         scraped_articles = scraper.scrape(news_urls, max_workers=2)
        
#         if scraped_articles:
#             for article in scraped_articles:
#                 print(f"Successfully scraped: {article['title']}\n")
#         else:
#             print("No articles were successfully scraped")
            
#     except Exception as e:
#         print(f"Scraping failed: {e}")



import os
from newsplease import NewsPlease
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures as cf
import psutil
import signal
import time
import random
from urllib.error import HTTPError
from requests.exceptions import RequestException

class NewsPleaseScraper:
    def __init__(self, timeout_per_url: int = 6):
        # Rotate between different user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        self.timeout_per_url = timeout_per_url
        self.child_processes = set()
        self.retry_count = 3
        self.retry_delay = 2  # seconds between retries
        
    def get_random_user_agent(self) -> str:
        """Get a random user agent from the list"""
        return random.choice(self.user_agents)

    def kill_child_processes(self, parent_pid: Optional[int] = None, timeout: int = 3) -> None:
        """Forcefully kill all child processes and their descendants"""
        if parent_pid is None:
            parent_pid = os.getpid()
        
        try:
            parent = psutil.Process(parent_pid)
            children = parent.children(recursive=True)
            
            # First try graceful termination
            for child in children:
                try:
                    process_name = child.name().lower()
                    if any(name in process_name for name in ["news", "selenium", "chrome", "firefox"]):
                        self.child_processes.add(child.pid)
                        child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Wait for processes to terminate
            gone, alive = psutil.wait_procs(children, timeout=timeout)
            
            # Force kill remaining processes
            for child in alive:
                try:
                    if child.pid in self.child_processes:
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                
            # Clear the set of tracked processes
            self.child_processes.clear()
            
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            print(f"Error during process cleanup: {e}")

    def _scrape_single_url(self, url: str, attempt: int = 1) -> Optional[Dict]:
        """
        Scrape a single URL with retries and delay
        """
        if attempt > self.retry_count:
            print(f"Max retries exceeded for {url}")
            return None

        try:
            # Add random delay between 1-3 seconds
            time.sleep(random.uniform(1, 3))
            
            articles = NewsPlease.from_urls(
                urls=[url],
                timeout=self.timeout_per_url,
                user_agent=self.get_random_user_agent()
            )
            
            if not articles or url not in articles or articles[url] is None:
                raise ValueError(f"No article data retrieved for {url}")
                
            article = articles[url]
            return {
                'authors': getattr(article, 'authors', []),
                'date_download': getattr(article, 'date_download', None),
                'date_modify': getattr(article, 'date_modify', None),
                'title': getattr(article, 'title', ''),
                'date': getattr(article, 'date_publish', None),
                'content': getattr(article, 'maintext', ''),
                'url': url,
                'description': getattr(article, 'description', ''),
                'image_url': getattr(article, 'image_url', ''),
                'language': getattr(article, 'language', ''),
                'source_domain': getattr(article, 'source_domain', '')
            }
            
        except HTTPError as e:
            if e.code == 403:
                print(f"403 Forbidden error for {url}. Retrying with different user agent...")
                time.sleep(self.retry_delay * attempt)  # Exponential backoff
                return self._scrape_single_url(url, attempt + 1)
            else:
                print(f"HTTP error {e.code} for {url}")
                return None
                
        except (RequestException, TimeoutError) as e:
            print(f"Request failed for {url}: {e}")
            time.sleep(self.retry_delay * attempt)  # Exponential backoff
            return self._scrape_single_url(url, attempt + 1)
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        finally:
            # Ensure we clean up any hanging processes
            self.kill_child_processes()

    def scrape(self, urls: Union[str, List[str]], max_workers: int = 2) -> List[Dict]:
        """
        Scrape multiple URLs with improved timeout and error handling
        """
        if isinstance(urls, str):
            urls = [urls]
                
        if not urls:
            return []

        scraped_articles = []
        total_timeout = min(self.timeout_per_url * len(urls) * self.retry_count, 300)  # Cap at 5 minutes
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._scrape_single_url, url): url 
                for url in urls
            }
            
            try:
                for future in as_completed(future_to_url, timeout=total_timeout):
                    url = future_to_url[future]
                    try:
                        article_data = future.result()
                        if article_data:
                            scraped_articles.append(article_data)
                            print(f"Successfully scraped: {url}")
                    except Exception as e:
                        print(f"Failed to scrape {url}: {e}")
                        
            except cf._base.TimeoutError:
                print(f"Scraping timed out after {total_timeout} seconds")
                self.kill_child_processes()
            except Exception as e:
                print(f"Unexpected error during scraping: {e}")
            finally:
                # Cancel all pending futures
                for future in future_to_url:
                    if not future.done():
                        future.cancel()
                # Clean up processes
                self.kill_child_processes()
                
        return scraped_articles

if __name__ == "__main__":
    scraper = NewsPleaseScraper(timeout_per_url=8)
    
    news_urls = [
        "https://www.factcheck.org/2024/12/factchecking-trumps-meet-the-press-interview/",
    ]

    try:
        scraped_articles = scraper.scrape(news_urls, max_workers=2)
        
        if scraped_articles:
            for article in scraped_articles:
                print(f"Successfully scraped: {article['title']}\n")
        else:
            print("No articles were successfully scraped")
            
    except Exception as e:
        print(f"Scraping failed: {e}")