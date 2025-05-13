# This script is used to download the direct annotations for the visual news data.
# However, currently, it is not working because of NewsPleaseScraper was removed. 
# We will update it later.

#!/usr/bin/env python
# coding: utf-8

import argparse
import requests
import os
import time
import json
import concurrent.futures as cf
from functools import partial
import tqdm
import os
import random
import urllib.parse
from typing import List, Dict, Any, Tuple, Optional
from filelock import FileLock
from datetime import datetime
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# Import NewsPleaseScraper
# from src.modules.evidence_retrieval_module.scraper.news_scraper.news_scraper import NewsPleaseScraper

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download dataset for direct search queries')
    parser.add_argument('--save_folder_path', type=str, default='queries_dataset',
                        help='location where to download data')
    parser.add_argument('--visual_news_data_path', type=str, default='test_dataset/visual_news_test.json',
                        help='path to the visual news data')
    parser.add_argument('--news_clippings_data_path', type=str, default='test_dataset/news_clippings_test.json',
                        help='path to the news clippings data')
    
    parser.add_argument('--google_cred_json', type=str, default='credentials.json',
                        help='json file for credentials')
                        
    parser.add_argument('--split_type', type=str, default='merged_balanced',
                        help='which split to use in the NewsCLIP dataset')
    parser.add_argument('--sub_split', type=str, default='test',
                        help='which split to use from train,val,test splits')
                        
    parser.add_argument('--how_many_queries', type=int, default=1,
                        help='how many query to issue for each item - each query is 10 images')
    parser.add_argument('--continue_download', type=int, default=1,
                        help='whether to continue download or start from 0 - should be 0 or 1')

    parser.add_argument('--how_many', type=int, default=-1,
                        help='how many items to query and download, 0 means download untill the end')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='where to end, if not specified, will be inferred from how_many')    
    parser.add_argument('--start_idx', type=int, default=-1,
                        help='where to start, if not specified will be inferred from the current saved json or 0 otherwise')
    parser.add_argument('--random_index_path', type=str, default=None,
                        help='path to the file containing the random indices')

    parser.add_argument('--hashing_cutoff', type=int, default=15,
                        help='threshold used in hashing')
    parser.add_argument('--skip_existing', action="store_true")
    
    # Selenium specific arguments
    parser.add_argument('--selenium_type', type=str, default='selenium')
    parser.add_argument('--headless', action="store_true", 
                        help='Run Chrome in headless mode')
    parser.add_argument('--chrome_path', type=str, default=None,
                        help='Path to Chrome binary')
    parser.add_argument('--driver_path', type=str, default=None,
                        help='Path to Chrome driver')
    parser.add_argument('--proxy', type=str, default=None,
                        help='Proxy to use for Selenium (format: host:port)')
    parser.add_argument('--max_wait_time', type=int, default=30,
                        help='Maximum time to wait for elements to load in seconds')
    
    # NewsPleaseScraper specific arguments
    parser.add_argument('--timeout_per_url', type=int, default=6,
                        help='Timeout for each URL when scraping')
    parser.add_argument('--max_workers', type=int, default=2,
                        help='Max number of workers for parallel scraping')
    
    args = parser.parse_args()
    return args

# Constants for allowed domains and excluded keywords
NEWS_DOMAINS = [
    # Major News Organizations
    "theguardian.com", "usatoday.com", "bbc.com", "bbc.co.uk", "cnn.com", 
    "edition.cnn.com", "latimes.com", "independent.co.uk", "nbcnews.com", 
    "npr.org", "aljazeera.com", "apnews.com", "cbsnews.com", "abcnews.go.com", 
    "pbs.org", "abc.net.au", "vox.com", "euronews.com",
    
    # Newspapers
    "denverpost.com", "tennessean.com", "thetimes.com", "sandiegouniontribune.com",
    "nytimes.com", "washingtontimes.com",
    
    # Magazines/Long-form Journalism
    "magazine.atavist.com", "newyorker.com", "theatlantic.com", "vanityfair.com",
    "economist.com", "ffxnow.com", "laist.com", "hudson.org", "rollcall.com",
    "nps.gov", "reuters.com"
]

# NEWS_DOMAINS = [
#     # Major News Organizations
#     "theguardian.com", "usatoday.com", "washingtontimes.com", "bbc.com", "bbc.co.uk", "cnn.com",
    
#     "pbs.org", "nbcnews.com", "latimes.com"
# ]

EXCLUDED_DOMAINS = [
    "mdpi", "yumpu", "scmp", "pinterest", "imdb",
    "movieweb", "shutterstock", "reddit", "alamy",
    "alamy.it", "alamyimages", "planetcricket",
    "cnnbrasil", "infomoney", "gettyimages",
    "washingtonpost", "youtube", "facebook", "researchgate", 
]

EXCLUDE_KEYWORDS = [
    'stock photography',
    'stock photo',
    'stock photos',
    'stock photo images',
    'stock photo images',
    'gallery',
    'archive',
    'wallpaper',
    'collection',
    'photo',
    'photos'
]

def _normalize_domain(domain: str) -> str:
    """Normalize domain string by removing www. prefix and lowercasing."""
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

def _normalize_domain_for_excluding(domain: str) -> str:
    """Normalize domain string by removing www. prefix and lowercasing."""
    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    domain = domain.split(".")[0]
    return domain

def filter_evidence_by_domain(urls: List[str], allowed_domains: List[str]) -> List[str]:
    """Filter evidence list by allowed domains."""
    # Normalize allowed domains
    normalized_domains = {_normalize_domain(domain) for domain in allowed_domains}
    
    # Filter evidence list
    filtered_urls = []
    for url in urls:
        try:
            domain = _normalize_domain(urllib.parse.urlparse(url).netloc)
            if domain in normalized_domains:
                filtered_urls.append(url)
        except Exception as e:
            print(f"Error parsing URL {url}: {e}")
    
    return filtered_urls

def get_random_user_agent() -> str:
    """Generate a random user agent"""
    try:
        # Try importing fake_useragent if installed
        from fake_useragent import UserAgent
        ua = UserAgent()
        return ua.random
    except ImportError:
        # Fallback user agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15'
        ]
        return random.choice(user_agents)

def setup_selenium_driver(args):
    """Set up and configure Selenium WebDriver"""
    chrome_options = Options()
    
    # Set user agent
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}")
    
    # Headless mode if specified
    if args.headless:
        chrome_options.add_argument("--headless")
    
    # Additional options to avoid detection
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Load cookies from file if they exist
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    
    # Set proxy if specified
    if args.proxy:
        chrome_options.add_argument(f"--proxy-server={args.proxy}")
    
    # Set Chrome binary path if specified
    if args.chrome_path:
        chrome_options.binary_location = args.chrome_path
    
    # Create driver
    try:
        if args.driver_path:
            service = Service(executable_path=args.driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
        # Set page load timeout
        driver.set_page_load_timeout(args.max_wait_time)
        
        # Add CDP commands to make detection harder
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
                """
        })
        
        return driver
    except Exception as e:
        print(f"Error setting up Selenium driver: {e}")
        raise

def load_cookies(driver):
    """Load cookies from file if they exist"""
    try:
        if os.path.exists('google_cookies.json'):
            driver.get("https://www.google.com")
            with open('google_cookies.json', 'r') as f:
                cookies = json.load(f)
                for cookie in cookies:
                    # Some cookies cannot be added directly, handle exceptions
                    try:
                        driver.add_cookie(cookie)
                    except Exception:
                        pass
            # Refresh to apply cookies
            driver.refresh()
    except Exception as e:
        print(f"Error loading cookies: {e}")

def save_cookies(driver):
    """Save cookies to a file"""
    try:
        cookies = driver.get_cookies()
        with open('google_cookies.json', 'w') as f:
            json.dump(cookies, f)
    except Exception as e:
        print(f"Error saving cookies: {e}")

def bypass_consent_page(driver):
    """Attempt to bypass Google's consent page if it appears"""
    try:
        # Look for the consent button and click it
        consent_buttons = [
            "//button[contains(., 'Accept all')]",
            "//button[contains(., 'I agree')]",
            "//button[contains(., 'Agree')]",
            "//div[contains(@role, 'dialog')]//button[1]"  # Usually the first button is Accept/Agree
        ]
        
        for button_xpath in consent_buttons:
            try:
                button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, button_xpath))
                )
                button.click()
                print("Clicked consent button")
                time.sleep(1)  # Wait for page to update
                return True
            except (TimeoutException, NoSuchElementException):
                continue
                
        return False
    except Exception as e:
        print(f"Error bypassing consent page: {e}")
        return False

def scroll_to_load_more_images(driver, scrolls=3):
    """Scroll down to load more images"""
    try:
        for i in range(scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for images to load
    except Exception as e:
        print(f"Error scrolling page: {e}")

def google_search_with_selenium(
    query: str, 
    driver, 
    how_many_queries: int = 1, 
    max_wait_time: int = 30
) -> List[str]:
    """
    Search both Google regular and image results using Selenium and return combined links
    
    Parameters:
    -----------
    query : str
        The search query
    driver : WebDriver
        Selenium WebDriver instance
    how_many_queries : int
        Number of result pages to scrape (10 results per page)
    max_wait_time : int
        Maximum time to wait for page loads
        
    Returns:
    --------
    List[str]
        Combined list of unique URLs from both regular and image searches
    """
    all_links = []
    
    try:
        for i in range(how_many_queries):
            start = i * 10
            
            # Regular search
            regular_search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&hl=en&start={start}"
            all_links.extend(_process_search(driver, regular_search_url, "regular", max_wait_time))
            
            # Image search
            image_search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch&hl=en&start={start}"
            all_links.extend(_process_search(driver, image_search_url, "image", max_wait_time))
                
    except Exception as e:
        print(f"Error in Google search with Selenium: {e}")
    
    # Remove duplicates and return
    return list(set(all_links))


def _process_search(driver, search_url: str, search_type: str, max_wait_time: int) -> List[str]:
    """
    Process a single search page and extract links
    
    Parameters:
    -----------
    driver : WebDriver
        Selenium WebDriver instance
    search_url : str
        URL to navigate to
    search_type : str
        Type of search ("regular" or "image")
    max_wait_time : int
        Maximum time to wait for page loads
        
    Returns:
    --------
    List[str]
        List of extracted URLs
    """
    links = []
    
    try:
        # Navigate to the search URL
        driver.get(search_url)
        
        # Wait for page to load
        time.sleep(2)
        
        # Check if Google detected automated traffic
        if "unusual traffic" in driver.page_source.lower() or "captcha" in driver.page_source.lower():
            print(f"Google detected automated traffic for {search_type} search")
            return links
        
        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        if search_type == "image":
            # Extract image links (data-lpage contains the source page URL)
            links = [item['data-lpage'] for item in soup.find_all(attrs={"data-lpage": True})]
            
            # If no links found with data-lpage, try another approach for images
            if not links:
                divs = soup.find_all('div', {'class': 'isv-r'})
                for div in divs:
                    a_tags = div.find_all('a')
                    for a in a_tags:
                        if 'href' in a.attrs and a['href'].startswith('/url?'):
                            url = a['href'].split('?q=')[1].split('&')[0]
                            links.append(urllib.parse.unquote(url))
        else:
            # Regular search results extraction
            # Look for search result links with href attributes starting with '/url?'
            for a_tag in soup.find_all('a'):
                if 'href' in a_tag.attrs and a_tag['href'].startswith('/url?'):
                    # Extract the URL parameter
                    url = a_tag['href'].split('?q=')[1].split('&')[0]
                    links.append(urllib.parse.unquote(url))
                    
            # Filter out Google's own domains if needed
            links = [link for link in links if not any(g_domain in link for g_domain in 
                    ['google.com/search', 'google.com/imgres', 'accounts.google', 'support.google'])]
                    
    except WebDriverException as e:
        print(f"Selenium error during {search_type} search: {e}")
    except Exception as e:
        print(f"Unexpected error during {search_type} search: {e}")
        
    return links

def init_files_and_paths(args):
    """Initialize files and paths needed for the script"""
    full_save_path = os.path.join(args.save_folder_path, args.split_type, 'direct_search', args.sub_split, args.selenium_type)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Initialize files
    json_download_file_name = os.path.join(full_save_path, args.sub_split + '.json')
    
    # Initialize or load existing annotations
    if os.path.isfile(json_download_file_name) and args.continue_download:
        if os.access(json_download_file_name, os.R_OK):
            with open(json_download_file_name, 'r') as fp:
                all_direct_annotations_idx = json.load(fp)
        else:
            # wait until the file is not locked
            while not os.access(json_download_file_name, os.R_OK):
                time.sleep(1)
            with open(json_download_file_name, 'r') as fp:
                all_direct_annotations_idx = json.load(fp)
    else:
        all_direct_annotations_idx = {}
        with open(json_download_file_name, 'w') as db_file:
            json.dump({}, db_file)
    
    return full_save_path, json_download_file_name, all_direct_annotations_idx

def download_and_save_image(url, save_dir, image_name):
    """Download and save an image"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            img_path = os.path.join(save_dir, f"{image_name}.jpg")
            with open(img_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            return True
        return False
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return False

def save_html(content, path):
    """Save HTML content to a file"""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving HTML to {path}: {e}")
        return False

def process_single_item(url_data, scraper):
    """Process a single URL using NewsPleaseScraper"""
    url, counter, save_folder_path = url_data
    image = {}
    
    # Basic URL parsing for domain and other info
    try:
        parsed_url = urllib.parse.urlparse(url)
        domain = _normalize_domain(parsed_url.netloc)
        
        # Extract some basic info
        image['img_link'] = url
        image['domain'] = domain
        image['page_link'] = url
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return None
    
    # Scrape article with NewsPleaseScraper
    try:
        scraped_articles = scraper.scrape([url], max_workers=1)
        
        if not scraped_articles:
            return ('no_captions', image)
        
        article = scraped_articles[0]
        
        image['img_link'] = article.get("image_url", "")
        
        if not download_and_save_image(image['img_link'], save_folder_path, str(counter)):
            return None

        image['image_path'] = os.path.join(save_folder_path, f"{counter}.jpg")
        image['domain'] = article.get("source_domain", "")
        image['page_link'] = article.get("url", "")
        
        # Save HTML content if available
        if 'html' in article:
            html_content = article['html']
            if save_html(html_content, os.path.join(save_folder_path, f"{counter}.txt")):
                image['html_path'] = os.path.join(save_folder_path, f"{counter}.txt")
        
        image['page_title'] = article.get("title", "")
        image['caption'] = article.get("description", "")
        image['snippet'] = article.get("content", "")
        
        if image['caption'] == "":
            return ('no_captions', {})
        
        return ('with_captions', image)
    
    except Exception as e:
        print(f"Error scraping article {url}: {e}")
    
    # If scraping failed or no content was found
    return ('no_captions', image)

def get_direct_search_annotation(urls, save_folder_path, scraper):
    """Process search results in parallel"""
    items_to_process = []
    counter = 0
    
    for url in urls:
        # Filter by news domains
        try:
            domain = _normalize_domain(urllib.parse.urlparse(url).netloc)
            if domain in NEWS_DOMAINS and "gallery" not in url and "video" not in url and ".pdf" not in url:
                items_to_process.append((url, counter, save_folder_path))
                counter += 1
                
                # Scrape only top 5 items
                if counter == 10:
                    break
        except Exception as e:
            print(f"Error parsing URL {url}: {e}")
    
    if not items_to_process:
        return {}

    print(items_to_process)

    results = defaultdict(list)
    
    # Process items in parallel
    with cf.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_item, item_data, scraper): item_data
            for item_data in items_to_process
        }
        
        try:
            for future in cf.as_completed(futures, timeout=60):  # Global timeout
                try:
                    result = future.result(timeout=30)  # Timeout per task
                    if result:
                        category, image = result
                        results[category].append(image)
                except Exception as e:
                    item_data = futures[future]
                    print(f'Failed to process item {item_data[1]}: {str(e)}')
        
        except KeyboardInterrupt:
            print("🛑 User interrupted! Shutting down all processes...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise  # Re-raise KeyboardInterrupt
        
        except Exception as e:
            print(f"🔥 Critical error: {str(e)}. Forcing shutdown.")
            executor.shutdown(wait=False, cancel_futures=True)

    if not results:
        return {}

    return {
        'images_with_captions': results['with_captions'],
        'images_with_no_captions': results['no_captions'],
        'images_with_caption_matched_tags': results['matched_tags']
    }

def main():
    args = parse_arguments()
    
    # Initialize environment and paths
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_cred_json
    full_save_path, json_download_file_name, all_direct_annotations_idx = init_files_and_paths(args)
    
    # Load datasets
    visual_news_data_mapping = json.load(open(args.visual_news_data_path))
    clip_data = json.load(open(args.news_clippings_data_path))
    clip_data_annotations = clip_data["annotations"]
    
    # Determine start and end indices
    start_counter = (args.start_idx if args.start_idx != -1 
                    else (int(list(all_direct_annotations_idx.keys())[-1])+2 
                          if all_direct_annotations_idx else 0))
    
    end_counter = (args.end_idx if args.end_idx > 0 
                  else (start_counter + 2*args.how_many if args.how_many > 0 
                        else len(clip_data_annotations)))
    
    if args.random_index_path:
        try:
            with open(args.random_index_path, 'r') as f:
                random_indices = [int(line.strip()) for line in f.readlines()]
        except Exception as e:
            print(f"Error in reading random indices file: {str(e)}")
    else:
        random_indices = list(range(start_counter, end_counter))
            
    # Select even indices in random_indices which are between start_counter and end_counter
    # If odd, then select the previous even number
    # Prevent duplicate indices
    
    indices = []
    for idx in random_indices:
        if idx % 2 == 0 and start_counter <= idx <= end_counter:
            indices.append(idx)
        elif idx % 2 == 1 and start_counter <= idx - 1 <= end_counter:
            indices.append(idx - 1)
            
    # Remove duplicate indices
    indices = list(set(indices))
    indices = [int(x) for x in indices]  # Convert all elements to integers
    indices.sort()
    if len(indices) > 0:
        print(f"Processing items from {indices[0]} to {indices[-1]}")
    
    # Create a temporary storage for search links
    search_links_path = os.path.join(full_save_path, "search_links.json")
    
    # Check if we have cached search links
    search_links_by_index = {}
    if os.path.exists(search_links_path) and args.continue_download:
        try:
            # Use file locking when loading to prevent race conditions
            lock_file = f"{search_links_path}.lock"
            with FileLock(lock_file):
                with open(search_links_path, 'r') as f:
                    search_links_by_index = json.load(f)
            print(f"Loaded {len(search_links_by_index)} cached search results")
        except Exception as e:
            print(f"Error loading cached search links: {str(e)}")
            search_links_by_index = {}
    
    # PHASE 1: Collect all search links
    if args.skip_existing:
        indices_to_search = [i for i in indices if str(i) not in search_links_by_index]
    else:
        indices_to_search = indices
            
    if indices_to_search:
        print(f"Phase 1: Collecting search links for {len(indices_to_search)} indices...")
        
        # Initialize Selenium WebDriver for search phase only
        driver = setup_selenium_driver(args)
        
        try:
            # Load cookies if available
            load_cookies(driver)
            
            # Main search loop
            for i in tqdm.tqdm(indices_to_search):
                start_time = time.time()
                
                try:
                    ann = clip_data_annotations[i]
                    text_query = visual_news_data_mapping[str(ann["id"])]["caption"]
                    
                    # Process single query using our Selenium Google Search function
                    links = google_search_with_selenium(
                        query=text_query,
                        driver=driver,
                        how_many_queries=args.how_many_queries,
                        max_wait_time=args.max_wait_time
                    )
                    
                    # Store the search links
                    if links:
                        search_links_by_index[str(i)] = {
                            "query": text_query,
                            "links": links,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Use file locking to prevent race conditions when saving
                        lock_file = f"{search_links_path}.lock"
                        with FileLock(lock_file):
                            # Read the latest version first in case other processes have updated it
                            if os.path.exists(search_links_path):
                                with open(search_links_path, 'r') as f:
                                    current_links = json.load(f)
                                # Update with our new link
                                current_links[str(i)] = search_links_by_index[str(i)]
                            else:
                                current_links = {str(i): search_links_by_index[str(i)]}
                                
                            # Write back the updated data
                            with open(search_links_path, 'w') as f:
                                json.dump(current_links, f)
                            
                            # Update our local copy with the complete dataset
                            search_links_by_index = current_links
                
                except Exception as e:
                    print(f"Error searching for item {i}: {str(e)}")
                
                print(f"Search for item {i} completed in {time.time() - start_time:.2f} seconds")
                
                # Random delay to avoid detection
                time.sleep(random.uniform(2, 5))
                
                # Refresh the driver every 10 searches to prevent stale sessions
                if len(search_links_by_index) % 20 == 0:
                    try:
                        driver.quit()
                        time.sleep(2)
                        driver = setup_selenium_driver(args)
                        load_cookies(driver)
                    except Exception as e:
                        print(f"Error refreshing driver: {str(e)}")
                        driver = setup_selenium_driver(args)
                        load_cookies(driver)
        
        finally:
            # Always close the driver to release resources
            try:
                driver.quit()
            except:
                pass
    
    # PHASE 2: Process the search links to get direct annotations
    print(f"Phase 2: Processing {len(search_links_by_index)} search results...")
    
    # Initialize NewsPleaseScraper
    scraper = NewsPleaseScraper(timeout_per_url=args.timeout_per_url)
    
    for i in tqdm.tqdm(indices):
        if args.skip_existing:
            if os.path.exists(os.path.join(full_save_path, str(i))):
                # If the folder exists, and the direct_annotation.json file exists, skip the item
                if os.path.exists(os.path.join(full_save_path, str(i), 'direct_annotation.json')):
                    with open(os.path.join(full_save_path, str(i), 'direct_annotation.json'), "r") as f:
                        data = json.load(f)
                    
                    scrape = False
                    for item in data["images_with_captions"]:
                        if item["caption"] == None:
                            scrape = True
                            
                    if not scrape:
                        continue
        
        # Skip if we don't have search links for this index
        if str(i) not in search_links_by_index:
            print(f"No search links found for index {i}, skipping")
            continue
        
        start_time = time.time()
        
        print(f"Processeing item {i}")
        
        # Get the search links for this index
        search_result = search_links_by_index[str(i)]
        links = search_result["links"]
        
        new_folder_path = os.path.join(full_save_path, str(i))
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Process the links using NewsPleaseScraper
        direct_search_results = get_direct_search_annotation(links, new_folder_path, scraper)
        
        # Save results
        if direct_search_results:
            try:
                ann = clip_data_annotations[i]
                new_entry = {
                    str(i): {
                        'image_id_in_visualNews': ann["image_id"],
                        'text_id_in_visualNews': ann["id"],
                        'folder_path': new_folder_path,
                        'query': search_result["query"]
                    }
                }
                
                try:
                    # Use file locking to prevent race conditions
                    lock_file = f"{json_download_file_name}.lock"
                    with FileLock(lock_file):
                        with open(json_download_file_name, 'r') as f:
                            current_data = json.load(f)
                        current_data.update(new_entry)
                        with open(json_download_file_name, 'w') as f:
                            json.dump(current_data, f)
                    
                    with open(os.path.join(new_folder_path, 'direct_annotation.json'), 'w') as f:
                        json.dump(direct_search_results, f)
                except Exception as e:
                    print(f"Error saving results for item {i}: {str(e)}")
            except Exception as e:
                print(f"Error processing results for item {i}: {str(e)}")
        
        print(f"Processed item {i} in {time.time() - start_time:.2f} seconds")
        
        # Smaller delay between content processing
        time.sleep(random.uniform(0.5, 1.5))
        
if __name__ == '__main__':
    main()