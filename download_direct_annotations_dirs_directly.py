#!/usr/bin/env python
# coding: utf-8

import argparse
from collections import defaultdict
import requests
import os
import PIL
import shutil
from PIL import Image
import imghdr
from bs4 import BeautifulSoup
import bs4
import time
import io
import json
import concurrent.futures as cf
from functools import partial
import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import random
import urllib.parse
import logging
from typing import List, Dict, Any, Tuple, Optional
from filelock import FileLock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("google_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import utils functions from original code
from utils import get_captions_from_page, save_html, download_and_save_image

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download dataset for direct search queries')
    parser.add_argument('--save_folder_path', type=str, default='queries_datasett',
                        help='location where to download data')
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
    
    # New arguments for proxy settings
    parser.add_argument('--use_proxies', action="store_true",
                        help='Use rotating proxies for requests')
    parser.add_argument('--proxy_list_path', type=str, default=None,
                        help='Path to a file containing proxy list (one per line)')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum number of retries for failed requests')
    
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
            logger.error(f"Error parsing URL {url}: {e}")
    
    return filtered_urls

def get_proxies(proxy_list_path: Optional[str] = None) -> List[str]:
    """Get a list of proxies either from file or from Geonode API"""
    if proxy_list_path and os.path.exists(proxy_list_path):
        try:
            with open(proxy_list_path, 'r') as f:
                proxies = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(proxies)} proxies from file")
            return proxies
        except Exception as e:
            logger.error(f"Error loading proxies from file: {e}")
    
    # Fallback to Geonode API
    try:
        url = "https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=lastChecked&sort_type=desc&filterUpTime=90&protocols=http,https"
        response = requests.get(url, timeout=10)
        data = response.json()
        proxies = []
        
        for proxy in data.get('data', []):
            protocol = proxy.get('protocols')[0].lower()
            ip = proxy.get('ip')
            port = proxy.get('port')
            proxy_str = f"{protocol}://{ip}:{port}"
            proxies.append(proxy_str)
            
        logger.info(f"Fetched {len(proxies)} proxies from Geonode")
        return proxies
    except Exception as e:
        logger.error(f"Error fetching proxies: {e}")
        return []

def validate_proxy(proxy: str) -> bool:
    """Check if proxy is working"""
    try:
        test_url = "https://www.google.com"
        response = requests.get(
            test_url, 
            proxies={"http": proxy, "https": proxy},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def get_working_proxies(proxy_list_path: Optional[str] = None, max_proxies: int = 10) -> List[str]:
    """Get a list of working proxies"""
    all_proxies = get_proxies(proxy_list_path)
    working_proxies = []
    
    with cf.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(validate_proxy, all_proxies)
        
        for proxy, is_valid in zip(all_proxies, results):
            if is_valid:
                working_proxies.append(proxy)
                if len(working_proxies) >= max_proxies:
                    break
                
    logger.info(f"Found {len(working_proxies)} working proxies")
    return working_proxies

def get_random_user_agent() -> str:
    """Generate a random user agent"""
    try:
        # Try importing fake_useragent if installed
        from fake_useragent import UserAgent
        ua = UserAgent()
        return ua.random
    except:
        # Fallback user agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15'
        ]
        return random.choice(user_agents)

def get_cookies() -> Dict:
    """Try to load cookies from a file or return empty dict"""
    try:
        if os.path.exists('google_cookies.json'):
            with open('google_cookies.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading cookies: {e}")
    return {}

def save_cookies(cookies) -> None:
    """Save cookies to a file"""
    try:
        with open('google_cookies.json', 'w') as f:
            json.dump(dict(cookies), f)
    except Exception as e:
        logger.error(f"Error saving cookies: {e}")

def google_search_with_requests(query: str, how_many_queries: int = 1, 
                                use_proxies: bool = False, proxy_list_path: Optional[str] = None, 
                                max_retries: int = 5) -> List[Dict]:
    """Search Google Images with rotating proxies and anti-detection techniques"""
    
    working_proxies = []
    if use_proxies:
        working_proxies = get_working_proxies(proxy_list_path)
        if not working_proxies:
            logger.warning("No working proxies found. Will try without proxy.")
            working_proxies = [None]  # Try without proxy as fallback
    else:
        working_proxies = [None]  # Don't use proxies
    
    cookies = get_cookies()
    results_list = []
    
    for i in range(how_many_queries):
        start = i * 10 + 1
        retries = 0
        search_results = {'items': []}
        
        while retries < max_retries and len(search_results['items']) < 10:
            # Select random proxy
            proxy = random.choice(working_proxies)
            proxies = {"http": proxy, "https": proxy} if proxy else None
            
            # Random delay to mimic human behavior
            time.sleep(random.uniform(1, 3))
            
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            # Try both exact match (with quotes) and broad match
            for search_type in ["exact", "broad"]:
                search_query = f'"{query}"' if search_type == "exact" else query
                
                # Additional parameters for broad search
                additional_params = {}
                if search_type == "broad":
                    additional_params = {
                        'sort': 'date:r:20100101:20161231'
                    }
                
                # Randomize search parameters slightly
                search_params = {
                    'q': search_query,
                    'tbm': 'isch',  # For image search
                    'hl': 'en',
                    'gl': random.choice(['us', 'uk', 'ca']),  # Random region
                    'start': start,
                    **additional_params
                }
                
                try:
                    logger.info(f"Sending {search_type} request with proxy: {proxy}")
                    response = requests.get(
                        'https://www.google.com/search', 
                        headers=headers,
                        params=search_params,
                        proxies=proxies,
                        cookies=cookies,
                        timeout=15  # Increased timeout for proxy requests
                    )
                    
                    # Save cookies for potential future use
                    save_cookies(response.cookies)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for indications of blocking
                        if 'unusual traffic' in response.text.lower() or 'captcha' in response.text.lower():
                            logger.warning(f"Google detected automated traffic for {search_type}. Proxy may be blocked.")
                            if proxy in working_proxies and proxy is not None:
                                working_proxies.remove(proxy)
                            continue
                        
                        # Extract all elements with data-lpage attribute (original URLs)
                        elements = soup.find_all(attrs={"data-lpage": True})
                        logger.info(f"Found {len(elements)} raw results for {search_type}")
                        
                        # Extract other attributes for each item
                        for element in elements:
                            try:
                                # Try to find parent element that contains the image
                                parent = element.find_parent('div', class_='isv-r')
                                
                                # Find the image element
                                img_element = None
                                if parent:
                                    img_element = parent.find('img')
                                else:
                                    img_element = element.find('img')
                                
                                # Extract image source
                                img_src = None
                                img_page_url = None
                                
                                h3 = element.find('h3').find('a')
                                print(h3)
                                
                                if img_page_url and not img_src and img_page_url.startswith('/imgres'):
                                    try:
                                        # Add the Google domain if it's a relative URL
                                        if img_page_url.startswith('/'):
                                            img_page_url = f"https://www.google.com{img_page_url}"
                                        
                                        logger.info(f"Following redirect to extract image URL: {img_page_url}")
                                        
                                        # Use a different user agent to avoid detection
                                        redirect_headers = {
                                            'User-Agent': get_random_user_agent(),
                                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                                            'Referer': 'https://www.google.com/',
                                        }
                                        
                                        # Make a request to the Google image page
                                        redirect_response = requests.get(img_page_url, headers=redirect_headers, allow_redirects=False)
                                        
                                        # Check if we got a redirect
                                        if redirect_response.status_code in (301, 302, 303, 307, 308) and 'Location' in redirect_response.headers:
                                            # The real image URL is in the Location header
                                            img_src = redirect_response.headers['Location']
                                            logger.info(f"Extracted image URL from redirect: {img_src}")
                                        else:
                                            # Try to parse the imgurl parameter from the URL
                                            if 'imgurl=' in img_page_url:
                                                img_src = img_page_url.split('imgurl=')[1].split('&')[0]
                                                img_src = urllib.parse.unquote(img_src)
                                                logger.info(f"Extracted image URL from URL parameter: {img_src}")
                                    except Exception as e:
                                        logger.error(f"Error following redirect: {e}")
                                
                                # Extract title
                                title_element = element.find('h3') or element.find('div', class_='iKjWAf')
                                title = title_element.get_text() if title_element else ""
                                
                                # Extract the domain/display link
                                domain_element = element.find('div', class_='ptes9b') or element.find('div', class_='Xxy7Vb')
                                domain = ""
                                if domain_element:
                                    domain_span = domain_element.find('span')
                                    domain = domain_span.get_text() if domain_span else ""
                                
                                # Build an item similar to Google API response format
                                item = {
                                    'link': img_src,
                                    'title': title,
                                    'displayLink': domain,
                                    'image': {
                                        'contextLink': element["data-lpage"]
                                    }
                                }
                                
                                # print(item)
                                
                                # Add to search results if not already present
                                if img_src:  # Only add if we found an image source
                                    # Check if this image URL is already in our results
                                    if not any(i['link'] == img_src for i in search_results['items']):
                                        search_results['items'].append(item)
                            except Exception as e:
                                logger.error(f"Error processing element: {e}")
                                continue
                    else:
                        logger.warning(f"Request failed with status code: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error during request: {e}")
                    # Remove failing proxy
                    if proxy in working_proxies and proxy is not None:
                        working_proxies.remove(proxy)
                
                # If we've run out of proxies, try without one
                if not working_proxies:
                    working_proxies = [None]
            
            # Check if we got enough results or need to retry
            if len(search_results['items']) >= 10:
                break
                
            retries += 1
            logger.info(f"Retry {retries}/{max_retries} - Got {len(search_results['items'])} items so far")
        
        # Add searchInformation to match Google API format
        search_results['searchInformation'] = {
            'totalResults': str(len(search_results['items']))
        }
        
        results_list.append(search_results)
        
        # Log what we found
        logger.info(f"Query {i+1}/{how_many_queries} complete. Found {len(search_results['items'])} items.")
    
    return results_list

def init_files_and_paths(args):
    """Initialize files and paths needed for the script"""
    full_save_path = os.path.join(args.save_folder_path, args.split_type, 'direct_search', args.sub_split)
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

def process_single_item(item_data):
    """Process a single search result item"""
    item, counter, save_folder_path = item_data
    image = {}
    
    # Basic information extraction
    for key, target in [('link', 'img_link'), ('displayLink', 'domain')]:
        if key in item:
            image[target] = item[key]
    
    if 'image' in item and 'contextLink' in item['image']:
        image['page_link'] = item['image']['contextLink']
    if 'snippet' in item:
        image['snippet'] = item['snippet']

    # Download image
    if not download_and_save_image(item['link'], save_folder_path, str(counter)):
        return None

    image['image_path'] = os.path.join(save_folder_path, f"{counter}.jpg")

    try:
        caption, title, code, req = get_captions_from_page(
            item['link'], 
            item['image']['contextLink']
        )
    except Exception as e:
        print(f'Error in getting captions for item {counter}: {str(e)}')
        return None

    # Save HTML
    if save_html(req, os.path.join(save_folder_path, f"{counter}.txt")):
        image['html_path'] = os.path.join(save_folder_path, f"{counter}.txt")

    if code and code[0] in ['4', '5']:
        image['is_request_error'] = True

    # Process title
    item_title = item.get('title', '') or ''
    title = title if title is not None else ''
    image['page_title'] = title if len(title) > len(item_title.strip()) else item_title

    # Process caption
    if caption:
        image['caption'] = caption
        return ('with_captions', image)
    
    try:
        caption, title, code, req = get_captions_from_page(
            item['link'],
            item['image']['contextLink'],
            req,
            # args.hashing_cutoff
        )
    except Exception as e:
        print(f'Error in getting captions for item {counter} (second attempt): {str(e)}')
        return None

    if caption:
        image['caption'] = caption
        return ('matched_tags', image)
    
    return ('no_captions', image)

def get_direct_search_annotation(search_results_lists, save_folder_path):
    """Process search results in parallel"""
    items_to_process = []
    counter = 0
    
    for result_list in search_results_lists:
        if 'items' in result_list:
            for item in result_list['items']:
                items_to_process.append((item, counter, save_folder_path))
                counter += 1

    if not items_to_process:
        return {}

    results = defaultdict(list)
    
    # Use a context manager for ProcessPoolExecutor
    with cf.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_item, item_data): item_data
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
            executor.shutdown(wait=False, cancel_futures=True)  # 🚀 Force stop all workers
            raise  # Re-raise KeyboardInterrupt
        
        except Exception as e:
            print(f"🔥 Critical error: {str(e)}. Forcing shutdown.")
            executor.shutdown(wait=False, cancel_futures=True)  # 🚀 Force stop all workers

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
    visual_news_data_mapping = json.load(open("test_dataset/visual_news_test.json"))
    clip_data = json.load(open("test_dataset/news_clippings_test.json"))
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
    indices.sort()
    print(f"Processing items from {indices[0]} to {indices[-1]}")
    
    # Main processing loop
    for i in tqdm.tqdm(indices):
        if args.skip_existing:
            if os.path.exists(os.path.join(full_save_path, str(i))):
                # If the folder exists, and the direct_annotation.json file exists, skip the item
                if os.path.exists(os.path.join(full_save_path, str(i), 'direct_annotation.json')):
                    continue
        
        start_time = time.time()
        
        try:
            ann = clip_data_annotations[i]
            text_query = visual_news_data_mapping[str(ann["id"])]["caption"]
        except Exception as e:
            print(f"Skipping item {i} due to error: {str(e)}")
            continue
            
        new_folder_path = os.path.join(full_save_path, str(i))
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Process single query using our new Google Search function
        result = google_search_with_requests(
            query=text_query,
            how_many_queries=args.how_many_queries,
            use_proxies=args.use_proxies,
            proxy_list_path=args.proxy_list_path,
            max_retries=args.max_retries
        )
        
        direct_search_results = get_direct_search_annotation(result, new_folder_path)
        
        # Save results
        if direct_search_results:
            new_entry = {
                str(i): {
                    'image_id_in_visualNews': ann["image_id"],
                    'text_id_in_visualNews': ann["id"],
                    'folder_path': new_folder_path
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
        
        print(f"Processed item {i} in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()