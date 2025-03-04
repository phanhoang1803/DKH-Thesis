#!/usr/bin/env python
# coding: utf-8

import time 
import argparse
import requests
import os
import PIL
import shutil
from PIL import Image
import imghdr
from bs4 import BeautifulSoup
import bs4
import time
from googleapiclient.discovery import build
from google.cloud import vision
import io
import os
from bs4 import NavigableString
import json
from utils import get_captions_from_page, save_html
import concurrent.futures as cf
from collections import defaultdict
import tqdm
# import fasttext

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download dataset for inverse search queries')
    parser.add_argument('--save_folder_path', type=str, default='queries_dataset_cosmos',
                        help='location where to download data')
    parser.add_argument('--google_cred_json', type=str, default='application_default_credentials.json',
                        help='json file for credentials')
    parser.add_argument('--google_api_key', type=str, default='',
                        help='api_key for the custom search engine')
                        
    parser.add_argument('--google_cse_id', type=str, default='',
                        help='custom search engine id')  
                        
    parser.add_argument('--split_type', type=str, default='merged_balanced',
                        help='which split to use in the NewsCLIP dataset')
    parser.add_argument('--sub_split', type=str, default='test',
                        help='which split to use from train,val,test splits')
                        
    parser.add_argument('--how_many_queries', type=int, default=10,
                        help='how many query to issue for each item - each query is 10 images')
    parser.add_argument('--continue_download', type=int, default=1,
                        help='whether to continue download or start from 0 - should be 0 or 1')

    parser.add_argument('--how_many', type=int, default=-1,
                        help='how many items to query and download, 0 means download untill the end')
                        
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='where to end, if not specified, will be inferred from how_many')
    parser.add_argument('--start_idx', type=int, default=-1,
                        help='where to start, if not specified will be inferred from the current saved json or 0 otherwise')

    parser.add_argument('--hashing_cutoff', type=int, default=15,
                        help='threshold used in hashing')
                        
    parser.add_argument('--max_workers', type=int, default=4,
                        help='maximum number of parallel workers')
    
    parser.add_argument('--skip_existing', action="store_true",
                        help='skip processing if output files already exist')
    
    return parser.parse_args()

def detect_web(path, how_many_queries):
    """Detects web annotations given an image."""
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.web_detection(image=image, max_results=how_many_queries)
    annotations = response.web_detection
    return annotations

def process_page_task(task):
    """Process a single page task with error handling"""
    match_type, img_url, page_url, page, save_folder_path, file_save_counter, hashing_cutoff = task
    try:
        # Get captions and process the page
        caption, title, code, req = get_captions_from_page(img_url, page_url)
                    
        if title is None: 
            title = ''   
        page_title = page.page_title if page.page_title else ''
        if len(title) < len(page_title.lstrip().rstrip()):
            title = page_title
            
        # Save HTML content
        saved_html_flag = save_html(req, os.path.join(save_folder_path, str(file_save_counter) + '.txt'))     
        if saved_html_flag:            
            html_path = os.path.join(save_folder_path, str(file_save_counter) + '.txt')
        else:
            html_path = ''
            
        # Process entry based on caption availability
        if caption:
            new_entry = {'page_link': page_url, 'image_link': img_url, 'title': title, 'caption': caption, 'html_path': html_path}  
        else:
            # Try again with hashing if no caption found
            caption, title, code, req = get_captions_from_page(img_url, page_url, req, hashing_cutoff)
            if caption:
                new_entry = {'page_link': page_url, 'image_link': img_url, 'caption': caption, 'html_path': html_path, 'matched_image': 1}
            else:            
                new_entry = {'page_link': page_url, 'image_link': img_url, 'html_path': html_path}  
        
        if title: 
            new_entry['title'] = title    
        return (match_type, new_entry)
    
    except Exception as e:
        print(f"Error in process_page_task: {str(e)}")
        return None

def get_inverse_search_annotation(web_annotations, id_in_clip, save_folder_path, hashing_cutoff, max_workers=4):
    """Process search results in parallel"""
    file_save_counter = -1

    annotations = {}
    best_guess_lbl = []
    entities = []
    entities_scores = []
    all_fully_matched_captions = []
    all_partially_matched_captions = []
    
    # Image matches with no captions 
    all_partially_matched_no_caption = []
    all_fully_matched_no_caption = []

    # Extract entities and labels
    for entity in web_annotations.web_entities:
        if len(entity.description) > 0:
            entities.append(entity.description)
            entities_scores.append(entity.score)

    if web_annotations.best_guess_labels:
        for label in web_annotations.best_guess_labels:
            best_guess_lbl.append(label.label)
    
    # Setup for parallel processing - collect all tasks
    page_tasks = []
    for page in web_annotations.pages_with_matching_images:
        file_save_counter += 1
        if page.full_matching_images:
            for image_url in page.full_matching_images:
                page_tasks.append(("full", image_url.url, page.url, page, save_folder_path, file_save_counter, hashing_cutoff))
        elif page.partial_matching_images:
            for image_url in page.partial_matching_images:
                page_tasks.append(("partial", image_url.url, page.url, page, save_folder_path, file_save_counter, hashing_cutoff))
    
    # Process pages in parallel using ThreadPoolExecutor
    results = []
    if page_tasks:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_page_task, task): task
                for task in page_tasks
            }
            
            try:
                for future in cf.as_completed(futures, timeout=60):  # Global timeout
                    try:
                        result = future.result(timeout=30)  # Timeout per task
                        if result:
                            results.append(result)
                    except Exception as e:
                        task = futures[future]
                        print(f"Error processing task {task[5]}: {str(e)}")
            
            except KeyboardInterrupt:
                print("🛑 User interrupted! Shutting down all threads...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            
            except Exception as e:
                print(f"🔥 Critical error: {str(e)}. Forcing shutdown.")
                executor.shutdown(wait=False, cancel_futures=True)
    
    # Organize results into appropriate categories
    for result in results:
        match_type, entry = result
        if 'caption' in entry:
            if match_type == "full":
                all_fully_matched_captions.append(entry)
            else:
                all_partially_matched_captions.append(entry)
        else:
            if match_type == "full":
                all_fully_matched_no_caption.append(entry)
            else:
                all_partially_matched_no_caption.append(entry)
    
    # If there is no entities or captions (any textual description), return none 
    if len(entities) == 0 and len(best_guess_lbl) == 0 and len(all_fully_matched_captions) == 0 and len(all_partially_matched_captions) == 0:
        return {}
        
    annotations = {
        'entities': entities,
        'entities_scores': entities_scores, 
        'best_guess_lbl': best_guess_lbl, 
        'all_fully_matched_captions': all_fully_matched_captions, 
        'all_partially_matched_captions': all_partially_matched_captions, 
        'partially_matched_no_text': all_partially_matched_no_caption, 
        'fully_matched_no_text': all_fully_matched_no_caption
    }
    return annotations

def save_json_file(file_path, dict_file, cur_id_in_clip, saved_errors_file, all_inverse_annotations_idx=None):
    """Save JSON file with error handling"""
    if all_inverse_annotations_idx is not None:
        with open(file_path, 'r') as fp:
            old_idx_file = json.load(fp)
    
    try:
        # Use file lock for thread safety
        from filelock import FileLock
        lock_file = f"{file_path}.lock"
        with FileLock(lock_file):
            with open(file_path, 'w') as db_file:
                json.dump(dict_file, db_file)
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        saved_errors_file.write(f"{cur_id_in_clip}\n")
        saved_errors_file.flush()
        
        if all_inverse_annotations_idx is not None:
            with open(file_path, 'w') as db_file:
                json.dump(old_idx_file, db_file)

def main():
    args = parse_arguments()
    
    # Set up Google API credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_cred_json
    
    # Initialize files and paths
    full_save_path = os.path.join(args.save_folder_path, 'inverse_search', args.sub_split)
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)

    # File for saving errors in saving
    if os.path.isfile(os.path.join(full_save_path, 'unsaved.txt')) and args.continue_download:
        saved_errors_file = open(os.path.join(full_save_path, 'unsaved.txt'), "a")
    else:
        saved_errors_file = open(os.path.join(full_save_path, 'unsaved.txt'), "w")

    # File for keys with no annotations
    if os.path.isfile(os.path.join(full_save_path, 'no_annotations.txt')) and args.continue_download:
        no_annotations_file = open(os.path.join(full_save_path, 'no_annotations.txt'), "a")
    else:
        no_annotations_file = open(os.path.join(full_save_path, 'no_annotations.txt'), "w")
        
    # JSON file containing the index and path of all downloaded items so far
    json_download_file_name = os.path.join(full_save_path, args.sub_split + '.json')

    # Continue using the current saved json file or start a new file
    if os.path.isfile(json_download_file_name) and os.access(json_download_file_name, os.R_OK) and args.continue_download:
        with open(json_download_file_name, 'r') as fp:
            all_inverse_annotations_idx = json.load(fp)
    else:
        with io.open(json_download_file_name, 'w') as db_file:
            db_file.write(json.dumps({}))
        with io.open(json_download_file_name, 'r') as db_file:
            all_inverse_annotations_idx = json.load(db_file)
    
    # Load dataset
    cosmos_data = json.load(open('test_dataset_cosmos/public_test_acm.json'))
    
    # Determine processing range
    start_counter = (args.start_idx if args.start_idx != -1 
                    else (int(list(all_inverse_annotations_idx.keys())[-1])+1 
                          if all_inverse_annotations_idx else 0))
    
    end_counter = (args.end_idx if args.end_idx > 0 
                  else (start_counter + args.how_many if args.how_many > 0 
                        else len(cosmos_data)))
    
    print("==========")
    print(f"subset to download is: {args.sub_split}")
    print(f"Starting from index: {start_counter:5d}")
    print(f"Ending at index: {end_counter:5d}")
    print(f"Using {args.max_workers} parallel workers for each item's processing")
    if args.continue_download == 1:
        print(f"Continue download on file: {json_download_file_name}")
    else:
        print(f"Start download from: {start_counter} with creating a new file: {json_download_file_name}")
    print("==========")
    
    # Sequential processing of items with progress bar
    for i in tqdm.tqdm(range(start_counter, end_counter), desc="Processing items"):
        print(f"Processing item {i}")
        
        if i >= len(cosmos_data):
            break
        
        # Skip if already processed and skip_existing is set
        if args.skip_existing:
            result_path = os.path.join(full_save_path, str(i), 'inverse_annotation.json')
            if os.path.exists(result_path):
                continue
        
        start_time = time.time()
        
        try:
            
            # Get item information
            ann = cosmos_data[i]
            image_path = os.path.join('test_dataset_cosmos', ann["img_local_path"])
            new_folder_path = os.path.join(full_save_path, str(i))
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Detect web annotations
            result = detect_web(image_path, how_many_queries=args.how_many_queries)
            
            # Process annotations in parallel
            inverse_search_results = get_inverse_search_annotation(
                result, i, new_folder_path, args.hashing_cutoff, args.max_workers
            )
            
            new_json_file_path = os.path.join(new_folder_path, 'inverse_annotation.json')
            
            # Save results if found
            if inverse_search_results:
                new_entry = {
                    str(i): {
                        'folder_path': new_folder_path
                    }
                }
                
                # Save individual result file
                try:
                    with open(new_json_file_path, 'w') as f:
                        json.dump(inverse_search_results, f)
                except Exception as e:
                    print(f"Error saving individual result file for item {i}: {str(e)}")
                
                # Update index file with thread safety
                try:
                    from filelock import FileLock
                    lock_file = f"{json_download_file_name}.lock"
                    with FileLock(lock_file):
                        with open(json_download_file_name, 'r') as f:
                            current_data = json.load(f)
                        current_data.update(new_entry)
                        with open(json_download_file_name, 'w') as f:
                            json.dump(current_data, f)
                except Exception as e:
                    print(f"Error updating index file for item {i}: {str(e)}")
                    saved_errors_file.write(f"{i}\n")
                    saved_errors_file.flush()
            else:
                # Note the lack of annotations
                no_annotations_file.write(f"{i}\n")
                no_annotations_file.flush()
            
            print(f"Processed item {i} in {time.time() - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("User interrupted! Exiting...")
            break
            
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
    
    # Cleanup
    saved_errors_file.close()
    no_annotations_file.close()
    
    print("Processing completed!")

if __name__ == '__main__':
    main()