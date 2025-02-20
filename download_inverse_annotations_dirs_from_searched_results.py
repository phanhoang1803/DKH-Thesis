#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import io
import fasttext
from utils import get_captions_from_page, save_html
import concurrent.futures as cf
from collections import defaultdict
import tqdm
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process existing inverse search results')
    parser.add_argument('--save_folder_path', type=str, default='queries_dataset',
                        help='location where to save processed data')
    parser.add_argument('--split_type', type=str, default='merged_balanced',
                        help='which split to use in the NewsCLIP dataset')
    parser.add_argument('--sub_split', type=str, default='test',
                        help='which split to use from train,val,test splits')
    parser.add_argument('--continue_download', type=int, default=0,
                        help='whether to continue processing or start from 0')
    parser.add_argument('--how_many', type=int, default=-1,
                        help='how many items to process, -1 means process until the end')
    parser.add_argument('--existing_results_path', type=str, required=True,
                        help='path to JSON file containing existing inverse search results')
    parser.add_argument('--hashing_cutoff', type=int, default=15,
                        help='threshold used in hashing')
    parser.add_argument('--skip_existing', action="store_true",
                        help='skip processing if output files already exist')
    parser.add_argument('--start_idx', type=int, default=-1,
                        help='where to start processing')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='where to end processing')
    
    return parser.parse_args()

def init_files_and_paths(args):
    """Initialize necessary files and paths"""
    full_save_path = os.path.join(args.save_folder_path, args.split_type, 'inverse_search', args.sub_split)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Initialize log files
    files_info = {}
    for filename in ['unsaved.txt', 'no_annotations.txt']:
        file_path = os.path.join(full_save_path, filename)
        mode = "a" if os.path.isfile(file_path) and args.continue_download else "w"
        files_info[filename] = open(file_path, mode)
    
    # Initialize or load index file
    json_download_file_name = os.path.join(full_save_path, args.sub_split + '.json')
    if os.path.isfile(json_download_file_name) and os.access(json_download_file_name, os.R_OK) and args.continue_download:
        with open(json_download_file_name, 'r') as fp:
            all_inverse_annotations_idx = json.load(fp)
    else:
        all_inverse_annotations_idx = {}
        with open(json_download_file_name, 'w') as db_file:
            json.dump({}, db_file)
    
    return full_save_path, json_download_file_name, all_inverse_annotations_idx, files_info

def process_url_pair(args):
    """Process a single URL pair with error handling"""
    img_url, page_url, counter, save_folder_path, hashing_cutoff = args
    try:
        caption, title, code, req = get_captions_from_page(img_url, page_url)
        
        if title is None:
            title = ''
        
        saved_html_flag = save_html(req, os.path.join(save_folder_path, f"{counter}.txt"))
        html_path = os.path.join(save_folder_path, f"{counter}.txt") if saved_html_flag else ''
        
        new_entry = {
            'page_link': page_url,
            'image_link': img_url,
            'html_path': html_path,
            'title': title
        }
        
        if caption:
            new_entry['caption'] = caption
        else:
            caption, title, code, req = get_captions_from_page(img_url, page_url, req, hashing_cutoff)
            if caption:
                new_entry['caption'] = caption
                new_entry['matched_image'] = 1
            if title:
                new_entry['title'] = title
        
        return new_entry
        
    except Exception as e:
        print(f"Error processing URL pair {counter}: {str(e)}")
        return None

def filter_non_english(entries, lang_model):
    """Filter out non-English entries based on title"""
    filtered_entries = []
    for entry in entries:
        if 'title' in entry and entry['title']:
            lang_pred = lang_model.predict(entry['title'].replace("\n", " "))
            if lang_pred[0][0] == '__label__en':
                filtered_entries.append(entry)
        else:
            filtered_entries.append(entry)  # Keep entries without title
    return filtered_entries

def process_one_item(item_id, result_data, save_folder_path, hashing_cutoff):
    """Process a single item's search results in parallel"""
    annotations = {
        'entities': result_data.get('entities', []),
        'all_matched_captions': [],
        'matched_no_text': []
    }
    
    # Prepare URL pairs for processing
    url_pairs = []
    counter = 0
    
    for img_url, page_url in result_data.get('links_inv_search', []):
        url_pairs.append((img_url, page_url, counter, save_folder_path, hashing_cutoff))
        counter += 1
    
    if not url_pairs and not annotations['entities']:
        return {}
    
    # Process URL pairs in parallel
    results = []
    if url_pairs:
        with cf.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_url_pair, url_pair): url_pair
                for url_pair in url_pairs
            }
            
            try:
                for future in cf.as_completed(futures, timeout=60):
                    try:
                        result = future.result(timeout=30)
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing future: {str(e)}")
                        continue
            
            except (KeyboardInterrupt, Exception) as e:
                print(f"{'🛑 User interrupted!' if isinstance(e, KeyboardInterrupt) else '🔥 Critical error: ' + str(e)}")
                executor.shutdown(wait=False, cancel_futures=True)
                if isinstance(e, KeyboardInterrupt):
                    raise
    
    return results

def save_json_file(file_path, dict_file, cur_id_in_clip, saved_errors_file, all_inverse_annotations_idx=None):
    """Save JSON file with error handling"""
    if all_inverse_annotations_idx is not None:
        with open(file_path, 'r') as fp:
            old_idx_file = json.load(fp)
    
    try:
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
    
    # Initialize model and load data
    lang_model = fasttext.load_model('lid.176.bin')
    with open(args.existing_results_path, 'r') as f:
        existing_results = json.load(f)
    
    # Initialize files and paths
    full_save_path, json_download_file_name, all_inverse_annotations_idx, files_info = init_files_and_paths(args)
    
    # Determine processing range
    start_counter = (args.start_idx if args.start_idx != -1 
                    else (int(list(all_inverse_annotations_idx.keys())[-1])+1 
                          if all_inverse_annotations_idx else 0))
    
    end_counter = (args.end_idx if args.end_idx > 0 
                  else (start_counter + args.how_many if args.how_many > 0 
                        else len(existing_results)))
    
    print(f"Processing items from {start_counter} to {end_counter}")
    
    # Main processing loop
    for item_id in tqdm.tqdm(range(start_counter, end_counter)):
        if str(item_id) not in existing_results:
            continue
        
        if args.skip_existing:
            result_path = os.path.join(full_save_path, str(item_id), 'inverse_annotation.json')
            if os.path.exists(result_path):
                continue
        
        start_time = time.time()
        new_folder_path = os.path.join(full_save_path, str(item_id))
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Process the item
        result_data = existing_results[str(item_id)]
        results = process_one_item(
            item_id, result_data, new_folder_path, 
            args.hashing_cutoff
        )
        
        if results:
            # Filter non-English results
            # filtered_results = filter_non_english(results, lang_model)
            
            # Organize results
            processed_results = {
                'entities': result_data.get('entities', []),
                'all_matched_captions': [],
                'matched_no_text': []
            }
            
            # for result in filtered_results:            
            for result in results:
                if 'caption' in result:
                    processed_results['all_matched_captions'].append(result)
                else:
                    processed_results['matched_no_text'].append(result)
            
            if processed_results['entities'] or processed_results['all_matched_captions']:
                # Save to index file
                new_entry = {str(item_id): {'folder_path': new_folder_path}}
                all_inverse_annotations_idx.update(new_entry)
                save_json_file(
                    json_download_file_name, 
                    all_inverse_annotations_idx, 
                    item_id, 
                    files_info['unsaved.txt'], 
                    all_inverse_annotations_idx
                )
                
                # Save individual result
                result_path = os.path.join(new_folder_path, 'inverse_annotation.json')
                save_json_file(result_path, processed_results, item_id, files_info['unsaved.txt'])
            else:
                files_info['no_annotations.txt'].write(f"{item_id}\n")
                files_info['no_annotations.txt'].flush()
        else:
            files_info['no_annotations.txt'].write(f"{item_id}\n")
            files_info['no_annotations.txt'].flush()
        
        print(f"Processed item {item_id} in {time.time() - start_time:.2f} seconds")
    
    # Cleanup
    for file_handle in files_info.values():
        file_handle.close()
    
    print("Processing completed!")

if __name__ == '__main__':
    main()