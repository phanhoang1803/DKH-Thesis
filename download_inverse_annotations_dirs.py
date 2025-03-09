# #!/usr/bin/env python
# # coding: utf-8

# # In[3]:
# import time 
# import argparse
# import requests
# import os
# import PIL
# import shutil
# from PIL import Image
# import imghdr
# from bs4 import BeautifulSoup
# import bs4
# import time
# from googleapiclient.discovery import build
# from google.cloud import vision
# import io
# import os
# from bs4 import NavigableString
# import json
# from utils import get_captions_from_page, save_html
# # import fasttext

# parser = argparse.ArgumentParser(description='Download dataset for inverse search queries')
# parser.add_argument('--save_folder_path', type=str, default='queries_dataset',
#                     help='location where to download data')
# parser.add_argument('--google_cred_json', type=str, default='credentials.json',
#                     help='json file for credentials')
# parser.add_argument('--google_api_key', type=str, default='',
#                     help='api_key for the custom search engine')
                    
# parser.add_argument('--google_cse_id', type=str, default='',
#                     help='custom search engine id')  
                    
# parser.add_argument('--split_type', type=str, default='merged_balanced',
#                     help='which split to use in the NewsCLIP dataset')
# parser.add_argument('--sub_split', type=str, default='test',
#                     help='which split to use from train,val,test splits')
                    
# parser.add_argument('--how_many_queries', type=int, default=20,
#                     help='how many query to issue for each item - each query is 10 images')
# parser.add_argument('--continue_download', type=int, default=1,
#                     help='whether to continue download or start from 0 - should be 0 or 1')

# parser.add_argument('--how_many', type=int, default=-1,
#                     help='how many items to query and download, 0 means download untill the end')
                    
# parser.add_argument('--end_idx', type=int, default=-1,
#                     help='where to end, if not specified, will be inferred from how_many')
# parser.add_argument('--start_idx', type=int, default=-1,
#                     help='where to start, if not specified will be inferred from the current saved json or 0 otherwise')

# parser.add_argument('--hashing_cutoff', type=int, default=15,
#                     help='threshold used in hashing')
                    
# args = parser.parse_args()

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_cred_json
# my_api_key = args.google_api_key
# my_cse_id =  args.google_cse_id

# PRETRAINED_MODEL_PATH = 'lid.176.bin'
# # model = fasttext.load_model(PRETRAINED_MODEL_PATH)

# full_save_path = os.path.join(args.save_folder_path,args.split_type,'inverse_search',args.sub_split)
# if not os.path.exists(full_save_path):
#     os.makedirs(full_save_path)

# ##file for saving errors in saving..
# if os.path.isfile(os.path.join(full_save_path,'unsaved.txt')) and args.continue_download:
#     saved_errors_file= open(os.path.join(full_save_path,'unsaved.txt'), "a")
# else:
#     saved_errors_file= open(os.path.join(full_save_path,'unsaved.txt'), "w")

# ##file for keys with no annotations..
# if os.path.isfile(os.path.join(full_save_path,'no_annotations.txt')) and args.continue_download:
#     no_annotations_file= open(os.path.join(full_save_path,'no_annotations.txt'), "a")
# else:
#     no_annotations_file= open(os.path.join(full_save_path,'no_annotations.txt'), "w")
    

# # json file containing the index and path of all downloaded items so far -- does not contain the actual annotations.
# json_download_file_name = os.path.join(full_save_path,args.sub_split+'.json')

# #continue using the current saved json file -- don't start a new file -- load the saved dict 
# if os.path.isfile(json_download_file_name) and os.access(json_download_file_name, os.R_OK) and args.continue_download:
#     with open(json_download_file_name, 'r') as fp:
#         all_inverse_annotations_idx = json.load(fp)
# #start a new file -- start from an empty dict 
# else:
#     with io.open(json_download_file_name, 'w') as db_file:
#         db_file.write(json.dumps({}))
#     with io.open(json_download_file_name, 'r') as db_file:
#         all_inverse_annotations_idx = json.load(db_file)         

# ### load visual_news and CLIP datasets ####
# visual_news_data_mapping = json.load(open("test_dataset/visual_news_test.json"))
# clip_data = json.load(open("test_dataset/news_clippings_test.json"))
# clip_data_annotations = clip_data["annotations"]

# ### find where to start from the clip_data ###
# ### if there is a start index, start from it ###
# ### if continue_download and so start index, get the last idx of the last downloaded item. ###
# ### otherwise, start from 0 ###
# if args.start_idx != -1:
#     start_counter = args.start_idx
# else:
#     if all_inverse_annotations_idx:
#         start_counter = int(list(all_inverse_annotations_idx.keys())[-1])+1
#     else:
#         start_counter = 0
    
# #set the end_counter using the how_many argument
# #if no valid how_many queries, set the end_counter to the end of the annotations list 
# if args.end_idx > 0:
#     end_counter = args.end_idx
# else:   
#     if args.how_many>0: 
#         end_counter = start_counter + args.how_many 
#     else:
#         end_counter = len(clip_data_annotations) 
    
# print("==========")
# print("subset to download is: "+args.split_type+" - "+args.sub_split)
# print("Starting from index: %5d"%start_counter)
# print("Ending at index: %5d"%end_counter)
# if args.continue_download == 1:
#     print("Continue download on file: "+json_download_file_name)
# else:
#     print("Start download from: " + str(start_counter) +" with creating a new file: "+json_download_file_name)
# print("==========") 

# def detect_web(path,how_many_queries):
#     """Detects web annotations given an image."""
#     client = vision.ImageAnnotatorClient()
#     with io.open(path, 'rb') as image_file:
#         content = image_file.read()
#     image = vision.Image(content=content)
#     response = client.web_detection(image=image, max_results=how_many_queries)
#     annotations = response.web_detection
#     return annotations

# def get_captions_and_process(img_url,page_url,page, save_folder_path,file_save_counter):
#     caption,title,code,req = get_captions_from_page(img_url, page_url)
                
#     if title is None: title = ''   
#     page_title = page.page_title if page.page_title else ''
#     if len(title) < len(page_title.lstrip().rstrip()):
#         title = page_title
#     # if title: 
#     #     lang_pred = model.predict(title.replace("\n"," "))
#     #     if lang_pred[0][0] != '__label__en':
#     #         return{}
#     saved_html_flag = save_html(req, os.path.join(save_folder_path,str(file_save_counter)+'.txt'))     
#     if saved_html_flag:            
#         html_path = os.path.join(save_folder_path,str(file_save_counter)+'.txt')
#     else:
#         html_path = ''
        
#     if caption:
#         new_entry = {'page_link':page_url,'image_link':img_url,'title': title, 'caption':caption, 'html_path': html_path}  
#     else:
#         caption,title,code,req = get_captions_from_page(img_url,page_url,req,args.hashing_cutoff)
#         if caption:
#             new_entry = {'page_link':page_url,'image_link':img_url, 'caption':caption, 'html_path': html_path, 'matched_image': 1}
#         else:            
#             new_entry = {'page_link':page_url,'image_link':img_url, 'html_path': html_path}  
#     if title: 
#         new_entry['title'] = title    
#     return new_entry
    
# def get_inverse_search_annotation(web_annotations,id_in_clip,save_folder_path):
#     file_save_counter = -1

#     annotations = {}
#     best_guess_lbl = []
#     entities = []
#     entities_scores = []
#     all_fully_matched_captions=[]
#     all_partially_matched_captions=[]
    
#     #image matches with no captions 
#     all_partially_matched_no_caption=[]
#     all_fully_matched_no_caption=[]

#     for entity in web_annotations.web_entities:
#         if len(entity.description) > 0:
#             entities.append(entity.description)
#             entities_scores.append(entity.score)

#     if web_annotations.best_guess_labels:
#         for label in web_annotations.best_guess_labels:
#             best_guess_lbl.append(label.label)
        
#     for page in web_annotations.pages_with_matching_images: 
#         file_save_counter = file_save_counter + 1
#         new_entry = {}
#         if page.full_matching_images:
#             for image_url in page.full_matching_images:
#                 try:
#                     new_entry = get_captions_and_process(image_url.url, page.url,page,save_folder_path,file_save_counter)
#                 except: 
#                     print("Error in getting captions - id in clip:%5d"%id_in_clip)
#                     continue
#                 if not 'caption' in new_entry.keys(): 
#                     continue 
#                 else:
#                     break 
#             if 'caption' in new_entry.keys():
#                 all_fully_matched_captions.append(new_entry)      
#             elif new_entry:
#                 all_fully_matched_no_caption.append(new_entry)
                        
#         elif page.partial_matching_images:
#             for image_url in page.partial_matching_images:
#                 try:
#                     new_entry = get_captions_and_process(image_url.url, page.url,page,save_folder_path,file_save_counter)
#                 except: 
#                     print("Error in getting captions - id in clip: %5d"%id_in_clip)
#                     continue
#                 if not 'caption' in new_entry.keys(): 
#                     continue 
#                 else:
#                     break                  
#             if 'caption' in new_entry.keys():
#                 all_partially_matched_captions.append(new_entry)
#             elif new_entry:
#                 all_partially_matched_no_caption.append(new_entry)
                    
#     #if there is no entities or captions (any textual description), return none 
#     if len(entities) == 0 and len(best_guess_lbl)==0 and len(all_fully_matched_captions)==0 and len(all_partially_matched_captions) == 0:
#         return {}
#     annotations = {'entities': entities,'entities_scores': entities_scores, 'best_guess_lbl': best_guess_lbl, 'all_fully_matched_captions': all_fully_matched_captions, 'all_partially_matched_captions': all_partially_matched_captions, 'partially_matched_no_text': all_partially_matched_no_caption, 'fully_matched_no_text': all_fully_matched_no_caption} 
#     return annotations 

# def search_and_save_one_query(image_path, id_in_clip, image_id_in_visualNews, text_id_in_visualNews, new_folder_path):
#     global all_inverse_annotations_idx, no_annotations_file 
    
#     result = detect_web(image_path,how_many_queries=args.how_many_queries)
#     inverse_search_results = get_inverse_search_annotation(result,id_in_clip,new_folder_path)
#     new_json_file_path = os.path.join(new_folder_path,'inverse_annotation.json')
    
#     if inverse_search_results:
#         {id_in_clip: {'image_id_in_visualNews':image_id_in_visualNews, 'text_id_in_visualNews': text_id_in_visualNews, 'foldnew_entry = er_path':new_folder_path}}
#         all_inverse_annotations_idx.update(new_entry)
#         save_json_file(json_download_file_name, all_inverse_annotations_idx, id_in_clip, saving_idx_file=True)
#         save_json_file(new_json_file_path, inverse_search_results, id_in_clip)
#     else:
#         #save that there is no annotations for that key... 
#         no_annotations_file.write(str(id_in_clip)+'\n')   
#         no_annotations_file.flush()         

# def save_json_file(file_path, dict_file, cur_id_in_clip, saving_idx_file=False):
#     global all_inverse_annotations_idx, saved_errors_file
#     #load the previous saved file 
#     if saving_idx_file:
#         with open(file_path, 'r') as fp:
#             old_idx_file = json.load(fp)  
#     try:
#         with io.open(file_path, 'w') as db_file:
#             json.dump(dict_file, db_file)
#     except:
#         saved_errors_file.write(str(cur_id_in_clip)+'\n')
#         saved_errors_file.flush()
#         if saving_idx_file:
#             all_inverse_annotations_idx = old_idx_file 
#             with io.open(file_path, 'w') as db_file:
#                 json.dump(old_idx_file, db_file)       

# ### Loop #### 
# for i in range(start_counter,end_counter):    
#     start_time = time.time()
#     ann = clip_data_annotations[i]
#     image_path = os.path.join('test_dataset',visual_news_data_mapping[str(ann["image_id"])]["image_path"])
#     id_in_clip = i 
#     image_id_in_visualNews = ann["image_id"]
#     text_id_in_visualNews = ann["id"]
#     new_folder_path = os.path.join(full_save_path,str(i))
#     if not os.path.exists(new_folder_path):
#         os.makedirs(new_folder_path)
        
#     search_and_save_one_query(image_path, id_in_clip, image_id_in_visualNews, text_id_in_visualNews, new_folder_path) 
#     end_time = time.time()   
#     print("--- Time elapsed for 1 query: %s seconds ---" % (end_time - start_time))        



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
from utils import download_and_save_image, get_captions_from_page, save_html, extract_page_content
import concurrent.futures as cf
from collections import defaultdict
import tqdm
# import fasttext

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download dataset for inverse search queries')
    parser.add_argument('--save_folder_path', type=str, default='queries_dataset',
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
                        
    parser.add_argument('--how_many_queries', type=int, default=15,
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
        
        # Extract content from the page
        page_content = ""
        if req and req.content:
            try:
                soup = BeautifulSoup(req.content.decode('utf-8'), "html.parser")
                page_content = extract_page_content(soup)
            except Exception as content_error:
                print(f"Error extracting content: {str(content_error)}")
            
        # Save HTML content
        saved_html_flag = save_html(req, os.path.join(save_folder_path, str(file_save_counter) + '.txt'))     
        if saved_html_flag:            
            html_path = os.path.join(save_folder_path, str(file_save_counter) + '.txt')
        else:
            html_path = ''
        
        image_path = ""
        if download_and_save_image(img_url, save_folder_path, str(file_save_counter)):
            image_path = os.path.join(save_folder_path, f"{file_save_counter}.jpg")
        
        # Process entry based on caption availability
        if caption:
            new_entry = {'page_link': page_url, 
                         'image_link': img_url, 
                         'title': title, 
                         'caption': caption, 
                         'html_path': html_path,
                         'image_path': image_path,
                         'content': page_content}  
        else:
            # Try again with hashing if no caption found
            caption, title, code, req = get_captions_from_page(img_url, page_url, req, hashing_cutoff)
            if caption:
                new_entry = {'page_link': page_url, 
                             'image_link': img_url, 
                             'title': title, 
                             'caption': caption, 
                             'html_path': html_path,
                             'image_path': image_path,
                             'content': page_content}
            else:            
                new_entry = {'page_link': page_url, 
                             'image_link': img_url, 
                             'html_path': html_path,
                             'image_path': image_path,
                             'content': page_content}  
        
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
        with cf.ThreadPoolExecutor() as executor:
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
    full_save_path = os.path.join(args.save_folder_path, args.split_type, 'inverse_search', args.sub_split)
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
    if os.path.isfile(json_download_file_name) and args.continue_download:
        if os.access(json_download_file_name, os.R_OK):
            with open(json_download_file_name, 'r') as fp:
                all_inverse_annotations_idx = json.load(fp)
        else:
            # wait until the file is not locked
            while not os.access(json_download_file_name, os.R_OK):
                time.sleep(1)
            with open(json_download_file_name, 'r') as fp:
                all_inverse_annotations_idx = json.load(fp)
    else:
        with io.open(json_download_file_name, 'w') as db_file:
            db_file.write(json.dumps({}))
        with io.open(json_download_file_name, 'r') as db_file:
            all_inverse_annotations_idx = json.load(db_file)
    
    # Load dataset
    visual_news_data_mapping = json.load(open("test_dataset/visual_news_test.json"))
    clip_data = json.load(open("test_dataset/news_clippings_test.json"))
    clip_data_annotations = clip_data["annotations"]
    
    # Determine processing range
    start_counter = (args.start_idx if args.start_idx != -1 
                    else (int(list(all_inverse_annotations_idx.keys())[-1])+1 
                          if all_inverse_annotations_idx else 0))
    
    end_counter = (args.end_idx if args.end_idx > 0 
                  else (start_counter + args.how_many if args.how_many > 0 
                        else len(clip_data_annotations)))
    
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
        
        if i >= len(clip_data_annotations):
            break
        
        # Skip if already processed and skip_existing is set
        if args.skip_existing:
            result_path = os.path.join(full_save_path, str(i), 'inverse_annotation.json')
            if os.path.exists(result_path):
                continue
        
        start_time = time.time()
        
        try:
            
            # Get item information
            ann = clip_data_annotations[i]
            image_path = os.path.join('test_dataset',visual_news_data_mapping[str(ann["image_id"])]["image_path"])
            new_folder_path = os.path.join(full_save_path, str(i))
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Detect web annotations
            result = detect_web(image_path, how_many_queries=args.how_many_queries)
            
            print(result)
            
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