# Import necessary libraries
import os
import json
import requests
import shutil
import imghdr
from PIL import Image
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The function you provided for downloading images
def download_and_save_image(image_url, save_folder_path, file_name):
    try:
        response = requests.get(image_url, stream=True, timeout=(60, 60))
        if response.status_code == 200:
            response.raw.decode_content = True
            image_path = os.path.join(save_folder_path, file_name)
            with open(image_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            if imghdr.what(image_path) and imghdr.what(image_path).lower() == 'png':
                img_fix = Image.open(image_path)
                img_fix.convert('RGB').save(image_path)
            return 1 
        else:
            print(f"Failed to download {image_url}: HTTP status {response.status_code}")
            return 0
    except Exception as e:
        print(f"Error downloading {image_url}: {str(e)}")
        return 0

# Function to get the filename from html_path and change extension to jpg
def get_image_filename(html_path):
    if not html_path:
        return None
    
    # Extract the base filename from the html_path
    base_name = os.path.basename(html_path)
    # Change extension to jpg
    if '.' in base_name:
        image_filename = os.path.splitext(base_name)[0] + '.jpg'
    else:
        image_filename = base_name + '.jpg'
    
    return image_filename

# Download task wrapper for parallel execution
def download_task(task):
    success = download_and_save_image(task["url"], task["folder"], task["filename"])
    if success:
        return True, task
    return False, task

# Process a single folder
def process_folder(folder_path):
    inverse_file = os.path.join(folder_path, "inverse_annotation.json")
    
    if not os.path.exists(inverse_file):
        print(f"No inverse_annotation.json found in {folder_path}")
        return False
    
    try:
        # Read the inverse annotation file
        with open(inverse_file, 'r', encoding='utf-8') as f:
            inverse_data = json.load(f)
        
        # Prepare download tasks
        download_tasks = []
        
        # Process all_matched_captions
        if "all_matched_captions" in inverse_data:
            for i, caption in enumerate(inverse_data["all_matched_captions"]):
                # Fixed the condition - was using 'item' instead of 'caption'
                if "image_link" in caption:
                    # Get filename from html_path or use default
                    image_filename = get_image_filename(caption.get("html_path"))
                    
                    if not image_filename:
                        image_filename = f"{i}.jpg"
                    
                    # Add to download tasks
                    download_tasks.append({
                        "url": caption["image_link"],
                        "folder": folder_path,
                        "filename": image_filename,
                        "entry": caption,
                        "index": i,
                        "type": "matched"
                    })
        
        # Process matched_no_text
        if "matched_no_text" in inverse_data:
            for i, item in enumerate(inverse_data["matched_no_text"]):
                if "image_link" in item:
                    # Get filename from html_path or use default
                    image_filename = get_image_filename(item.get("html_path"))
                    
                    if not image_filename:
                        image_filename = f"no_text_{i}.jpg"
                    
                    # Add to download tasks
                    download_tasks.append({
                        "url": item["image_link"],
                        "folder": folder_path,
                        "filename": image_filename,
                        "entry": item,
                        "index": i,
                        "type": "no_text"
                    })
        
        # If no tasks, return early
        if not download_tasks:
            return False
        
        # Process download tasks in parallel
        changes_made = False
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_task, task) for task in download_tasks]
            
            for future in as_completed(futures):
                success, task = future.result()
                if success:
                    # Store just the filename in image_path, not the full path
                    if task["type"] == "matched":
                        inverse_data["all_matched_captions"][task["index"]]["image_path"] = os.path.join(task["folder"], task["filename"])
                    else:
                        inverse_data["matched_no_text"][task["index"]]["image_path"] = os.path.join(task["folder"], task["filename"])
                    changes_made = True
        
        # Save the updated inverse file if changes were made
        if changes_made:
            with open(inverse_file, 'w', encoding='utf-8') as f:
                json.dump(inverse_data, f, indent=4, ensure_ascii=False)
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing folder {folder_path}: {str(e)}")
        return False

# Function to process folders in parallel
def process_folders_parallel(test_dir, max_workers=10):
    # Get all folders in the test directory
    test_folders = [os.path.join(test_dir, folder) for folder in os.listdir(test_dir) 
                    if os.path.isdir(os.path.join(test_dir, folder))]
    print(f"Found {len(test_folders)} folders to process")
    
    # Process folders in parallel
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect futures
        future_to_folder = {executor.submit(process_folder, folder): folder for folder in test_folders}
        
        # Process as they complete with progress bar
        for future in tqdm(as_completed(future_to_folder), total=len(test_folders), desc="Processing folders"):
            folder = future_to_folder[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                print(f"Unhandled error processing {folder}: {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully updated: {success_count} folders")
    print(f"Failed or no changes needed: {fail_count} folders")
    
    return success_count, fail_count

# Main execution - can be run directly
if __name__ == "__main__":
    # Define the base test directory
    test_dir = "queries_dataset/merged_balanced/inverse_search/test"
    
    # Process all folders in parallel
    process_folders_parallel(test_dir)