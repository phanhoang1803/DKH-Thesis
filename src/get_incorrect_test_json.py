# python .\src\inference_2.py --no_shuffle --ner_model "dslim/bert-base-NER" --start_idx 0 --output_dir_path "./result_incorrect_cases/" --data_path "data/incorrect_test_cases_acm.json"

import json
import os
import argparse
from typing import List, Dict

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Filter incorrect test cases from JSON file')
    
    parser.add_argument('--test_file', 
                      type=str,
                      default='data/public_test_acm.json',
                      help='Path to original test JSON file')
    
    parser.add_argument('--result_dir', 
                      type=str,
                      default='result',
                      help='Directory contain result files')
    
    parser.add_argument('--output_file',
                      type=str,
                      default='data/incorrect_test_cases_acm.json',
                      help='Output file path for incorrect cases')
    
    return parser.parse_args()

def load_json_file(filepath: str) -> List[Dict]:
    """Load a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return []

def get_result_files(base_dir: str) -> List[str]:
    """Get all result files with full path, excluding final_results.json"""
    result_files = []
    try:
        files = os.listdir(base_dir)
        for file in files:
            if file.endswith('.json') and file != 'final_results.json':
                result_files.append(os.path.join(base_dir, file))
    except Exception as e:
        print(f"Error reading directory {base_dir}: {str(e)}")
    return result_files

def create_caption_map(test_data: List[Dict]) -> Dict[str, Dict]:
    """Create a map of caption to test data"""
    caption_map = {}
    for item in test_data:
        caption = item.get('caption1', '')
        if caption:
            caption_map[caption] = item
    return caption_map

def get_incorrect_test_cases(test_data: List[Dict], results: List[Dict]) -> List[Dict]:
    """Find test cases where prediction was incorrect using caption mapping"""
    incorrect_cases = []
    caption_to_test = create_caption_map(test_data)
    processed_captions = set()
    
    # Process each result
    for result in results:
        caption = result.get('caption', '')
        # print(caption)
        if not caption or caption in processed_captions:
            continue
        
        test_item = caption_to_test.get(caption)
        # print(test_item)
        if test_item is None:
            continue
            
        ground_truth = test_item.get('context_label', None)
        prediction = result.get('final_result', {}).get('OOC', None)
        
        if ground_truth != prediction:
            incorrect_cases.append(test_item)
            
        processed_captions.add(caption)
        
        # try:
        #     caption = result.get('caption', '')
        #     if not caption or caption in processed_captions:
        #         continue
                
        #     test_item = caption_to_test.get(caption)
        #     if test_item is None:
        #         continue
                
        #     ground_truth = test_item.get('ground_truth', None)
        #     prediction = result.get('final_result', {}).get('OOC', None)
        #     print(ground_truth)
        #     print(prediction)
            
        #     if ground_truth != prediction:
        #         incorrect_cases.append(test_item)
                
        #     processed_captions.add(caption)
        # except:
        #     pass
    
    # Print statistics about matching
    print(f"\nMatching Statistics:")
    print(f"Total test cases: {len(test_data)}")
    print(f"Matched cases: {len(processed_captions)}")
    print(f"Unmatched cases: {len(test_data) - len(processed_captions)}")
    
    return incorrect_cases

def save_json_file(data: List[Dict], filepath: str):
    """Save data to a JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    # Parse arguments
    args = parse_args()
    
    # Load original test data
    print(f"Loading test file: {args.test_file}")
    test_data = load_json_file(args.test_file)
    if not test_data:
        print("Failed to load test data")
        
    # Load all results
    print(f"Loading results from: {args.result_dir}")
    result_files = get_result_files(args.result_dir)
    
    if not result_files:
        print(f"No result files found in {args.result_dir}")
        return

    results = []
    for result_file in result_files:
        try:
            print(f"Processing: {result_file}")
            result_data = load_json_file(result_file)
            results.append(result_data)
        except:
            pass
    
    # Get incorrect test cases
    incorrect_cases = get_incorrect_test_cases(test_data, results)
    
    # Save new test file with only incorrect cases
    save_json_file(incorrect_cases, args.output_file)
    
    print(f"\nAnalysis Complete:")
    print(f"Total test cases: {len(test_data)}")
    print(f"Total results processed: {len(results)}")
    print(f"Incorrect cases: {len(incorrect_cases)}")
    print(f"File saved as: {args.output_file}")

    # Print error distribution
    total_matched = len(results)  # Using matched cases as total
    if total_matched > 0:
        incorrect_count = len(incorrect_cases)
        accuracy = ((total_matched - incorrect_count) / total_matched) * 100
        
        print(f"\nAccuracy Analysis:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Error rate: {100-accuracy:.2f}%")
    else:
        print("\nNo matched cases found to calculate accuracy")

if __name__ == "__main__":
    main()