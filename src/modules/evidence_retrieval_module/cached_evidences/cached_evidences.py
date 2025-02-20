import json
import os
from typing import Dict, Optional, Union, List
from dataclasses import dataclass

@dataclass
class Evidence:
    domain: str
    image: str
    title: str
    caption: str
    content: str

class EvidencesModule:
    def __init__(self, json_file_path: str):
        """
        Initialize the EvidencesModule with a JSON file path.
        
        Args:
            json_file_path (str): Path to the JSON file containing evidence data
        """
        self.json_file_path = json_file_path
        self.data = self._load_json()
    
    def _load_json(self) -> Dict:
        """
        Load and parse the JSON file.
        
        Returns:
            Dict: Parsed JSON data
        """
        try:
            with open(self.json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at {self.json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file {self.json_file_path}")

    def get_evidence_by_index(self, index: Union[int, str]):
        """
        Get evidence for a specific index. For odd indices, returns the evidence
        of the preceding even index.
        
        Args:
            index (Union[int, str]): The index to look up
            
        Returns:
            List[Evidence]: List of Evidence objects containing domain, image, title, caption, and content
        """
        # Convert index to int if it's a string
        idx = int(index) if isinstance(index, str) else index
        
        # For odd indices, use the preceding even index
        if idx % 2 == 1:
            idx -= 1
        
        item = self.data.get(str(idx))
        
        folder_path = item.get("folder_path")
        
        evidence_list = []
        try:
            with open(os.path.join(folder_path, "direct_annotation.json"), 'r') as file:
                annotation_data = json.load(file)
            
            # Process images with captions
            for img in annotation_data.get('images_with_captions', []):
                caption = img.get('caption', {}).get('caption_node', '') if isinstance(img.get('caption'), dict) else ''
                evidence_list.append(Evidence(
                    domain=img.get('domain', ''),
                    image=img.get('image_path', ''),
                    title=img.get('page_title', ''),
                    caption=caption,
                    content=img.get('snippet', '')
                ))
            
            # Process images without captions
            for img in annotation_data.get('images_with_no_captions', []):
                evidence_list.append(Evidence(
                    domain=img.get('domain', ''),
                    image=img.get('image_path', ''),
                    title=img.get('page_title', ''),
                    caption='',
                    content=img.get('snippet', '')
                ))
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading direct annotation file for index {idx}: {str(e)}")
            return []
        
        return evidence_list

if __name__ == "__main__":
    # Example path - adjust this to your actual metadata file path
    metadata_path = "path/to/metadata.json"
    
    # Initialize the module
    module = EvidencesModule(metadata_path)
    
    # Test with both even and odd indices
    test_indices = [0, 1, 2, 3]
    
    for idx in test_indices:
        print(f"\nProcessing index {idx}:")
        evidences = module.get_evidence_by_index(idx)
        
        print(f"Found {len(evidences)} evidences")
        
        # Print details of each evidence
        for i, ev in enumerate(evidences, 1):
            print(f"\nEvidence {i}:")
            print(f"Domain: {ev.domain}")
            print(f"Image: {ev.image}")
            print(f"Title: {ev.title}")
            print(f"Caption: {ev.caption}")
            print(f"Content: {ev.content}")