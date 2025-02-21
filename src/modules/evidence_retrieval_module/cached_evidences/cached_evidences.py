import json
import os
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
from PIL import Image
import base64
from io import BytesIO

@dataclass
class Evidence:
    domain: str
    image_path: str
    image_data: str  # Base64 encoded image data
    title: str
    caption: str
    content: str
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing/replacing problematic characters."""
        if not text:
            return ""
            
        # List of problematic Unicode characters to remove
        chars_to_remove = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\u202a',  # Left-to-right embedding
            '\ufeff',  # Zero-width no-break space
            '\u2011',  # Non-breaking hyphen
            '\u2033',  # Double prime
            '\u0107',  # Latin small letter c with acute
            '\u0219',  # Latin small letter s with comma below
            '\u010d',  # Latin small letter c with caron
            '\u0101',  # Latin small letter a with macron
            '\u014d',  # Latin small letter o with macron
            '\u2665',  # Black heart suit
            '\U0001f61b'  # Face with stuck-out tongue
        ]
        
        # Remove problematic characters
        for char in chars_to_remove:
            text = text.replace(char, '')
        
        # Remove or fix other special characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    def to_dict(self):
        try:
            return {
                "title": self._clean_text(self.title),
                "content": self._clean_text(self.content),
                "caption": self._clean_text(self.caption),
                "domain": self._clean_text(self.domain)
            }
        except Exception as e:
            return {
                "error": f"Serialization failed: {str(e)}",
                "url": self._clean_text(self.url),
                "title": self._clean_text(self.title)
            }

class EvidencesModule:
    def __init__(self, json_file_path: str):
        """
        Initialize the EvidencesModule with a JSON file path.
        
        Args:
            json_file_path (str): Path to the JSON file containing evidence data
        """
        self.json_file_path = json_file_path
        with open(json_file_path, "r") as file:
            self.data = json.load(file)
    
    def _load_and_encode_image(self, image_path: str) -> str:
        """
        Load an image from path and encode it in base64.
        
        Args:
            image_path (str): Path to the image file
            folder_path (str): Base folder path
            
        Returns:
            str: Base64 encoded image data or empty string if loading fails
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if image is in RGBA mode
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Save image to bytes buffer
                buffer = BytesIO()
                img.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                
                # Encode to base64
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return ""

    def get_evidence_by_index(self, index: Union[int, str], max_evidences = 5) -> List[Evidence]:
        """
        Get evidence for a specific index. For odd indices, returns the evidence
        of the preceding even index.
        
        Args:
            index (Union[int, str]): The index to look up
            
        Returns:
            List[Evidence]: List of Evidence objects containing domain, image data, title, caption, and content
        """
        # Convert index to int if it's a string
        idx = int(index) if isinstance(index, str) else index
        
        # For odd indices, use the preceding even index
        if idx % 2 == 1:
            idx -= 1
        
        item = self.data.get(str(idx))
        if not item:
            return []
            
        folder_path = item.get("folder_path")
        if not folder_path:
            return []
        
        evidence_list = []
        try:
            annotation_file = os.path.join(folder_path, "direct_annotation.json")
            with open(annotation_file, 'r') as file:
                annotation_data = json.load(file)
            
            # Process images with captions
            for img in annotation_data.get('images_with_captions', []):
                image_path = img.get('image_path', '')
                image_data = self._load_and_encode_image(image_path)
                if image_data == "":
                    continue
                
                caption = img.get('caption', {}).get('caption_node', '') if isinstance(img.get('caption'), dict) else ''
                
                evidence_list.append(Evidence(
                    domain=img.get('domain', ''),
                    image_path=image_path,
                    image_data=image_data,
                    title=img.get('page_title', ''),
                    caption=caption,
                    content=img.get('snippet', '')
                ))
            
            # Process images without captions
            for img in annotation_data.get('images_with_no_captions', []):
                image_path = img.get('image_path', '')
                image_data = self._load_and_encode_image(image_path)
                if image_data == "":
                    continue
                
                evidence_list.append(Evidence(
                    domain=img.get('domain', ''),
                    image_path=image_path,
                    image_data=image_data,
                    title=img.get('page_title', ''),
                    caption='',
                    content=img.get('snippet', '')
                ))
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading direct annotation file for index {idx}: {str(e)}")
            return []
        
        # Filter to max_evidences based on unique titles
        filtered_evidence = []
        seen_titles = set()
        
        for evidence in evidence_list:
            title = evidence.title.strip()
            if title not in seen_titles and len(filtered_evidence) < max_evidences:
                filtered_evidence.append(evidence)
                seen_titles.add(title)
        
        # Second pass: if we still need more evidence, include duplicates
        if len(filtered_evidence) < max_evidences:
            for evidence in evidence_list:
                if evidence not in filtered_evidence and len(filtered_evidence) < max_evidences:
                    filtered_evidence.append(evidence)
        
        return filtered_evidence

if __name__ == "__main__":
    # Example path - adjust this to your actual metadata file path
    metadata_path = "queries_dataset/merged_balanced/direct_search/test/test.json"
    
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
            print(f"Image Path: {ev.image_path}")
            print(f"Image Data Length: {len(ev.image_data)} bytes")  # Print length of base64 data
            print(f"Title: {ev.title}")
            print(f"Caption: {ev.caption}")
            print(f"Content: {ev.content}")