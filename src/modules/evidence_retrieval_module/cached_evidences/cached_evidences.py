import json
import os
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from PIL import Image
import base64
from io import BytesIO

from sentence_transformers import SentenceTransformer
import torch
from urllib.parse import urlparse

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
                "title": self._clean_text(self.title)
            }

class BaseEvidencesModule:
    """
    Base class for evidence modules
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize the BaseEvidencesModule with a JSON file path.
        
        Args:
            json_file_path (str): Path to the JSON file containing evidence data
        """
        self.json_file_path = json_file_path
        with open(json_file_path, "r") as file:
            self.data = json.load(file)

        # Initialize SentenceTransformer model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.NEWS_DOMAINS = [
            # Major News Organizations
            "theguardian.com", "usatoday.com", "bbc.com", "cnn.com", "latimes.com",
            "independent.co.uk", "nbcnews.com", "npr.org", "aljazeera.com",
            "apnews.com", "cbsnews.com", "abcnews.go.com", "pbs.org", "abc.net.au",
            
            # Newspapers
            "denverpost.com", "tennessean.com", "thetimes.com", "sandiegouniontribune.com",
            
            # Magazines/Long-form Journalism
            # "magazine.atavist.com", "newyorker.com", "theatlantic.com", "vanityfair.com"
            "newyoker.com", "ffxnow.com", "laist.com", "hudson.org", "rollcall.com",
            "economist.com", "nps.gov", 
            
            # COSMOS ADDED DOMAINS
            'nytimes.com', 'washingtontimes.com', 'edition.cnn.com', "bbc.co.uk",
            "magazine.atavist.com", "newyoker.com", "aljazeera.com", "vox.com",
            "euronews.com"
        ]

        self.EXCLUDED_DOMAINS = [
            "snopes.com", "en.wikipedia.org", "mdpi.com", "yumpu.com",
            "reddit.com", "scmp.com", "pinterest.com", "imdb.com",
            "movieweb.com"
        ]
        
    def batch_similarity(self, query_text: str, texts: List[str]) -> torch.Tensor:
        """Calculate similarities for multiple texts at once."""
        if not texts:
            return torch.tensor([])
            
        # Encode query and all texts in separate batches
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        text_embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=32)
        
        # Calculate similarities all at once
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), 
            text_embeddings
        )
        
        return similarities

    def filter_by_similarity(self, query: str, evidence_list: List[Evidence], 
                           threshold: float = 0.7) -> List[Tuple[Evidence, float]]:
        """Filter evidence based on similarity with query."""
        if not evidence_list:
            return []

        # Prepare lists of titles and captions
        titles = [ev.title for ev in evidence_list]
        captions = [ev.caption if ev.caption else "" for ev in evidence_list]
        
        # Calculate similarities in batch
        title_similarities = self.batch_similarity(query, titles)
        caption_similarities = self.batch_similarity(query, captions)
        
        # Combine similarities and evidence
        evidence_scores = []
        for i, evidence in enumerate(evidence_list):
            title_sim = float(title_similarities[i]) if len(title_similarities) > i else 0.0
            caption_sim = float(caption_similarities[i]) if len(caption_similarities) > i else 0.0
            similarity = max(title_sim, caption_sim)
            
            if similarity >= threshold:
                evidence_scores.append((evidence, similarity))
            
        # Sort by similarity score
        evidence_scores.sort(key=lambda x: x[1], reverse=True)
        return evidence_scores

    def filter_evidence_by_domain(self, evidence_list, allowed_domains):
        """
        Filter evidence list by allowed domains.
        
        Args:
            evidence_list (list): List of Evidence objects to filter
            allowed_domains (list): List of domain strings to allow
            
        Returns:
            list: Filtered list of Evidence objects
        """
        # Normalize allowed domains
        normalized_domains = set()
        for domain in allowed_domains:
            domain = domain.lower().strip()
            if domain.startswith("www."):
                domain = domain[4:]
            normalized_domains.add(domain)
        
        # Filter evidence list
        filtered_evidence = []
        for ev in evidence_list:
            domain = ev.domain.lower().strip()
            if domain.startswith("www."):
                domain = domain[4:]
                
            if domain in normalized_domains:
                filtered_evidence.append(ev)
                
        return filtered_evidence
    
    def filter_evidence_by_excluding_domains(self, evidence_list, excluded_domains):
        """
        Filter evidence list by excluding domains.
        
        Args:
            evidence_list (list): List of Evidence objects to filter
            excluded_domains (list): List of domain strings to exclude
            
        Returns:
            list: Filtered list of Evidence objects
        """
        normalized_excluded_domains = set()
        for domain in excluded_domains:
            domain = domain.lower().strip()
            if domain.startswith("www."):
                domain = domain[4:]
            normalized_excluded_domains.add(domain)
        
        filtered_evidence = []
        for ev in evidence_list:
            domain = ev.domain.lower().strip()
            if domain.startswith("www."):
                domain = domain[4:]
                
            if domain not in normalized_excluded_domains:
                filtered_evidence.append(ev)
                
        return filtered_evidence
    
    def filter_evidences(self, max_evidences: int, evidence_list: List[Evidence]):
        """Filter evidence list to maximum size while preserving uniqueness by title."""
        filtered_evidence = []
        seen_titles = set()
        
        # First pass: include unique titles
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

    # def get_evidence_by_index(self, index: Union[int, str], query: str,
    #                         max_results: int = 5, threshold: float = 0.7,
    #                         min_results: int = 1) -> List[Evidence]:
    #     """
    #     Get evidence for a specific index, filtered by similarity to query.
    #     Base implementation to be overridden by subclasses.
    #     """
    #     raise NotImplementedError("Subclasses must implement get_evidence_by_index")


class TextEvidencesModule(BaseEvidencesModule):
    """
    Evidences retrieved by using text search on Google
    """
    
    # def _load_and_encode_image(self, image_path: str) -> str:
    #     """
    #     Load an image from path and encode it in base64.
        
    #     Args:
    #         image_path (str): Path to the image file
            
    #     Returns:
    #         str: Base64 encoded image data or empty string if loading fails
    #     """
    #     try:
    #         with Image.open(image_path) as img:
    #             # Convert to RGB if image is in RGBA mode
    #             if img.mode == 'RGBA':
    #                 img = img.convert('RGB')
                
    #             # Save image to bytes buffer
    #             buffer = BytesIO()
    #             img.save(buffer, format='JPEG')
    #             image_bytes = buffer.getvalue()
                
    #             # Encode to base64
    #             return base64.b64encode(image_bytes).decode('utf-8')
    #     except Exception as e:
    #         print(f"Error loading image {image_path}: {str(e)}")
    #         return ""
    
    def _load_and_encode_image(self, image_path: str, max_size: int = 1024):
        """
        Load an image from path, resize it, and encode it in base64.
        
        Args:
            image_path (str): Path to the image file
            max_size (int): Maximum dimension (width or height) in pixels
                
        Returns:
            str: Base64 encoded image data or empty string if loading fails
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if image is in RGBA mode
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Resize image while maintaining aspect ratio
                width, height = img.size
                if width > height:
                    if width > max_size:
                        new_width = max_size
                        new_height = int(height * (max_size / width))
                else:
                    if height > max_size:
                        new_height = max_size
                        new_width = int(width * (max_size / height))
                        
                # Only resize if needed
                if width > max_size or height > max_size:
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save image to bytes buffer
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=90)
                image_bytes = buffer.getvalue()
                
                # Encode to base64
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return ""

    def get_evidence_by_index(self, index: Union[int, str], query: str,
                            max_results: int = 5, threshold: float = 0.7,
                            min_results: int = 1):
        """
        Get evidence for a specific index, filtered by similarity to query.
        
        Args:
            index (Union[int, str]): The index to look up
            query (str): Query text to filter evidence by similarity
            max_results (int): Maximum number of results to return
            threshold (float): Minimum similarity score threshold
            min_results (int): Minimum number of results to return even if below threshold
            
        Returns:
            List[Evidence]: List of Evidence objects filtered by similarity to query
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
            for item in annotation_data.get('images_with_captions', []):
                image_path = item.get('image_path', '')
                image_data = self._load_and_encode_image(image_path)
                if image_data == "":
                    continue
                
                caption = ''
                if isinstance(item.get('caption'), dict):
                    caption = item.get('caption', {}).get('caption_node', '')
                    if caption == '':
                        caption = item.get('caption', {}).get('alt_node', '')
                    
                evidence_list.append(Evidence(
                    domain=item.get('domain', ''),
                    image_path=image_path,
                    image_data=image_data,
                    title=item.get('page_title', ''),
                    caption=caption,
                    content=item.get('snippet', '')
                ))
            
            # Process images without captions
            for item in annotation_data.get('images_with_no_captions', []):
                image_path = item.get('image_path', '')
                image_data = self._load_and_encode_image(image_path)
                if image_data == "":
                    continue
                
                evidence_list.append(Evidence(
                    domain=item.get('domain', ''),
                    image_path=image_path,
                    image_data=image_data,
                    title=item.get('page_title', ''),
                    caption='',
                    content=item.get('snippet', '')
                ))
                
            for item in annotation_data.get('images_with_caption_matched_tags', []):
                image_path = item.get('image_path', '')
                image_data = self._load_and_encode_image(image_path)
                if image_data == "":
                    continue
                
                caption = ''
                if isinstance(item.get('caption'), dict):
                    caption = item.get('caption', {}).get('caption_node', '')
                    if caption == '':
                        caption = item.get('caption', {}).get('alt_node', '')
                
                evidence_list.append(Evidence(
                    domain=item.get('domain', ''),
                    image_path=image_path,
                    image_data=image_data,
                    title=item.get('page_title', ''),
                    caption=caption,
                    content=item.get('snippet', '')
                ))
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading direct annotation file for index {idx}: {str(e)}")
            return []
        
        # Filter by domain first
        filtered_evidence_list = self.filter_evidence_by_domain(evidence_list, self.NEWS_DOMAINS)
        
        # if filtered_evidence_list == []:
        #     filtered_evidence_list = self.filter_evidence_by_excluding_domains(evidence_list, self.EXCLUDED_DOMAINS)
        #     max_results = 1
        
        # Then filter by similarity
        evidence_scores = self.filter_by_similarity(query, filtered_evidence_list, threshold)
        
        # If we don't have enough results meeting the threshold, include top results
        if len(evidence_scores) < min_results:
            evidence_scores = self.filter_by_similarity(query, filtered_evidence_list, threshold=0.0)[:min_results]
        
        # Get just the evidence objects, scores no longer needed
        filtered_evidence = [ev for ev, _ in evidence_scores]
        
        # Filter to max_evidences based on unique titles
        final_evidence = self.filter_evidences(max_results, filtered_evidence)
        
        return final_evidence


class ImageEvidencesModule(BaseEvidencesModule):
    """
    Evidences for image search without loading actual images
    """
    def get_entities_by_index(self, index: Union[int, str]):
        """
        Retrieve entities for a specific image index.
        
        Args:
            index (Union[int, str]): The index of the image in the JSON data
            
        Returns:
            Optional[List[str]]: List of entities for the specified index, or None if not found
        """
        idx = int(index) if isinstance(index, str) else index
        
        item = self.data.get(str(idx))
        if not item:
            return []
            
        folder_path = item.get("folder_path")
        if not folder_path:
            return []
        
        try:
            annotation_file = os.path.join(folder_path, "inverse_annotation.json")
            with open(annotation_file, 'r') as file:
                annotation_data = json.load(file)
                
            entities = annotation_data.get("entities", [])
            return entities
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading direct annotation file for index {idx}: {str(e)}")
            return []
    
    def get_evidence_by_index(self, index: Union[int, str],
                              max_results: int = 5):
        """
        Get evidence for a specific index.
        This version doesn't load actual image data.
        
        Args:
            index (Union[int, str]): The index to look up
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Evidence]: List of Evidence objects filtered by similarity to query
        """
        # Convert index to int if it's a string
        idx = int(index) if isinstance(index, str) else index
        
        item = self.data.get(str(idx))
        if not item:
            return []
            
        folder_path = item.get("folder_path")
        if not folder_path:
            return []
        
        evidence_list = []
        try:
            annotation_file = os.path.join(folder_path, "inverse_annotation.json")
            with open(annotation_file, 'r') as file:
                annotation_data = json.load(file)
            
            # Process images with captions
            for item in annotation_data.get('all_matched_captions', []):
                image_path = item.get('image_path', '')
                
                # Skip image data loading, just set to empty string
                image_data = ""
                
                # Get caption
                caption = ''
                if isinstance(item.get('caption'), dict):
                    caption = item.get('caption', {}).get('caption_node', '')
                    if caption == '':
                        caption = item.get('caption', {}).get('alt_node', '')
                
                # Get domain
                domain = item.get('page_link', '')
                
                parsed_url = urlparse(domain)
                domain = parsed_url.netloc
                
                evidence_list.append(Evidence(
                    domain=domain,
                    image_path=image_path,
                    image_data=image_data,  # Empty image data
                    title=item.get('title', ''),
                    caption=caption,
                    content=item.get('snippet', '')
                ))
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading direct annotation file for index {idx}: {str(e)}")
            return []
        
        # Filter by domain first
        filtered_evidence = self.filter_evidence_by_domain(evidence_list, self.NEWS_DOMAINS)
        
        # if filtered_evidence == []:
        #     filtered_evidence = self.filter_evidence_by_excluding_domains(evidence_list, self.EXCLUDED_DOMAINS)
        #     max_results = 1

        # Filter to max_evidences based on unique titles
        final_evidence = self.filter_evidences(max_results, filtered_evidence)
        
        return final_evidence