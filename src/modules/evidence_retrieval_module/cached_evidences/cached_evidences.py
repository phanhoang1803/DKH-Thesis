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
            '\u200b', '\u200c', '\u200d', '\u202a', '\ufeff',
            '\u2011', '\u2033', '\u0107', '\u0219', '\u010d',
            '\u0101', '\u014d', '\u2665', '\U0001f61b'
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
    """Base class for evidence modules"""
    
    def __init__(self, json_file_path: str):
        """Initialize the BaseEvidencesModule with a JSON file path."""
        self.json_file_path = json_file_path
        with open(json_file_path, "r") as file:
            self.data = json.load(file)

        # Initialize SentenceTransformer model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define domains for filtering
        self.NEWS_DOMAINS = [
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
        
        self.EXCLUDED_DOMAINS = [
            "mdpi.com", "yumpu.com", "scmp.com", "pinterest.com", "imdb.com",
            "movieweb.com", "shutterstock.com", "reddit.com", "alamy.com",
            "alamy.it", "alamyimages.fr", "planetcricket.org"
        ]
    
    def _load_and_encode_image(self, image_path: str, max_size: int = 1024) -> str:
        """
        Load an image from path, resize it, and encode it in base64.
        
        Args:
            image_path: Path to the image file
            max_size: Maximum dimension (width or height) in pixels
                
        Returns:
            Base64 encoded image data or empty string if loading fails
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if image is in RGBA mode
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Resize image while maintaining aspect ratio
                width, height = img.size
                if max(width, height) > max_size:
                    if width > height:
                        new_width = max_size
                        new_height = int(height * (max_size / width))
                    else:
                        new_height = max_size
                        new_width = int(width * (max_size / height))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save image to bytes buffer
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=90)
                
                # Encode to base64
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return ""
        
    def batch_similarity(self, query_text: str, texts: List[str]) -> torch.Tensor:
        """Calculate similarities for multiple texts at once."""
        if not texts:
            return torch.tensor([])
            
        # Encode query and all texts
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        text_embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=32)
        
        # Calculate similarities
        return torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), 
            text_embeddings
        )

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
            title_sim = float(title_similarities[i]) if i < len(title_similarities) else 0.0
            caption_sim = float(caption_similarities[i]) if i < len(caption_similarities) else 0.0
            similarity = max(title_sim, caption_sim)
            
            if similarity >= threshold:
                evidence_scores.append((evidence, similarity))
            
        # Sort by similarity score
        evidence_scores.sort(key=lambda x: x[1], reverse=True)
        return evidence_scores
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain string by removing www. prefix and lowercasing."""
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def filter_evidence_by_domain(self, evidence_list: List[Evidence], 
                                allowed_domains: List[str]) -> List[Evidence]:
        """Filter evidence list by allowed domains."""
        # Normalize allowed domains
        normalized_domains = {self._normalize_domain(domain) for domain in allowed_domains}
        
        # Filter evidence list
        return [ev for ev in evidence_list 
                if self._normalize_domain(ev.domain) in normalized_domains]
    
    def filter_evidence_by_excluding_domains(self, evidence_list: List[Evidence], 
                                           excluded_domains: List[str]) -> List[Evidence]:
        """Filter evidence list by excluding domains."""
        normalized_excluded = {self._normalize_domain(domain) for domain in excluded_domains}
        
        return [ev for ev in evidence_list 
                if self._normalize_domain(ev.domain) not in normalized_excluded]
    
    def filter_evidences(self, max_evidences: int, evidence_list: List[Evidence]) -> List[Evidence]:
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
        
    def filter_unique_by_domain_title(self, evidences: List[Evidence]) -> List[Evidence]:
        """Filter a list of evidence to keep only unique domain+title combinations."""
        seen = set()
        unique_evidences = []
        
        for evidence in evidences:
            # Skip bot verification pages
            if evidence.title == 'Bot Verification':
                continue
                
            # Create a key from domain and title
            key = (evidence.domain, evidence.title)
            
            # Only add if we haven't seen this combination before
            if key not in seen:
                seen.add(key)
                unique_evidences.append(evidence)
        
        return unique_evidences
    
    def get_item_folder_path(self, index: Union[int, str]) -> Optional[str]:
        """Get the folder path for an item by index."""
        # Convert index to int if it's a string
        idx = int(index) if isinstance(index, str) else index
        
        # For odd indices, use the preceding even index (TextEvidencesModule specific)
        # Removed from base class as it's specific to TextEvidencesModule
        
        item = self.data.get(str(idx))
        if not item:
            return None
            
        return item.get("folder_path")

    def get_evidence_by_index(self, index: Union[int, str], query: str = "",
                            max_results: int = 5, threshold: float = 0.7,
                            min_results: int = 1) -> List[Evidence]:
        """
        Get evidence for a specific index, filtered by similarity to query.
        Base implementation to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_evidence_by_index")


class TextEvidencesModule(BaseEvidencesModule):
    """Evidences retrieved by using text search on Google"""
    
    def get_evidence_by_index(self, index: Union[int, str], query: str = "",
                            max_results: int = 5, threshold: float = 0.7,
                            min_results: int = 1, reference_image: str = None,
                            image_similarity_threshold: float = 0.7) -> List[Evidence]:
        """Get evidence for a specific index, filtered by similarity to query."""
        # Convert index to int if it's a string
        idx = int(index) if isinstance(index, str) else index
        
        # For odd indices, use the preceding even index
        if idx % 2 == 1:
            idx -= 1
        
        folder_path = self.get_item_folder_path(idx)
        if not folder_path:
            return []
        
        evidence_list = []
        try:
            annotation_file = os.path.join(folder_path, "direct_annotation.json")
            with open(annotation_file, 'r') as file:
                annotation_data = json.load(file)
            
            # Helper function to extract caption from potentially nested structure
            def extract_caption(caption_data):
                if not caption_data:
                    return ''
                
                if isinstance(caption_data, dict):
                    caption_node = caption_data.get('caption_node', '')
                    alt_node = caption_data.get('alt_node', '')
                    if caption_node and alt_node:
                        return caption_node + '|' + alt_node
                    return caption_node or alt_node
            
            # Process all image categories
            image_categories = [
                'images_with_captions', 
                'images_with_no_captions',
                'images_with_caption_matched_tags'
            ]
            
            for category in image_categories:
                for item in annotation_data.get(category, []):
                    image_path = item.get('image_path', '')
                    image_data = self._load_and_encode_image(image_path)
                    if not image_data:
                        continue
                    
                    caption = extract_caption(item.get('caption', ''))
                    
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
        
        # Filter by domain first, then exclude certain domains
        filtered_evidence_list = self.filter_evidence_by_domain(evidence_list, self.NEWS_DOMAINS)
        
        # Filter by unique domain+title combinations
        filtered_evidence_list = self.filter_unique_by_domain_title(filtered_evidence_list)
        
        # Filter by image similarity if reference image is provided
        if reference_image:
            filtered_evidence_list = self.filter_evidence_by_image_similarity(
                filtered_evidence_list, 
                reference_image, 
                image_similarity_threshold, 
                is_base64=True
            )
        
        # Filter to max_evidences based on unique titles
        final_evidence = self.filter_evidences(max_results, filtered_evidence_list)
        
        return final_evidence

    def filter_evidence_by_image_similarity(self, evidence_list: List[Evidence], 
                                          reference_image: str, 
                                          threshold: float = 0.6,
                                          is_base64: bool = False) -> List[Evidence]:
        """Filter evidence by image similarity to a reference image.
        
        Args:
            evidence_list: List of Evidence objects to filter
            reference_image: Either a path to the reference image or a base64-encoded image string
            threshold: Minimum similarity threshold (0.0 to 1.0)
            is_base64: Whether the reference_image is a base64-encoded string
            
        Returns:
            Filtered list of Evidence objects
        """
        if not evidence_list or not reference_image:
            return evidence_list
        
        # Load reference image features
        if is_base64:
            reference_features = self._extract_image_features_from_base64(reference_image)
        else:
            reference_features = self._extract_image_features(reference_image)
            
        if reference_features is None:
            print(f"Failed to extract features from reference image")
            return evidence_list
        
        similar_evidence = []
        for evidence in evidence_list:
            # Extract features from evidence image
            evidence_features = self._extract_image_features_from_base64(evidence.image_data)
            if evidence_features is None:
                continue
            
            # Calculate similarity score
            similarity = self._calculate_image_similarity(reference_features, evidence_features)
            
            # Add evidence if similarity is above threshold
            if similarity >= threshold:
                # Add similarity score to evidence for debugging/sorting
                evidence.similarity_score = similarity
                similar_evidence.append(evidence)
        
        return similar_evidence
    
    def _extract_image_features(self, image_path: str):
        """Extract features from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image features or None if extraction failed
        """
        try:
            # You can use various methods to extract features
            # Example using PIL and a simple approach:
            from PIL import Image
            import numpy as np
            
            # Load image
            img = Image.open(image_path)
            
            # Resize to standard size for comparison
            img = img.resize((224, 224))
            
            # Convert to numpy array and normalize
            features = np.array(img).flatten() / 255.0
            
            return features
        except Exception as e:
            print(f"Error extracting features from image {image_path}: {str(e)}")
            return None
            
    def _extract_image_features_from_base64(self, base64_string):
        """Extract features from a base64-encoded image string.
        
        Args:
            base64_string: Base64-encoded image string
            
        Returns:
            Image features or None if extraction failed
        """
        try:
            import base64
            import io
            from PIL import Image
            import numpy as np
            
            # Remove header if present (e.g., "data:image/jpeg;base64,")
            if ',' in base64_string:
                base64_string = base64_string.split(',', 1)[1]
                
            # Decode base64 string to bytes
            image_data = base64.b64decode(base64_string)
            
            # Load image from bytes
            img = Image.open(io.BytesIO(image_data))
            
            # Resize to standard size for comparison
            img = img.resize((224, 224))
            
            # Convert to numpy array and normalize
            features = np.array(img).flatten() / 255.0
            
            return features
        except Exception as e:
            print(f"Error extracting features from base64 image: {str(e)}")
            return None
    
    def _calculate_image_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        import numpy as np
        
        # Calculate cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))

class ImageEvidencesModule(BaseEvidencesModule):
    """Evidences for image search without loading actual images"""
    
    def get_entities_by_index(self, index: Union[int, str]) -> List[str]:
        """Retrieve entities for a specific image index."""
        folder_path = self.get_item_folder_path(index)
        if not folder_path:
            return []
        
        try:
            annotation_file = os.path.join(folder_path, "inverse_annotation.json")
            with open(annotation_file, 'r') as file:
                annotation_data = json.load(file)
                
            return annotation_data.get("entities", [])
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading inverse annotation file for index {index}: {str(e)}")
            return []
    
    def get_evidence_by_index(self, index: Union[int, str], query: str = "",
                             max_results: int = 5) -> List[Evidence]:
        """
        Get evidence for a specific index.
        This version doesn't load actual image data.
        """
        folder_path = self.get_item_folder_path(index)
        if not folder_path:
            return []
        
        evidence_list = []
        try:
            annotation_file = os.path.join(folder_path, "inverse_annotation.json")
            with open(annotation_file, 'r') as file:
                annotation_data = json.load(file)
            
            # Helper function to extract caption from potentially nested structure
            def extract_caption(caption_data):
                if not caption_data:
                    return ''
                if isinstance(caption_data, dict):
                    caption_node = caption_data.get('caption_node', '')
                    alt_node = caption_data.get('alt_node', '')
                    if caption_node and alt_node:
                        return caption_node + '|' + alt_node
                    return caption_node or alt_node
                return caption_data
                
            # Helper function to extract domain from page_link
            def extract_domain(page_link):
                if not page_link:
                    return ''
                parsed_url = urlparse(page_link)
                return parsed_url.netloc
                
            # Helper function to get content
            def get_content(item):
                return item.get('content', '') or item.get('snippet', '')
            
            # Process all categories of matched images
            categories = [
                'all_fully_matched_captions',
                'all_partially_matched_captions',
                'fully_matched_no_text',
                'all_match_captions',
                'partially_matched_no_text',
                'matched_no_text'
            ]
            
            for category in categories:
                for item in annotation_data.get(category, []):
                    evidence_list.append(Evidence(
                        domain=extract_domain(item.get('page_link', '')),
                        image_path=item.get('image_link', ''),
                        image_data="",  # Empty image data
                        title=item.get('title', ''),
                        caption=extract_caption(item.get('caption')),
                        content=get_content(item)
                    ))
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading inverse annotation file for index {index}: {str(e)}")
            return []
        
        filtered_evidence = self.filter_evidence_by_domain(evidence_list, self.NEWS_DOMAINS)
        
        # Filter by excluding domains
        # filtered_evidence = self.filter_evidence_by_excluding_domains(evidence_list, self.EXCLUDED_DOMAINS)
        # filtered_evidence = evidence_list
        
        # Filter evidences with the same domain and title
        filtered_evidence = self.filter_unique_by_domain_title(filtered_evidence)

        # Filter to max_evidences based on unique titles
        final_evidence = self.filter_evidences(max_results, filtered_evidence)
        
        return final_evidence