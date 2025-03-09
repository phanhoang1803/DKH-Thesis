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

import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io

import io
from transformers import AutoImageProcessor, AutoModel

@dataclass
class Evidence:
    domain: str
    image_path: str
    image_data: str  # Base64 encoded image data
    title: str
    caption: str
    content: str
    source: Optional[str] = None
    
    def __init__(self, domain="", image_path="", image_data="", title="", caption="", content="", source=""):
        self.domain = domain
        self.image_path = image_path
        self.image_data = image_data
        self.title = title
        self.caption = caption
        self.content = content
        self.source = source
        self.similarity_score = 0.0
    
    def _clean_text(self, text: str):
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
                "domain": self._clean_text(self.domain),
                "source": self.source 
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
            "alamy.it", "alamyimages.fr", "planetcricket.org",
            "www.cnnbrasil.com.br", "www.infomoney.com.br", "gettyimages.com", "gettyimages.co.uk",
            
        ]
    
    def _initialize_vit_model(self, model_ckpt="google/vit-base-patch16-224"):
        """Initialize the Vision Transformer model for semantic image similarity."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
            self.vit_model = AutoModel.from_pretrained(model_ckpt).to(self.device)
            self.vit_model.eval()
            self.vit_initialized = True
            print(f"ViT model initialized successfully on {self.device}")
        except Exception as e:
            print(f"Error initializing ViT model: {str(e)}")
            self.vit_initialized = False
    
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
    
    def filter_evidence_by_image_similarity(self, evidence_list: List[Evidence], 
                                          reference_image: str, 
                                          threshold: float = 0.6,
                                          is_base64: bool = False):
        """Filter evidence by semantic image similarity to a reference image using ViT.
        
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
            
        # Check if ViT model was initialized successfully
        if not hasattr(self, 'vit_initialized') or not self.vit_initialized:
            print("ViT model not initialized. Falling back to basic similarity.")
            # Fallback to the original implementation
            return self._filter_evidence_by_image_similarity_basic(
                evidence_list, reference_image, threshold, is_base64
            )
        
        # Extract embeddings from reference image
        if is_base64:
            reference_embeddings = self._extract_embeddings_from_base64(reference_image)
        else:
            reference_embeddings = self._extract_embeddings(reference_image)
            
        if reference_embeddings is None:
            print(f"Failed to extract embeddings from reference image")
            return evidence_list
        
        similar_evidence = []
        
        # For visualization purposes if needed
        try:
            if is_base64:
                reference_pil_image = Image.open(io.BytesIO(base64.b64decode(str(reference_image))))
            else:
                reference_pil_image = Image.open(reference_image)
        except Exception as e:
            print(f"Warning: Could not open reference image for visualization: {e}")
            reference_pil_image = None
        
        for evidence in evidence_list:
            # Extract embeddings from evidence image
            evidence_embeddings = self._extract_embeddings_from_base64(evidence.image_data)
            if evidence_embeddings is None:
                continue
            
            # Calculate semantic similarity score
            similarity = self._calculate_image_similarity(reference_embeddings, evidence_embeddings)
            
            print(f"Semantic Similarity: {similarity}")
            
            # Optional: Show comparison images for debugging
            # if similarity > threshold and reference_pil_image is not None:
            #     try:
            #         evidence_image = Image.open(io.BytesIO(base64.b64decode(str(evidence.image_data))))
                    
            #         # Create a new image with both images side by side
            #         comparison_image = Image.new('RGB', (reference_pil_image.width + evidence_image.width, 
            #                                           max(reference_pil_image.height, evidence_image.height)))
            #         comparison_image.paste(reference_pil_image, (0, 0))
            #         comparison_image.paste(evidence_image, (reference_pil_image.width, 0))
                    
            #         # Show the comparison image
            #         comparison_image.show()
            #     except Exception as e:
            #         print(f"Warning: Could not show comparison image: {e}")
            
            # Add evidence if similarity is above threshold
            if similarity >= threshold:
                # Add similarity score to evidence for debugging/sorting
                evidence.similarity_score = similarity
                similar_evidence.append(evidence)
        
        # Sort by similarity score (highest first)
        similar_evidence.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_evidence
    
    def _calculate_image_similarity(self, embeddings1, embeddings2):
        """Calculate cosine similarity between two embedding vectors.
        
        Args:
            embeddings1: First embedding vector
            embeddings2: Second embedding vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Compute cosine similarity between the embeddings
        dot_product = np.dot(embeddings1, embeddings2)
        norm1 = np.linalg.norm(embeddings1)
        norm2 = np.linalg.norm(embeddings2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))

    def _extract_embeddings(self, image_path):
        """Extract semantic embeddings from an image file using ViT.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image embeddings or None if extraction failed
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process image for ViT model
            inputs = self.image_processor(images=image, return_tensors="pt", use_fast=True).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                # Use CLS token as image embedding
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
            
            return embeddings
        except Exception as e:
            print(f"Error extracting embeddings from image {image_path}: {str(e)}")
            return None
        
    def _extract_embeddings_from_base64(self, base64_string):
        """Extract semantic embeddings from a base64-encoded image string using ViT.
        
        Args:
            base64_string: Base64-encoded image string
            
        Returns:
            Image embeddings or None if extraction failed
        """
        try:
            # Handle data URI format if present
            if isinstance(base64_string, str) and base64_string.startswith('data:image/'):
                base64_string = base64_string.split(';base64,', 1)[1]
                    
            # Decode base64 string to bytes
            image_data = base64.b64decode(base64_string)
            
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Process image for ViT model
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                # Use CLS token as image embedding
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
            
            return embeddings
        except Exception as e:
            print(f"Error extracting embeddings from base64 image: {str(e)}")
            return None
    
    # Keeping the original methods as fallback
    def _filter_evidence_by_image_similarity_basic(self, evidence_list, reference_image, threshold=0.6, is_base64=False):
        """Original pixel-based image similarity as fallback."""
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
            similarity = self._calculate_basic_similarity(reference_features, evidence_features)
            
            print(f"Basic Similarity: {similarity}")
            
            # Add evidence if similarity is above threshold
            if similarity >= threshold:
                # Add similarity score to evidence for debugging/sorting
                evidence.similarity_score = similarity
                similar_evidence.append(evidence)
        
        return similar_evidence
        
    def _calculate_basic_similarity(self, features1, features2):
        """Calculate similarity using cosine similarity (fallback method)."""
        try:
            # Use cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Error calculating basic similarity: {e}")
            similarity = 0.0
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))

    def _extract_image_features(self, image_path: str):
        """Extract basic features from an image file (fallback method)."""
        try:
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
        """Extract basic features from a base64-encoded image string (fallback method)."""
        try:
            if isinstance(base64_string, str) and base64_string.startswith('data:image/'):
                base64_string = base64_string.split(';base64,', 1)[1]
                    
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
            
            if (evidence.title == "" or evidence.caption == "") and evidence.content == "":
                print(f"Skipping evidence: {evidence.title} {evidence.caption} {evidence.content}")
                continue
            
            if evidence.title == "" or evidence.caption == "" or evidence.content == "":
                print(f"Skipping evidence: {evidence.title} {evidence.caption} {evidence.content}")
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the ViT model and processor
        self._initialize_vit_model()
    
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
                        content=item.get('snippet', ''),
                        source="TextEvidencesModule"
                    ))
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading direct annotation file for index {idx}: {str(e)}")
            return []
        
        # Filter by domain first, then exclude certain domains
        # filtered_evidence = self.filter_evidence_by_domain(evidence_list, self.NEWS_DOMAINS)
        filtered_evidence = self.filter_evidence_by_excluding_domains(evidence_list, self.EXCLUDED_DOMAINS)
                
        # Filter by image similarity if reference image is provided
        if reference_image:
            filtered_evidence = self.filter_evidence_by_image_similarity(
                filtered_evidence, 
                reference_image, 
                image_similarity_threshold, 
                is_base64=True
            )
        
        # Filter by similarity to query
        if query != "":
            evidence_scores = self.filter_by_similarity(query, evidence_list, threshold)
            
            # If we don't have enough results meeting the threshold, include top results
            if len(evidence_scores) < min_results:
                evidence_scores = self.filter_by_similarity(query, evidence_list, threshold=0.0)[:min_results]
            
            # Get just the evidence objects, scores no longer needed
            filtered_evidence = [ev for ev, _ in evidence_scores]
            
        # Filter by unique domain+title combinations
        filtered_evidence = self.filter_unique_by_domain_title(filtered_evidence)
        
        # Filter to max_evidences based on unique titles
        final_evidence = self.filter_evidences(max_results, filtered_evidence)
        
        return final_evidence

class ImageEvidencesModule(BaseEvidencesModule):
    """Evidences for image search without loading actual images"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the ViT model and processor
        self._initialize_vit_model()
        
    def get_entities_by_index(self, index: Union[int, str]) -> List[str]:
        """Retrieve entities for a specific image index."""
        folder_path = self.get_item_folder_path(index)
        print(f"Folder path: {folder_path}")
        if not folder_path:
            return []
        
        try:
            annotation_file = os.path.join(folder_path, "inverse_annotation.json")
            with open(annotation_file, 'r', encoding='utf-8') as file:
                annotation_data = json.load(file)
                
            return annotation_data.get("entities", [])
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading inverse annotation file for index {index}: {str(e)}")
            return []
    
    def get_evidence_by_index(self, index: Union[int, str], query: str = "",
                             max_results: int = 5, reference_image: str = None,
                             image_similarity_threshold: float = 0.95) -> List[Evidence]:
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
            with open(annotation_file, 'r', encoding='utf-8') as file:
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
                'all_matched_captions',
                'partially_matched_no_text',
                'matched_no_text'
            ]
            
            for category in categories:
                for item in annotation_data.get(category, []):
                    image_data = self._load_and_encode_image(item.get("image_path", None))
                    if not image_data:
                        continue
                    
                    evidence_list.append(Evidence(
                        domain=extract_domain(item.get('page_link', '')),
                        image_path=item.get('image_link', ''),
                        image_data=image_data,  # Empty image data
                        title=item.get('title', ''),
                        caption=extract_caption(item.get('caption')),
                        content=get_content(item),
                        source="ImageEvidencesModule"
                    ))
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading inverse annotation file for index {index}: {str(e)}")
            return []
        
        # filtered_evidence = self.filter_evidence_by_domain(evidence_list, self.NEWS_DOMAINS)
        
        # Filter by excluding domains
        filtered_evidence = self.filter_evidence_by_excluding_domains(evidence_list, self.EXCLUDED_DOMAINS)
        # filtered_evidence = evidence_list
        
        print(len(filtered_evidence))
        
        if reference_image:
            filtered_evidence = self.filter_evidence_by_image_similarity(
                filtered_evidence, 
                reference_image, 
                image_similarity_threshold, 
                is_base64=True
            )
        
        # Filter evidences with the same domain and title
        filtered_evidence = self.filter_unique_by_domain_title(filtered_evidence)

        # Filter to max_evidences based on unique titles
        final_evidence = self.filter_evidences(max_results, filtered_evidence)
        
        return final_evidence