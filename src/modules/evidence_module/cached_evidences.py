from functools import lru_cache
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
from PIL import Image
import io

from transformers import AutoImageProcessor, AutoModel
from langdetect import detect, LangDetectException

from newspaper import Article
import trafilatura

@dataclass
class Evidence:
    domain: str
    image_path: str
    image_data: str  # Base64 encoded image data
    title: str
    caption: str
    content: str
    source: Optional[str] = None
    
    def __init__(self, domain="", image_path="", image_data="", title="", caption="", content="", html_content="", source=""):
        self.domain = domain
        self.image_path = image_path
        self.image_data = image_data
        self.title = title
        self.caption = caption
        self.content = content
        self.html_content = html_content
        self.source = source
        self.image_similarity_score = 0.0
        self.text_similarity_score = 0.0
        self.combined_score = 0.0
    
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
            result = {
                "title": self._clean_text(self.title),
                "content": self._clean_text(self.content),
                "html_content": self._clean_text(self.html_content),
                "caption": self._clean_text(self.caption),
                "domain": self._clean_text(self.domain),
                "source": self.source 
            }
            if self.image_similarity_score:
                result["image_similarity_score"] = self.image_similarity_score
            if self.text_similarity_score:
                result["text_similarity_score"] = self.text_similarity_score
            if self.combined_score:
                result["combined_score"] = self.combined_score
            return result
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
            "mdpi", "yumpu", "scmp", "pinterest", "imdb",
            "movieweb", "shutterstock", "reddit", "alamy",
            "alamy.it", "alamyimages", "planetcricket",
            "cnnbrasil", "infomoney", "gettyimages",
        ]
    
    def _initialize_vit_model(self, model_ckpt="google/vit-base-patch16-224"):
        """Initialize the Vision Transformer model for semantic image similarity."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)
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
            if not os.path.exists(image_path):
                return ""
            
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
    
    def _load_html_content(self, html_path: str) -> str:
        """Load HTML content from a file."""
        try:
            raw_html = ""
            with open(html_path, 'r', encoding='utf-8') as file:
                raw_html = file.read()
            
            return trafilatura.extract(raw_html)
        
        except Exception as e:
            print(f"Error loading HTML content from {html_path}: {str(e)}")
            return ""
    
    def _calculate_image_similarity(self, embeddings1, embeddings2):
        """Calculate cosine similarity between two embedding vectors."""
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
        """Extract semantic embeddings from an image file using ViT."""
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
        """Extract semantic embeddings from a base64-encoded image string using ViT."""
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
            
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain string by removing www. prefix and lowercasing."""
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _normalize_domain_for_excluding(self, domain: str) -> str:
        """Normalize domain string by removing www. prefix and lowercasing."""
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        domain = domain.split(".")[0]
        return domain
    
    def filter_evidennce_by_domains(self, evidence_list: list[Evidence],
                                    domains: List[str]) -> List[Evidence]:
        return [ev for ev in evidence_list
                if self._normalize_domain(ev.domain) in domains]
    
    def filter_evidence_by_excluding_domains(self, evidence_list: List[Evidence], 
                                           excluded_domains: List[str]) -> List[Evidence]:
        return [ev for ev in evidence_list 
                if self._normalize_domain_for_excluding(ev.domain) not in excluded_domains]
    
    def filter_non_english_evidence(self, evidences: List[Evidence]):
        """Filter out non-English evidence."""
        # Detect language of title 
        english_evidences = []
        for evidence in evidences:
            try:
                # Check title language if it exists
                if evidence.title and len(evidence.title.strip()) > 10:  # Need some minimal text for reliable detection
                    title_lang = detect(evidence.title)
                    if title_lang != 'en':
                        continue
                
                english_evidences.append(evidence)
            except LangDetectException:
                # Skip evidence we can't classify
                continue
            
        return english_evidences
    
    def get_item_folder_path(self, index: Union[int, str]) -> Optional[str]:
        """Get the folder path for an item by index."""
        # Convert index to int if it's a string
        idx = int(index) if isinstance(index, str) else index
        
        item = self.data.get(str(idx))
        if not item:
            return None
            
        return item.get("folder_path")

    def get_evidence_by_index(self, index: Union[int, str], query: str = "",
                            max_results: int = 5, reference_image: str = None,
                            a: float = 1.0, b: float = 1.0, c: float = 0.5,
                            use_filter_by_excluding_domains: bool = True):
        """
        Get evidence for a specific index with combined scoring method.
        
        Args:
            index: The index to retrieve evidence for
            query: Optional text query for text similarity scoring
            max_results: Maximum number of results to return
            reference_image: Optional reference image for image similarity scoring
            a: Weight for visual similarity score
            b: Weight for text similarity score
            c: Weight for the interaction term (vs*ts)
            use_filter_by_excluding_domains: Whether to filter out excluded domains
            
        Returns:
            List of Evidence objects with combined scores
        """
        # This is a base implementation to be overridden by subclasses
        raise NotImplementedError("Subclasses must implement get_evidence_by_index")


class TextEvidencesModule(BaseEvidencesModule):
    """Evidences retrieved by using text search on Google"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the ViT model and processor
        self._initialize_vit_model()
    
    @lru_cache(maxsize=100)
    def get_raw_evidence_by_index(self, index: Union[int, str]):
        """Get raw evidence for a specific index."""
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
            
            parent_folder_path = os.path.dirname(folder_path)
            selenium_annotation_file = os.path.join(parent_folder_path, "selenium", str(index), "direct_annotation.json")
            
            selenium_data = None
            if os.path.exists(selenium_annotation_file):
                print(f"Found {selenium_annotation_file}")
                with open(selenium_annotation_file, 'r') as file:
                    selenium_data = json.load(file)
            
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
                        # Return the longer one
                        caption = caption_node if len(caption_node) > len(alt_node) else alt_node
                    else:
                        caption = caption_node or alt_node

                else:
                    caption = caption_data  # If caption_data is a string, use it directly

                # Extract text before "Photograph"
                for marker in ["Photograph:", "Photo:"]:
                    caption = caption.split(marker)[0].strip()

                return caption
                
            # Process all image categories
            image_categories = [
                'images_with_captions', 
                'images_with_no_captions',
                'images_with_caption_matched_tags'
            ]
            
            for category in image_categories:
                items = annotation_data.get(category, [])
                if selenium_data:
                    items = items + selenium_data.get(category, [])
                    
                for item in items:
                    image_path = item.get('image_path', '')
                    image_data = self._load_and_encode_image(image_path)
                    if not image_data:
                        continue
                    
                    html_path = item.get('html_path', '')
                    html_content = self._load_html_content(html_path)
                    
                    caption = extract_caption(item.get('caption', ''))
                    
                    if caption == "" and item.get('title', '') == "":
                            continue
                    
                    evidence_list.append(Evidence(
                        domain=item.get('domain', ''),
                        image_path=image_path,
                        image_data=image_data,
                        title=item.get('page_title', ''),
                        caption=caption,
                        # Reduce content to 30000 words to reduce maximum tokens error
                        content=item.get('snippet', ''),
                        html_content=html_content,
                        source="TextEvidencesModule"
                    ))
             
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading direct annotation file for index {idx}: {str(e)}")
            return []
        
        return evidence_list
    
    def get_evidence_by_index(self, index: Union[int, str], query: str = "",
                            max_results: int = 5, reference_image: str = None,
                            a: float = 1.0, b: float = 1.0, c: float = 0.5,
                            use_filter_by_excluding_domains: bool = True):
        """
        Get evidence for a specific index with combined scoring method.
        
        Args:
            index: The index to retrieve evidence for
            query: Optional text query for text similarity scoring
            max_results: Maximum number of results to return
            reference_image: Optional reference image for image similarity scoring
            a: Weight for visual similarity score
            b: Weight for text similarity score
            c: Weight for the interaction term (vs*ts)
            use_filter_by_excluding_domains: Whether to filter out excluded domains
            
        Returns:
            List of Evidence objects with combined scores
        """
        # Get all raw evidence
        evidence_list = self.get_raw_evidence_by_index(index)
        
        # Filter non-English evidence
        evidence_list = self.filter_non_english_evidence(evidence_list)
        
        # Only apply excluding domains filter if requested
        if use_filter_by_excluding_domains:
            evidence_list = self.filter_evidence_by_excluding_domains(evidence_list, self.EXCLUDED_DOMAINS)
        
        # Calculate image similarity scores if reference image is provided
        if reference_image:
            for evidence in evidence_list:
                # Extract embeddings from evidence image
                evidence_embeddings = self._extract_embeddings_from_base64(evidence.image_data)
                reference_embeddings = self._extract_embeddings_from_base64(reference_image) if reference_image else None
                
                if evidence_embeddings is not None and reference_embeddings is not None:
                    # Calculate image similarity score
                    evidence.image_similarity_score = self._calculate_image_similarity(reference_embeddings, evidence_embeddings)
                else:
                    evidence.image_similarity_score = 0.0
        else:
            # Set default image similarity score to 0 if no reference image
            for evidence in evidence_list:
                evidence.image_similarity_score = 0.0
        
        # Calculate text similarity scores if query is provided
        if query and query != "":
            # Prepare lists of texts to compare
            captions = [ev.caption for ev in evidence_list]
            titles = [ev.title for ev in evidence_list]
            
            # Calculate similarities in batch
            caption_similarities  = self.batch_similarity(query, captions)
            title_similarities  = self.batch_similarity(query, titles)
            
            # Assign text similarity scores to evidence objects
            for i, evidence in enumerate(evidence_list):
                evidence.text_similarity_score = max(float(caption_similarities[i]), float(title_similarities[i]))
            
            # texts = [ev.caption for ev in evidence_list]
            
            # # Calculate similarities in batch
            # similarities = self.batch_similarity(query, texts)
            
            # # Assign text similarity scores to evidence objects
            # for i, evidence in enumerate(evidence_list):
            #     evidence.text_similarity_score = float(similarities[i]) if i < len(similarities) else 0.0
        else:
            # Set default text similarity score to 0 if no query
            for evidence in evidence_list:
                evidence.text_similarity_score = 0.0
        
        # Calculate combined scores
        for evidence in evidence_list:
            vs = evidence.image_similarity_score
            ts = evidence.text_similarity_score
            # Combined score = a*VS + b*TS + c*VS*TS
            evidence.combined_score = a * vs + b * ts + c * vs * ts
        
        # Sort by combined score (highest first)
        evidence_list.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Return top max_results
        return evidence_list[:max_results]


class ImageEvidencesModule(BaseEvidencesModule):
    """Evidences for image search without loading actual images"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the ViT model and processor
        self._initialize_vit_model()
        
    def get_entities_by_index(self, index: Union[int, str], threshold: float = 0.0, min_results: int = 0, return_scores: bool = False) -> List[str]:
        """Retrieve entities for a specific image index."""
        folder_path = self.get_item_folder_path(index)
        if not folder_path:
            return []
        
        try:
            annotation_file = os.path.join(folder_path, "inverse_annotation.json")
            with open(annotation_file, 'r', encoding='utf-8') as file:
                annotation_data = json.load(file)
                
            entities = annotation_data.get("entities", [])
            entities_scores = annotation_data.get("entities_scores", [])
            
            # If dont have entities scores, get first min_results entities
            if entities_scores is None or len(entities_scores) == 0:
                if return_scores:
                    return entities[:min_results], None
                return entities[:min_results]
            
            # Filter entities by threshold and min_results
            filtered_entities = [entity for entity, score in zip(entities, entities_scores) if score >= threshold]
            if len(filtered_entities) < min_results:
                filtered_entities = entities[:min_results]
                
            if return_scores:
                return filtered_entities, entities_scores[:len(filtered_entities)]
            return filtered_entities
        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading inverse annotation file for index {index}: {str(e)}")
            return []
    
    def get_raw_evidence_by_index(self, index: Union[int, str]):
        """Get raw evidence for a specific index."""
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
                        # Return the longer one
                        caption = caption_node if len(caption_node) > len(alt_node) else alt_node
                    else:
                        caption = caption_node or alt_node

                else:
                    caption = caption_data  # If caption_data is a string, use it directly

                for marker in ["Photograph:", "Photo:"]:
                    caption = caption.split(marker)[0].strip()

                return caption
                
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
                    
                    html_path = item.get('html_path', '')
                    html_content = self._load_html_content(html_path)
                    
                    if extract_caption(item.get('caption')) == "" and item.get('title', '') == "":
                        continue
                    
                    evidence_list.append(Evidence(
                        domain=extract_domain(item.get('page_link', '')),
                        image_path=item.get('image_link', ''),
                        image_data=image_data,
                        title=item.get('title', ''),
                        caption=extract_caption(item.get('caption')),
                        content=' '.join(get_content(item)[:30000]),
                        html_content=html_content,
                        source="ImageEvidencesModule"
                    ))
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading inverse annotation file for index {index}: {str(e)}")
            return []
        
        return evidence_list
    
    def get_evidence_by_index(self, index: Union[int, str], query: str = "",
                            max_results: int = 5, reference_image: str = None,
                            a: float = 1.0, b: float = 1.0, c: float = 0.5,
                            use_filter_by_domains: bool = False,
                            use_filter_by_excluding_domains: bool = True,
                            use_filter_non_captions: bool = False
                            ):
        """
        Get evidence for a specific index with combined scoring method.
        
        Args:
            index: The index to retrieve evidence for
            query: Optional text query for text similarity scoring
            max_results: Maximum number of results to return
            reference_image: Optional reference image for image similarity scoring
            a: Weight for visual similarity score
            b: Weight for text similarity score
            c: Weight for the interaction term (vs*ts)
            use_filter_by_excluding_domains: Whether to filter out excluded domains
            
        Returns:
            List of Evidence objects with combined scores
        """
        # Get all raw evidence
        evidence_list = self.get_raw_evidence_by_index(index)
        
        # Filter non-English evidence
        evidence_list = self.filter_non_english_evidence(evidence_list)
        
        # Only apply excluding domains filter if requested
        if use_filter_by_excluding_domains:
            evidence_list = self.filter_evidence_by_excluding_domains(evidence_list, self.EXCLUDED_DOMAINS)
        
        if use_filter_by_domains:
            evidence_list = self.filter_evidennce_by_domains(evidence_list, self.NEWS_DOMAINS)
        
        if use_filter_non_captions:
            evidence_list = [ev for ev in evidence_list if ev.caption != ""]
        
        # Calculate image similarity scores if reference image is provided
        if reference_image:
            for evidence in evidence_list:
                # Extract embeddings from evidence image
                evidence_embeddings = self._extract_embeddings_from_base64(evidence.image_data)
                reference_embeddings = self._extract_embeddings_from_base64(reference_image) if reference_image else None
                
                if evidence_embeddings is not None and reference_embeddings is not None:
                    # Calculate image similarity score
                    evidence.image_similarity_score = self._calculate_image_similarity(reference_embeddings, evidence_embeddings)
                else:
                    evidence.image_similarity_score = 0.0
        else:
            # Set default image similarity score to 0 if no reference image
            for evidence in evidence_list:
                evidence.image_similarity_score = 0.0
        
        # Calculate text similarity scores if query is provided
        if query and query != "":
            # Prepare lists of texts to compare
            # if ev.caption else ev.title 
            texts = [ev.caption for ev in evidence_list]
            
            # Calculate similarities in batch
            similarities = self.batch_similarity(query, texts)
            
            # Assign text similarity scores to evidence objects
            for i, evidence in enumerate(evidence_list):
                evidence.text_similarity_score = float(similarities[i]) if i < len(similarities) else 0.0
        else:
            # Set default text similarity score to 0 if no query
            for evidence in evidence_list:
                evidence.text_similarity_score = 0.0
        
        # Calculate combined scores
        for evidence in evidence_list:
            vs = evidence.image_similarity_score
            ts = evidence.text_similarity_score
            # Combined score = a*VS + b*TS + c*VS*TS
            evidence.combined_score = a * vs + b * ts + c * vs * ts
        
        # Sort by combined score (highest first)
        evidence_list.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Return top max_results
        return evidence_list[:max_results]