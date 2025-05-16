from functools import lru_cache
import json
import os
from typing import Union

from .base_evidences import BaseEvidencesModule, Evidence

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
            # if os.path.exists(selenium_annotation_file):
            #     print(f"Found {selenium_annotation_file}")
            #     with open(selenium_annotation_file, 'r') as file:
            #         selenium_data = json.load(file)
            
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
                    content = item.get('snippet', '')
                    content = self.clean_text(content)
                    
                    if (caption == "" and item.get('title', '') == "") or (content == ""):
                        continue
                    
                    evidence_list.append(Evidence(
                        domain=item.get('domain', ''),
                        image_path=image_path,
                        image_data=image_data,
                        title=item.get('page_title', ''),
                        caption=caption,
                        # Reduce content to 30000 words to reduce maximum tokens error
                        content=content,
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