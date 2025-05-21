import json
import os
from typing import Union, List
from urllib.parse import urlparse

from .base_evidences import BaseEvidencesModule, Evidence

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
                text = item.get('content', '') or item.get('snippet', '')
                text = self.clean_text(text)
                return text
            
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
                    
                    # For image evidence module, if image_data is None, set score to 0.8.
                    # if not image_data:
                    #     continue
                    
                    html_path = item.get('html_path', '')
                    html_content = self._load_html_content(html_path)
                    
                    content = get_content(item)
                    
                    if (extract_caption(item.get('caption')) == "" and item.get('title', '') == "") or content == "":
                        continue
                    
                    evidence_list.append(Evidence(
                        domain=extract_domain(item.get('page_link', '')),
                        image_path=item.get('image_link', ''),
                        image_data=image_data,
                        title=item.get('title', ''),
                        caption=extract_caption(item.get('caption')),
                        content=content,
                        html_content=html_content,
                        source="ImageEvidencesModule"
                    ))
            
            for item in annotation_data.get("newspaper", []):
                image_data = self._load_and_encode_image(item.get("image_path", None))
                
                html_path = item.get('html_path', '')
                html_content = self._load_html_content(html_path)
                
                content = get_content(item)
                
                if (extract_caption(item.get('caption')) == "" and item.get('title', '') == "") or content == "":
                    continue
                
                evidence = Evidence(
                    domain=extract_domain(item.get('page_link', '')),
                    image_path=item.get('image_link', ''),
                    image_data=image_data,
                    title=item.get('title', ''),
                    caption=extract_caption(item.get('caption')),
                    content=content,
                    html_content=html_content,
                    source="ImageEvidencesModule"
                )
                
                if self._is_included_domain(evidence.domain, self.NEWS_DOMAINS):
                    evidence_list.append(evidence)
            
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
        
        # Apply all filtering steps in a single pass
        filtered_evidence = []
        for ev in evidence_list:
            # Check if it's English
            if not self._is_english(ev):
                continue
                
            # Check domain filters
            if use_filter_by_excluding_domains and self._is_excluded_domain(ev.domain, self.EXCLUDED_DOMAINS):
                continue
                
            if use_filter_by_domains and not self._is_included_domain(ev.domain, self.NEWS_DOMAINS):
                continue
                
            # Check caption filter
            if use_filter_non_captions and ev.caption == "":
                continue
                
            # Set default scores
            ev.image_similarity_score = 0.0
            ev.text_similarity_score = 0.0
            
            filtered_evidence.append(ev)
        
        # Calculate image similarity scores if reference image is provided
        if reference_image:
            reference_embeddings = self._extract_embeddings_from_base64(reference_image)
            for evidence in filtered_evidence:
                if evidence.image_data is None or evidence.image_data == "":
                    # Because evidence from image evidence module used google image search, so image are similar to reference image, so in case evidence cant scrape image, set score to 0.8
                    evidence.image_similarity_score = 0.8
                    continue
                
                # Extract embeddings from evidence image
                evidence_embeddings = self._extract_embeddings_from_base64(evidence.image_data)
                
                if evidence_embeddings is not None and reference_embeddings is not None:
                    # Calculate image similarity score
                    evidence.image_similarity_score = self._calculate_image_similarity(reference_embeddings, evidence_embeddings)
        
        # Calculate text similarity scores if query is provided
        if query and query != "":
            # Prepare lists of texts to compare
            # if ev.caption else ev.title 
            texts = [ev.caption for ev in filtered_evidence]
            
            # Calculate similarities in batch
            similarities = self.batch_similarity(query, texts)
            
            # Assign text similarity scores to evidence objects
            for i, evidence in enumerate(filtered_evidence):
                evidence.text_similarity_score = float(similarities[i]) if i < len(similarities) else 0.0
        
        # Calculate combined scores
        for evidence in filtered_evidence:
            vs = evidence.image_similarity_score
            ts = evidence.text_similarity_score
            # Combined score = a*VS + b*TS + c*VS*TS
            evidence.combined_score = a * vs + b * ts + c * vs * ts
        
        # Sort by combined score (highest first)
        filtered_evidence.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Return top max_results
        return filtered_evidence[:max_results]
