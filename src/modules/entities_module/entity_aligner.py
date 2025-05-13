# entity_alignment.py
from typing import List, Dict, Tuple, Any
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from thefuzz import fuzz

class EntityAligner:
    """
    Class to align entities extracted from visual and textual sources.
    This alignment goes beyond simple string matching by considering:
    1. Semantic similarity using word embeddings
    2. Fuzzy string matching for handling variations in entity names
    3. Hypernym/hyponym relationships (e.g., "dog" is a type of "animal")
    4. Lemmatization to normalize word forms
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7, 
                 fuzzy_threshold: int = 80,
                 spacy_model: str = "en_core_web_md"):
        """
        Initialize the EntityAlignment class.
        
        Args:
            similarity_threshold (float): Threshold for semantic similarity (0-1)
            fuzzy_threshold (int): Threshold for fuzzy string matching (0-100)
            spacy_model (str): SpaCy model to use for word embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        
        # Initialize NLP tools
        try:
            self.nlp = spacy.load(spacy_model)
            self.lemmatizer = WordNetLemmatizer()
            # Ensure NLTK resources are available
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print(f"Successfully initialized EntityAlignment with {spacy_model}")
        except Exception as e:
            print(f"Failed to initialize NLP components: {e}")
            raise
    
    def _preprocess_entity(self, entity: str) -> str:
        """
        Preprocess entity text for better matching.
        
        Args:
            entity (str): Entity text to preprocess
            
        Returns:
            str: Preprocessed entity text
        """
        # Convert to lowercase and strip whitespace
        entity = entity.lower().strip()
        
        # Remove any special characters that might interfere with matching
        entity = ''.join(c for c in entity if c.isalnum() or c.isspace())
        
        return entity
    
    def _get_entity_text_from_ner_result(self, ner_result: Dict) -> str:
        """
        Extract entity text from a NER result dictionary.
        
        Args:
            ner_result (Dict): NER result dictionary from the TextualEntityExtractor
            
        Returns:
            str: The entity text
        """
        return ner_result.get('word', '')
    
    def _calculate_semantic_similarity(self, entity1: str, entity2: str) -> float:
        """
        Calculate semantic similarity between two entities using word embeddings.
        
        Args:
            entity1 (str): First entity text
            entity2 (str): Second entity text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            doc1 = self.nlp(entity1)
            doc2 = self.nlp(entity2)
            
            # If either entity doesn't have a vector, fall back to fuzzy matching
            if not doc1.has_vector or not doc2.has_vector:
                return fuzz.ratio(entity1, entity2) / 100
            
            return doc1.similarity(doc2)
        except Exception as e:
            print(f"Warning: Error calculating semantic similarity: {e}")
            # Fall back to fuzzy matching if semantic similarity fails
            return fuzz.ratio(entity1, entity2) / 100
    
    def _check_wordnet_relationship(self, entity1: str, entity2: str) -> bool:
        """
        Check if there's a hypernym/hyponym relationship between entities using WordNet.
        
        Args:
            entity1 (str): First entity text
            entity2 (str): Second entity text
            
        Returns:
            bool: True if there's a relationship, False otherwise
        """
        try:
            # Get lemmatized forms
            lemma1 = self.lemmatizer.lemmatize(entity1)
            lemma2 = self.lemmatizer.lemmatize(entity2)
            
            # Get synsets for both entities
            synsets1 = wordnet.synsets(lemma1)
            synsets2 = wordnet.synsets(lemma2)
            
            # Check for hypernym/hyponym relationships
            for syn1 in synsets1:
                for syn2 in synsets2:
                    # Check if syn1 is a hypernym of syn2 or vice versa
                    if syn1 in syn2.hypernyms() or syn2 in syn1.hypernyms():
                        return True
                    
                    # Check up to two levels of hypernyms
                    for hyper1 in syn1.hypernyms():
                        if hyper1 == syn2 or hyper1 in syn2.hypernyms():
                            return True
                    
                    for hyper2 in syn2.hypernyms():
                        if hyper2 == syn1 or hyper2 in syn1.hypernyms():
                            return True
            
            return False
        except Exception as e:
            print(f"Warning: Error checking WordNet relationship: {e}")
            return False
    
    def _are_entities_related(self, visual_entity: str, textual_entity: Dict) -> float:
        """
        Determine if a visual entity and a textual entity are related.
        
        Args:
            visual_entity (str): Visual entity text
            textual_entity (Dict): Textual entity dictionary from the TextualEntityExtractor
            
        Returns:
            float: Relation score between 0 and 1
        """
        textual_entity_text = self._get_entity_text_from_ner_result(textual_entity)
        
        # Preprocess both entities
        visual_entity_processed = self._preprocess_entity(visual_entity)
        textual_entity_processed = self._preprocess_entity(textual_entity_text)
        
        # Check for exact matches first (after preprocessing)
        if visual_entity_processed == textual_entity_processed:
            return 1.0
        
        # Calculate semantic similarity
        semantic_score = self._calculate_semantic_similarity(
            visual_entity_processed, textual_entity_processed
        )
        
        # Calculate fuzzy match score
        fuzzy_score = fuzz.ratio(visual_entity_processed, textual_entity_processed) / 100
        
        # Check for WordNet relationship
        wordnet_relation = self._check_wordnet_relationship(
            visual_entity_processed, textual_entity_processed
        )
        
        # Combine scores with weights
        # Semantic similarity: 50%, Fuzzy matching: 30%, WordNet: 20%
        combined_score = (semantic_score * 0.5) + (fuzzy_score * 0.3) + (int(wordnet_relation) * 0.2)
        
        return combined_score
    
    def align_entities(self, 
                       visual_entities: List[str], 
                       textual_entities: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Align visual and textual entities based on semantic similarity and other factors.
        
        Args:
            visual_entities (List[str]): List of visual entities from VisualEntityExtractor
            textual_entities (List[Dict]): List of textual entities from TextualEntityExtractor
            
        Returns:
            Tuple[List[str], List[Dict]]: Tuple containing aligned visual and textual entities
        """
        if not visual_entities or not textual_entities:
            print("Warning: Empty entity list provided for alignment")
            return [], []
        
        aligned_visual = []
        aligned_textual = []
        
        # Create a similarity matrix
        similarity_matrix = np.zeros((len(visual_entities), len(textual_entities)))
        
        # Calculate similarity scores for each pair
        for i, visual_entity in enumerate(visual_entities):
            for j, textual_entity in enumerate(textual_entities):
                similarity_matrix[i, j] = self._are_entities_related(visual_entity, textual_entity)
        
        # Find the best matches using the similarity matrix
        # First, try to find 1:1 matches above the threshold
        visual_matched = set()
        textual_matched = set()
        
        # Sort pairs by similarity score (highest first)
        pairs = []
        for i in range(len(visual_entities)):
            for j in range(len(textual_entities)):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    pairs.append((i, j, similarity_matrix[i, j]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Assign best matches first
        for i, j, score in pairs:
            if i not in visual_matched and j not in textual_matched:
                aligned_visual.append(visual_entities[i])
                aligned_textual.append(textual_entities[j])
                visual_matched.add(i)
                textual_matched.add(j)
        
        print(f"Aligned {len(aligned_visual)} entities")
        return aligned_visual, aligned_textual
    
    def align_and_combine(self, 
                        visual_entities: List[str], 
                        textual_entities: List[Dict]) -> Dict[str, List]:
        """
        Align entities and return a combined dictionary with both aligned and unaligned entities.
        
        Args:
            visual_entities (List[str]): List of visual entities from VisualEntityExtractor
            textual_entities (List[Dict]): List of textual entities from TextualEntityExtractor
            
        Returns:
            Dict[str, List]: Dictionary with 'aligned', 'unaligned_visual', and 'unaligned_textual' keys
        """
        aligned_visual, aligned_textual = self.align_entities(visual_entities, textual_entities)
        
        # Find unaligned entities
        unaligned_visual = [entity for entity in visual_entities if entity not in aligned_visual]
        unaligned_textual = [entity for entity in textual_entities if entity not in aligned_textual]
        
        return {
            'aligned_visual': aligned_visual,
            'aligned_textual': aligned_textual,
            'unaligned_visual': unaligned_visual,
            'unaligned_textual': unaligned_textual
        }


if __name__ == "__main__":
    # Example usage
    from visual_entity_extractor import VisualEntityExtractor
    from textual_entity_extractor import TextualEntityExtractor
    
    # Initialize extractors
    visual_extractor = VisualEntityExtractor("test_dataset/links_test.json")
    visual_entities = visual_extractor.get_entities_by_index(20)
    
    textual_extractor = TextualEntityExtractor(
        model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
        tokenizer_name="dbmdz/bert-large-cased-finetuned-conll03-english"
    )
    textual_extractor.connect()
    
    # Sample text related to the visual entities
    sample_text = "Donald Trump is the latest president of American"
    textual_entities = textual_extractor.extract_textual_entities(sample_text)
    print(f"Textual entities: {textual_entities}")
    
    
    # Align entities
    aligner = EntityAligner()
    aligned_visual, aligned_textual = aligner.align_entities(visual_entities, textual_entities)
    
    print(f"Aligned visual entities: {aligned_visual}")
    print(f"Aligned textual entities: {aligned_textual}")
    
    # Get combined results
    combined_results = aligner.align_and_combine(visual_entities, textual_entities)
    print(f"Combined results: {combined_results}")