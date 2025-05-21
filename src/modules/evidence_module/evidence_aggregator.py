from modules.entities_module import TextualEntityExtractor
from modules.entities_module import EntityAligner
from modules.evidence_module import ImageEvidencesModule, TextEvidencesModule
from .evidence_reranker import EvidenceReranker

class EvidenceAggregator:
    def __init__(self, image_evidences_module: ImageEvidencesModule, text_evidences_module: TextEvidencesModule, vlm_connector):
        self.image_evidences_module = image_evidences_module
        self.text_evidences_module = text_evidences_module
        self.textual_entity_extractor = TextualEntityExtractor(
            model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
            tokenizer_name="dbmdz/bert-large-cased-finetuned-conll03-english"
        )
        self.textual_entity_extractor.connect()
        self.entity_aligner = EntityAligner()
        self.reranker = EvidenceReranker(vlm_connector=vlm_connector)
        
    def get_aggregated_evidence(self, index: int, caption: str, image_base64: str):
        # Get visual entities from the image
        visual_entities = self.image_evidences_module.get_entities_by_index(index)

        # Get textual entities from the caption
        textual_entities = self.textual_entity_extractor.extract_textual_entities(caption)

        # Align the entities
        # aligned_visual_entities, aligned_textual_entities = self.entity_aligner.align_entities(visual_entities, textual_entities)

        # Get evidence using combined similarity scoring
        image_evidence = self.image_evidences_module.get_evidence_by_index(
            index, 
            query=caption, 
            reference_image=image_base64, 
            max_results=1, 
            a=0.4,    # Weight for visual similarity
            b=0.4,     # Weight for text similarity
            c=0.2     # Weight for interaction term
        )

        # Get evidence using combined similarity scoring
        text_evidence = self.text_evidences_module.get_evidence_by_index(
            index,
            query=caption,
            reference_image=image_base64, 
            max_results=1, 
            a=0.4,    # Weight for visual similarity
            b=0.4,     # Weight for text similarity
            c=0.2     # Weight for interaction term
        )

        # For text evidence, let's remove the evidence have low combined score
        text_evidence = [ev for ev in text_evidence if ev.image_similarity_score > 0.80]

        evidences = image_evidence
        for ev in text_evidence:
            # Check if the evidence is already in the image evidence, (mean same title or caption)
            if not any(ev.title == existing_ev.title or ev.caption == existing_ev.caption for existing_ev in image_evidence):
                evidences.append(ev)
        
        # Rerank evidences
        # reranked_evidences = self.reranker.rerank(evidences, reference_image=image_base64)
        reranked_evidences = evidences
        
        result = {
            "visual_entities": visual_entities,
            "textual_entities": textual_entities,
            
            # "aligned_visual_entities": aligned_visual_entities,
            # "aligned_textual_entities": aligned_textual_entities,
            
            "aligned_visual_entities": None,
            "aligned_textual_entities": None,
            
            "evidences": evidences,
            "reranked_evidences": reranked_evidences
        }
        
        return result
    
    def get_aggregated_evidence_with_vlm_ranking(self, index: int, caption: str, image_base64: str):
        # Get visual entities from the image
        visual_entities = self.image_evidences_module.get_entities_by_index(index)

        # Get textual entities from the caption
        textual_entities = self.textual_entity_extractor.extract_textual_entities(caption)

        # Align the entities
        # aligned_visual_entities, aligned_textual_entities = self.entity_aligner.align_entities(visual_entities, textual_entities)

        evidences = self.text_evidences_module.get_evidence_by_index(
            index,
            query=caption,
            reference_image=image_base64, 
            max_results=3, 
            a=0.6,    # Weight for visual similarity
            b=0.2,     # Weight for text similarity
            c=0.2     # Weight for interaction term
        )

        # Rerank evidences
        if len(evidences) > 0:
            reranked_evidences = self.reranker.rerank(evidences, reference_image=image_base64)
            print(f"Is accurate representation: {reranked_evidences[0].is_accurate_representation}")
        else:
            reranked_evidences = []
        
        result = {
            "visual_entities": visual_entities,
            "textual_entities": textual_entities,
            
            # "aligned_visual_entities": aligned_visual_entities,
            # "aligned_textual_entities": aligned_textual_entities,
            
            "aligned_visual_entities": None,
            "aligned_textual_entities": None,
            
            "evidences": evidences,
            "reranked_evidences": reranked_evidences
        }
        
        return result

