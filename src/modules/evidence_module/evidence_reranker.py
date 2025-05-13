class EvidenceReranker:
    def __init__(self, vlm_connector):
        self.vlm_connector = vlm_connector
        
    def rerank(self, evidences, reference_image=None):
        if not evidences:
            return []
            
        # Create reranking prompt if template is provided
        rerank_prompt = self._create_rerank_prompt(evidences)
        
        system_prompt = """
        You are an expert evidence selector for fact checking. 
        Your task is to analyze the image and select the most relevant evidence that directly relates to the visual content.
        Focus on factual information and specific entities visible in the image."""
        
        response = self.vlm_connector.call_with_structured_output(
            prompt=rerank_prompt,
            schema={
                "type": "object",
                "properties": {
                    "evidence_index": {
                        "type": "integer",
                        "description": "The index of the evidence that is most relevant to the authentic image"
                    }
                }
            },
            images=[reference_image],
            system_prompt=system_prompt
        )
        
        reranked_evidences = [evidences[response["evidence_index"]]]
        
        return reranked_evidences
    
    def _create_rerank_prompt(self, evidences):
        evidence_texts = []
        for i, evidence in enumerate(evidences):
            evidence_text = f"<Evidence {i}> {evidence.text if hasattr(evidence, 'text') else str(evidence)} </Evidence {i}>"
            evidence_texts.append(evidence_text)
            
        # Combine all evidence texts
        all_evidences = "\n".join(evidence_texts)
        
        # Format the full prompt
        # prompt = f"""
        # {all_evidences}
        # You should output 1 evidence index that can assist you most in image fact-checking,
        # select the most relevant textual evidence related to the authentic image.
        
        # Your answer:
        # """
        prompt = f"""
        Below are several pieces of evidence:

        {all_evidences}

        Look at the image carefully and select the evidence that:
        1. Contains information directly visible in the image
        2. Mentions specific names, places, or events shown in the image
        3. Provides the most relevant context about what's in the image

        Select the single most relevant evidence index that best helps with image fact-checking.
                
        Your answer:
        """
        
        return prompt