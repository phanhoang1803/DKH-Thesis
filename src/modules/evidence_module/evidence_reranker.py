class EvidenceReranker:
    def __init__(self, vlm_connector):
        self.vlm_connector = vlm_connector
        
    def rerank(self, evidences, reference_image=None):
        if not evidences:
            return []
            
        # Create reranking prompt if template is provided
        rerank_prompt = self._create_rerank_prompt(evidences)
        
        system_prompt = """
        You are an expert evidence selector and fact checker. 
        Your task is to analyze the image and evaluate the evidence.
        1. Select the most relevant evidence that directly relates to the visual content
        2. Determine whether this evidence truly represents accurate information about what's in the image
        Focus on factual information and verify whether the evidence correctly describes the image content.
        Be very critical - many pieces of evidence might appear relevant but contain misleading information,
        incorrect context, or inaccurate descriptions of what's actually shown in the image.
        """
        
        response = self.vlm_connector.call_with_structured_output(
            prompt=rerank_prompt,
            schema={
                "type": "object",
                "properties": {
                    "evidence_index": {
                        "type": "integer",
                        "description": "The index of the evidence that is most relevant to the authentic image"
                    },
                    "is_accurate_representation": {
                        "type": "boolean",
                        "description": "Whether the selected evidence contains accurate, truthful information that correctly represents the image information (true) or contains misleading, incorrect, or fabricated information about the image content (false)"
                    }
                },
                "required": ["evidence_index", "is_accurate_representation"]
            },
            images=[reference_image],
            system_prompt=system_prompt
        )
                
        reranked_evidences = [evidences[response["evidence_index"]]]
        # You can also store the relevance information in the evidence object if needed
        reranked_evidences[0].is_accurate_representation = response["is_accurate_representation"]
        
        return reranked_evidences
    
    def _create_rerank_prompt(self, evidences):
        evidence_texts = []
        for i, evidence in enumerate(evidences):
            evidence_caption = evidence.caption if evidence.caption else ""
            if evidence.content != None:
                evidence_content = evidence.content[:2000]
            else:
                evidence_content = ""

            evidence_text = f"Title: {evidence.title} \n\n Image Caption: {evidence_caption}" + f"\n\nContent: {evidence_content}"
            evidence_text = f"<Evidence {i}> {evidence_text} </Evidence {i}>"
            evidence_texts.append(evidence_text)
            
        # Combine all evidence texts
        all_evidences = "\n".join(evidence_texts)
        
        prompt = f"""
        Below are several pieces of evidence:

        {all_evidences}

        Look at the image carefully and analyze each piece of evidence to:
        1. Select the evidence that is most relevant to the image content
        2. Determine if this evidence is factually accurate about the image
        
        IMPORTANT:
        - Consider whether the evidence might be related but contain misleading information about what's actually in the image
        - Verify that the evidence correctly describes the real visual content of the image
        - Some evidence might describe what the image claims to be, not what it actually shows
        
        Select the single most relevant evidence index and determine if it accurately represents the image information.
                
        Your answer:
        """
        
        return prompt