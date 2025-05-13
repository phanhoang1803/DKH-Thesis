from typing import Dict, List, Optional, Union

class RetrievalAgent:
    def __init__(self, vlm_connector=None):
        self.vlm_connector = vlm_connector
        self.system_prompt = """You are a Retrieval Agent responsible for the first phase of misinformation detection.
        Your task is to compare the input news (caption) with retrieved evidence and flag any inconsistencies.
        Focus on comparing visual and textual entities, checking for alignment between what's described and what's shown.
        You should identify potential mismatches without making final judgments about whether the content is misinformation.
        """
    
    def analyze(self, caption: str, evidence: Dict, image_base64: str = None) -> Dict:
        # Construct the prompt for the Retrieval Agent
        prompt = self._construct_analysis_prompt(caption, evidence)
        
        # Make the call to the VLM
        response = self.vlm_connector.call_with_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "assessment": {
                        "type": "string",
                        "description": "A detailed assessment of the inconsistencies found if any"
                    }
                },
                "required": ["assessment"]
            },
            images=[image_base64] if image_base64 else None,
            system_prompt=self.system_prompt
        )
        
        return {
            "agent_id": "RetrievalAgent",
            "assessment": response["assessment"]
        }
    
    def _construct_analysis_prompt(self, caption: str, evidence: Dict) -> str:
        prompt = f"""
        As a Retrieval Agent, your task is to cross-reference the input news with the retrieved evidence and flag any inconsistencies.
        
        CAPTION TO ANALYZE: {caption}
        
        EVIDENCE INFORMATION:
        Visual entities identified in the image: {', '.join(evidence['visual_entities'])}
        Textual entities in the caption: {', '.join(evidence['textual_entities'])}
        Summary of news articles related to the image: {evidence['summary']}
        
        YOUR TASK:
        1. Compare the entities mentioned in the caption with those present in the image and evidence
        2. Identify any discrepancies between events described in the caption and what the evidence shows
        3. Check for inconsistencies in context (time, location, situation) between the caption and evidence
        4. Provide specific details for each inconsistency you find
        
        Focus on observable facts and objective comparisons rather than subjective interpretations.
    
        Provide your assessment in a detailed paragraph that begins with either:
        "There is no inconsistency between the caption and evidence <reasoning for this conclusion>"
        OR
        "There is a potential inconsistency between the caption and evidence <reasoning for this conclusion>"
        
        Follow this with specific details about any inconsistencies found and their nature.
        """ 
        return prompt