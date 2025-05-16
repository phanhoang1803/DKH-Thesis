from typing import Dict, List, Optional, Union

class DebateAgent:
    """Agent that participates in the debate, believing it's interacting with a human"""
    
    def __init__(self, agent_id: str, vlm_connector=None, system_prompt: str = "You are a fact-checker analyzing whether an image-caption pair constitutes misinformation. You will be provided with related image information to help you make your determination."):
        self.agent_id = agent_id
        self.vlm_connector = vlm_connector
        self.initial_belief = None
        self.system_prompt = system_prompt
    
    def reset(self):
        """Reset the agent's state"""
        self.initial_belief = None
    
    def update_vlm_connector(self, vlm_connector):
        self.vlm_connector = vlm_connector
    
    def form_initial_opinion(self, caption: str, evidence: Dict, image_base64: str = None, retrieval_result: Dict = None):
        """Form initial opinion about whether the caption is misinformation"""
        prompt = self._construct_initial_prompt(caption, evidence, retrieval_result)
        
        response = self.vlm_connector.call_with_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "opinion": {
                        "type": "string", 
                        "enum": ["YES", "NO"],
                        "description": "A final verdict of YES or NO"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "A confidence score between 0 and 1"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "A detailed explanation of your reasoning"
                    }
                },
                "required": ["opinion", "confidence", "reasoning"]
            },
            images=[image_base64] if image_base64 else None,
            system_prompt=self.system_prompt,
        )
        
        self.initial_belief = response
        
        return {
            "agent_id": self.agent_id,
            "opinion": response["opinion"],
            "confidence": response["confidence"],
            "reasoning": response["reasoning"]
        }
    
    def debate_round(self, opponent_response: Dict, caption: str, evidence: Dict, image_base64: str = None, round_num: int = 1):
        """Respond to opponent's argument in a debate round"""
        # Construct prompt based on debate history
        prompt = self._construct_debate_prompt(opponent_response, caption, evidence, round_num)
        
        system_prompt = """
        You are debating whether an image-caption pair constitutes misinformation. Analyze the evidence and your opponent’s reasoning critically. Defend your original stance unless compelling evidence suggests otherwise. Identify flaws or missing information, and cite specifics. Avoid agreeing without justification.
        """
        
        # Make VLM call
        kwargs = {
            "prompt": prompt,
            "schema": {
                "type": "object",
                "properties": {
                    "opinion": {
                        "type": "string", 
                        "enum": ["YES", "NO"],
                        "description": "A final verdict of YES or NO"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "A confidence score between 0 and 1"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "A detailed explanation of your reasoning"
                    }
                },
                "required": ["opinion", "confidence", "reasoning"]
            },
            "system_prompt": system_prompt,
        }
        
        # Add images for first round only
        if round_num == 1 and image_base64:
            kwargs["images"] = [image_base64]
        
        response = self.vlm_connector.call_with_structured_output(**kwargs)
        
        return {
            "agent_id": self.agent_id,
            "opinion": response["opinion"],
            "confidence": response["confidence"],
            "reasoning": response["reasoning"]
        }
    
    def _construct_initial_prompt(self, caption: str, evidence: Dict, retrieval_result: Dict) -> str:
        """Construct the initial prompt for opinion formation"""
        prompt = f"""
        This is a summary of news articles (scraped from the internet using the vision search) related to the image: {evidence['summary']}
        
        Visual entities identified in the image: {', '.join(evidence['visual_entities'])}
        Textual entities in the caption: {', '.join(evidence['textual_entities'])}
        **External information retrieval assessment (inconsistency check result between the caption and external information)**: {retrieval_result["assessment"]}
        
        Based on this, you need to decide if the caption given below represents the image 
        or if it is being used to spread false information to mislead people.
        
        CAPTION: {caption}
        
        Note that the image is real. It has not been digitally altered. Captions typically don't describe everything visible in an image. 
        The absence of mentioning certain elements in the caption is NOT automatically misinformation. 
        Focus instead on whether what IS stated in the caption conflicts with or misrepresents what's in the image or known facts.
        
        Consider the possibility that the image might be a scene from a TV show, film, advertisement, or other media production. In such cases, the caption might describe the broader narrative or context of the scene, which might extend beyond the specific visual elements captured in the frame.
        
        Carefully examine the evidence for any known entities, people, watermarks, dates, landmarks, 
        flags, text, logos and other details which could give you important information to better explain your answer. 
        
        The goal is to correctly identify if this image caption pair is misinformation or not 
        and to explain your answer in detail. Be specific about what aspects make you believe it is 
        or isn't misinformation.
        
        At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
        """
        return prompt
    
    def _construct_debate_prompt(self, opponent_response: Dict, caption: str, 
                               evidence: Dict, round_num: int):
        """Construct prompt for debate rounds"""
        
        information = f"""
        Visual entities identified in the image: {', '.join(evidence['visual_entities'])}
        Textual entities in the caption: {', '.join(evidence['textual_entities'])}
        Summary of news articles related to the image: {evidence['summary']}
        Caption: {caption}
        """
        
        if round_num > 1:
            # For rounds after round 1
            prompt = f"""
            Original opinion: {self.initial_belief["opinion"]}
            Original reasoning: {self.initial_belief["reasoning"]}
            
            Topic information: {information}
            
            I see what you mean and this is what I think: {opponent_response['reasoning']}
            
            Do you agree with me? If not then point out the inconsistencies in my argument (e.g. location, 
            time or person related logical confusion) and explain why you are correct. 
            
            If you disagree with me then clearly state why and what information I am overlooking. 
            Find disambiguation in my answer if any and ask questions to resolve them. 
            
            I want you to help me improve my argument and explanation. Don't give up your original opinion 
            without clear reasons, DO NOT simply agree with me without proper reasoning.
            
            At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """
        else:
            # For round 1
            prompt = f"""
            Topic information: {information}
            
            This is what I think: {opponent_response['reasoning']}
            
            Do you agree with me? If you think I am wrong then convince me why you are correct. 
            Clearly state your reasoning and tell me if I am missing out on some important information 
            or am making some logical error. Do not describe the image.
            
            If you disagree with me then clearly state why and what information I am overlooking. 
            Find disambiguation in my answer if any and ask questions to resolve them. 
            
            I want you to help me improve my argument and explanation. Don't give up your original opinion 
            without clear reasons, DO NOT simply agree with me without proper reasoning.
            
            At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """
        return prompt