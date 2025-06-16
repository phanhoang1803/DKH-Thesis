import json
from typing import Dict, List, Optional, Union

class DebateAgent:
    """Agent that participates in the debate, believing it's interacting with a human"""
    
    def __init__(self, agent_id: str, vlm_connector=None, system_prompt: str = "", stance: str = None):
        self.agent_id = agent_id
        self.vlm_connector = vlm_connector
        self.system_prompt = system_prompt
        self.messages: List[Dict] = []
        self.initial_belief = None
        self.stance = stance
        
    def reset(self):
        """Reset the agent's state"""
        self.messages = []
        self.initial_belief = None
    
    def update_vlm_connector(self, vlm_connector):
        self.vlm_connector = vlm_connector
    
    async def form_initial_opinion(self, caption: str, evidence: Dict, image_base64: str = None, retrieval_result: Dict = None):
        """Form initial opinion about whether the caption is misinformation"""
        self.reset() # Ensure a clean slate for initial opinion
        
        prompt = self._construct_initial_prompt(caption, evidence, retrieval_result)
        self.messages.append({"role": "user", "content": prompt})
        
        response = await self.vlm_connector.call_with_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "opinion": {
                        "type": "string", 
                        "enum": ["YES", "NO"],
                        "description": "A final verdict of YES or NO based on the stance provided. Whether the image-caption pair is misinformation or not."
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
        
        self.messages.append({"role": "assistant", "content": json.dumps(response)})
        self.initial_belief = response
        
        return {
            "agent_id": self.agent_id,
            "opinion": response["opinion"],
            "confidence": response["confidence"],
            "reasoning": response["reasoning"]
        }
    
    def _construct_initial_prompt(self, caption: str, evidence: Dict, retrieval_result: Dict):
        """Construct the initial prompt for opinion formation, incorporating stance if provided."""
        
        summary = evidence.get('summary', 'No summary available.')
        visual_entities = ', '.join(evidence.get('visual_entities', []))
        textual_entities = ', '.join(evidence.get('textual_entities', []))
        retrieval_assessment = retrieval_result.get('assessment', 'No external retrieval assessment available.')

        # Base prompt
        prompt = f"""
        **Task:** Analyze the following image-caption pair to determine if it definitively constitutes misinformation. Your decision should be based *solely* on the provided evidence.

        **Provided Information:**

        Here is a summary of news articles related to the image (from vision search):
        {summary}
        
        Visual entities identified in the image: {visual_entities}
        Textual entities in the caption: {textual_entities}
        IMPORTANT INFORMATION: External information retrieval assessment (inconsistency check result between the caption and external information): {retrieval_assessment}
        
        CAPTION: {caption}
        
        ---

        """
        
        # Add stance guidance if stance is provided
        if self.stance is not None:
            if self.stance.upper() == 'YES':
                stance_guidance = f"""
        **CRITICAL STANCE - YOU BELIEVE THIS IS MISINFORMATION:** 
        You are STRONGLY predisposed to believe this content is misinformation. Your default assumption is that something is wrong with this image-caption pair.
        
        **Your approach should be:**
        - ACTIVELY look for inconsistencies, misrepresentations, and misleading context
        - Interpret ambiguous evidence as SUPPORTING misinformation
        - Demand OVERWHELMING proof to consider it legitimate
        - Focus on finding flaws, contradictions, and red flags
        - Be SKEPTICAL of explanations that suggest the content is accurate
        
        Only change your stance if the evidence is absolutely, undeniably, and overwhelmingly against misinformation.
        """
            else:
                stance_guidance = f"""
        **CRITICAL STANCE - YOU BELIEVE THIS IS LEGITIMATE:** 
        You are STRONGLY predisposed to believe this content is NOT misinformation. Your default assumption is that this image-caption pair is accurate and legitimate.
        
        **Your approach should be:**
        - ACTIVELY look for evidence supporting accuracy and legitimacy
        - Interpret ambiguous evidence as SUPPORTING authenticity
        - Demand OVERWHELMING proof to consider it misinformation
        - Focus on reasonable explanations and contextual justifications
        - Be SKEPTICAL of claims suggesting the content is misleading
        
        Only change your stance if the evidence is absolutely, undeniably, and overwhelmingly proving misinformation.
        """
            prompt += stance_guidance
            
        prompt += """
        
        **Critical Guidelines for Your Decision:**

        1.  **Strong Evidence for Misinformation is Required:** You must find *clear, explicit, and undeniable evidence* that the caption **misrepresents** the image, or that the image itself is used in a **demonstrably false context** to mislead.
        2.  **Default Assumption is Non-Misinformation:** If the evidence is ambiguous, inconclusive, or simply lacks sufficient information to definitively prove misinformation, you should lean towards classifying the pair as **NOT misinformation**. Do not infer misinformation from a lack of perfect alignment.
        3.  **Image Authenticity:** The image itself is assumed to be real and not digitally altered unless there is specific, direct evidence to the contrary within the provided information.
        4.  **Caption Scope:** Captions do not always describe every visible element. The absence of specific details in the caption or a lack of perfect one-to-one visual correspondence is **NOT** automatically a sign of misinformation. Focus strictly on whether what **IS stated** in the caption directly conflicts with or fundamentally misrepresents the image or known facts from the evidence.
        5.  **Contextual Nuances:** Consider if the image might be a scene from a TV show, film, advertisement, staged event, or other media production where the caption describes a broader narrative or context beyond the literal frame. In such cases, the caption might be accurate within its intended context, even if it doesn't describe every visible detail.
        6.  **Analyze Entities:** Carefully examine the evidence for any known entities, people, watermarks, dates, landmarks, flags, text, logos, and other details that could provide crucial information.
        
        **Your Goal:**
        Conclusively identify if this image-caption pair is **misinformation (YES)** or **not misinformation (NO)**. Provide a clear, detailed, and evidence-backed reasoning for your decision. Ensure your reasoning explicitly addresses the points above, especially if you classify it as 'YES' misinformation.
        """
        
        return prompt
    async def debate_round(self, opponent_response: Dict, caption: str, evidence: Dict, image_base64: str = None, round_num: int = 1):
        """Respond to opponent's argument in a debate round"""
        # Construct prompt based on debate history
        prompt = self._construct_debate_prompt(opponent_response, caption, evidence, round_num)
        self.messages.append({"role": "user", "content": prompt})
        
        system_prompt = """
        You are debating whether an image-caption pair constitutes misinformation. Analyze the evidence and your opponent's reasoning critically. Defend your original stance unless compelling evidence suggests otherwise. Identify flaws or missing information, and cite specifics. Avoid agreeing without justification.
        """
        
        # Only attach image for the very first round (round 1), subsequent rounds assume image is "in context"
        images_for_round = [image_base64] if round_num == 1 and image_base64 else None
        
        response = await self.vlm_connector.call_with_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "opinion": {
                        "type": "string", 
                        "enum": ["YES", "NO"],
                        "description": "A final verdict of YES (is misinformation) or NO (is not misinformation)"
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
            images=images_for_round,
            messages=self.messages,
            system_prompt=system_prompt,
        )
        
        self.messages.append({"role": "assistant", "content": json.dumps(response)})
        
        return {
            "agent_id": self.agent_id,
            "opinion": response["opinion"],
            "confidence": response["confidence"],
            "reasoning": response["reasoning"]
        }
    
    def _construct_debate_prompt(self, opponent_response: Dict, caption: str, 
                               evidence: Dict, round_num: int):
        """Construct prompt for debate rounds"""
        
        if round_num > 1:
            # For rounds after round 1
            prompt = f"""
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
            This is what I think: {opponent_response['reasoning']}
            
            Do you agree with me? If you think I am wrong then convince me why you are correct. 
            Clearly state your reasoning and tell me if I am missing out on some important information 
            or am making some logical error. Do not describe the image.
            
            At the end give a definite YES or NO answer to this question: IS THIS MISINFORMATION?
            """
        return prompt