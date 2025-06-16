import os
import json
from typing import List, Dict

class EvidenceSummarizer:
    def __init__(self, vlm_connector, llm_connector=None):
        self.vlm_connector = vlm_connector
        self.llm_connector = llm_connector

    async def generate_verification_questions(self, caption: str) -> List[str]:
        """Generate neutral questions from caption claims"""
        
        prompt = f"""
        Caption: "{caption}"
        
        Generate questions that would help verify BOTH the specific claims AND the broader context.
        Create questions at different levels:
        
        1. SPECIFIC CLAIM questions (directly from caption)
        2. CONTEXTUAL questions (broader event/situation)
        3. VERIFICATION questions (evidence that would confirm/refute)
        
        Example:
        Caption: "Climate activists block traffic in London during rush hour in October 2024"
        
        Questions should include:
        - Is there evidence of traffic being blocked or disrupted?
        - What indicates this is a climate-related protest (signs, symbols, messages)?
        - What suggests this is London (architecture, road signs, vehicle types)?
        - What indicates the time is rush hour (traffic density, lighting, commuter presence)?
        - What seasonal/weather indicators match October?
        - Are there any visible dates, news crews, or police presence suggesting a notable event?
        - What is the scale and organization level of this gathering?
        
        Generate 5-7 questions that go beyond surface description to event understanding.
        """
        
        connector = self.llm_connector if self.llm_connector else self.vlm_connector
        response = await connector.call_with_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
                "required": ["questions"]
            }
        )
        return response["questions"]

    async def generate_image_information_paragraph(self, image_base64: str, questions: List[str], visual_entities: List[str]) -> str:
        """Use questions to guide VLM in creating comprehensive paragraph"""
        
        prompt = f"""
        Analyze this image and the detected entities and write ONE detailed paragraph addressing these aspects:
        Detected entities in the image (Quite important): {', '.join(visual_entities)}
        
        {chr(10).join(f"- {q}" for q in questions)}
        
        Important instructions:
        - Describe what you ACTUALLY see, not what you might expect
        - Note when something asked about is NOT visible or unclear
        - Include all relevant details that help answer the questions
        - Mention any text, signs, or identifying markers exactly as shown
        - Use "appears to be" or "possibly" for uncertainties
        
        Write a comprehensive paragraph covering all these points based solely on visual evidence.
        """
        
        system_prompt = """
        You are a neutral observer. Describe ONLY what is visible in the image.
        Address the questions but base everything on actual visual evidence.
        Do not make assumptions beyond what you can see.
        """
        
        response = await self.vlm_connector.call_with_structured_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "A coherent and contextually attuned content from the image"
                    }
                },
                "required": ["content"]
            },
            images=[image_base64],
            system_prompt=system_prompt
        )
        
        return response["content"]

    async def summarize_evidence(self, evidences: List[Dict], image_base64: str, caption: str, visual_entities: List[str], image_information_save_dir: str=None, index: int=None) -> str:
        """Summarize the evidence using VLM"""
    
        if evidences == []:
            print(f"No evidence found for index {index}")
            if os.path.exists(os.path.join(image_information_save_dir, f"{index}.json")):
                print(f"Loading image information from {os.path.join(image_information_save_dir, f'{index}.json')}")
                with open(os.path.join(image_information_save_dir, f"{index}.json"), "r") as f:
                    image_information = json.load(f)
                if "content" in image_information:
                    return image_information["content"]
                else:
                    return image_information["detailed_description"]
            
            # Generate verification questions
            questions = await self.generate_verification_questions(caption)
            
            # Generate image information paragraph
            image_information_paragraph = await self.generate_image_information_paragraph(image_base64, questions, visual_entities)
            
            # Save the image information paragraph
            os.makedirs(image_information_save_dir, exist_ok=True)
            with open(os.path.join(image_information_save_dir, f"{index}.json"), "w") as f:
                json.dump({"content": image_information_paragraph}, f, indent=2, ensure_ascii=False)
            
            return image_information_paragraph
        
        # The existing code for when there is relevant evidence
        text = ""
        for evidence in evidences:
            evidence_caption = evidence.caption if evidence.caption else ""
            if evidence.content != None:
                evidence_content = evidence.content[:2000]
            else:
                evidence_content = ""

            evidence_text = f"Title: {evidence.title} \n\n Image Caption: {evidence_caption}" + f"\n\nContent: {evidence_content}"
            text += evidence_text + "\n\n"
        
        text = text.strip()
        
        # System prompt for evidence rewriting
        system_prompt = """
        You are a helpful assistant that rewrites the textual evidence into a coherent form.
        """
        
        # Prompt to rewrite the evidence text into a coherent form
        rewrite_prompt = f"""
        Now I give you the authentic image and its textual evidence.
        
        TEXTUAL EVIDENCE:
        {text}
        
        REMEMBER: DO NOT describe the image, should be only the information from the textual evidence. Pay more attention to the **CAPTION** but don't forget the title and content.
        Please help me generate a coherent and contextually attuned content from the textual evidence. 
        """
        
        # Get rewritten evidence
        vlm_connector = self.vlm_connector
        if vlm_connector:
            rewritten_evidence_response = await vlm_connector.call_with_structured_output(
                prompt=rewrite_prompt,
                schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "A coherent and contextually attuned content"
                    }
                },
                "required": ["content"]
            },
            images=[image_base64],
            system_prompt=system_prompt
        )
        
        # Extract just the content from the response
        return rewritten_evidence_response["content"]

