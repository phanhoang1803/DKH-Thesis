from typing import Any, Dict, List, Optional
import json
from openai import OpenAI

VISION_SYSTEM_PROMPT = """You are a highly accurate and logical image verification assistant. Your primary role is to evaluate whether an input news image is supported or aligned with reference images by analyzing visual elements, context, and temporal consistency.

### **Responsibilities**
1. Analyze the input news image in detail, considering both its visual elements and context.
2. Compare against reference images to validate:
   - **Visual Consistency**: Check alignment of scenes, objects, people, and settings
   - **Temporal Consistency**: Verify timeline consistency across images
   - **Contextual Accuracy**: Ensure the overall context matches between images
3. Address any visual discrepancies or manipulations explicitly
4. Provide a comprehensive evaluation of the image's authenticity and alignment with reference images

### **Validation Criteria**
- **Visual Accuracy**: Confirm visual elements match across images
- **Temporal Consistency**: Ensure chronological alignment
- **Context**: Verify the broader scene and setting consistency
- **Manipulation**: Identify any signs of digital alterations

Your analysis should strictly follow the output format requirements and focus on concrete visual evidence."""

VISION_FINAL_SCHEMA = {
   "type": "object",
   "required": ["verdict", "alignment_score", "confidence_score", "explanation", "key_observations"],
   "properties": {
       "verdict": {
           "type": "boolean",
           "description": "Whether the news image significantly aligns with reference images (True/False)"
       },
       "alignment_score": {
           "type": "integer",
           "description": "Score from 0 to 100 indicating how well the images align visually and contextually"
       },
       "confidence_score": {
           "type": "integer",
           "description": "Confidence in the analysis (0-10)"
       },
       "explanation": {
           "type": "string",
           "description": "Comprehensive explanation of the visual comparison (up to 500 words)"
       },
       "key_observations": {
           "type": "string", 
           "description": "Specific points about matching elements, discrepancies, temporal indicators, and manipulation signs"
       }
   }
}

class GPTVisionConnector:
    def __init__(self, api_key: str, model_name: str = "gpt-4-vision-0125"):
        """Initialize the GPT Vision connector."""
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def call_with_structured_output(
            self,
            prompt: str,
            schema: Dict[str, Any],
            image_base64: Optional[str] = None,
            ref_images_base64: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
        """
        Call GPT-4 Vision with structured output using the provided schema.
        
        Args:
            prompt (str): The prompt to send to the model
            schema (Dict[str, Any]): JSON schema for structured output
            image_base64 (Optional[str]): Base64 encoded main image
            ref_images_base64 (Optional[List[str]]): List of base64 encoded reference images
            
        Returns:
            Dict[str, Any]: Structured response following the provided schema
        """
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # Add main image if provided
        if image_base64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })

        # Add reference images if provided
        if ref_images_base64:
            for ref_image in ref_images_base64:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ref_image}"
                    }
                })

        # Call GPT-4 Vision API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            functions=[{
                "name": "analyze_images",
                "parameters": schema
            }],
            function_call={"name": "analyze_images"}
        )
        
        # Parse and return the analysis results
        return json.loads(response.choices[0].message.function_call.arguments)
