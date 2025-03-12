from typing import Any, Dict, Optional
import openai
import json

SYSTEM_PROMPT = """You are a fact-checking assistant evaluating news captions against image content.

Your responsibilities:
1. Analyze whether captions accurately represent their associated images
2. Base determinations primarily on check results and image evidence
3. Ensure consistency between your OOC determination and explanation
4. For high confidence check results (7-10), defer to the check result
5. Set OOC=false when captions correctly represent images, and OOC=true when they don't
"""

VISUAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "explanation", "confidence_score", "supporting_evidences"],
    "properties": {
        "verdict": {
            "type": "boolean",
            "description": "True if the evidence confirms the caption accurately represents the image; False otherwise."
        },
        "explanation": {
            "type": "string",
            "description": "A detailed explanation based on specific evidences and how it relates to the caption."
        },
        "confidence_score": {
            "type": "integer",
            "description": "A score from 0 (no confidence) to 10 (complete confidence) indicating how certain the verdict is."
        },
        "supporting_evidences": {
            "type": "array",
            "description": "List of specific evidence that supports the verdict",
            'items': {
                'type': 'string'
            }
        }
    }
}

TEXTUAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "explanation", "confidence_score", "supporting_evidences"],
    "properties": {
        "verdict": {
            "type": "boolean", 
            "description": "True if the combined visual and textual evidence confirms the caption accurately represents the image; False otherwise."
        },
        "explanation": {
            "type": "string",
            "description": "A detailed explanation based on analyzing both the image and textual evidence in relation to the caption."
        },
        "confidence_score": {
            "type": "integer",
            "description": "A score from 0 (no confidence) to 10 (complete confidence) indicating how certain the verdict is."
        },
        "supporting_evidences": {
            "type": "array",
            "description": "List of specific evidence that supports the verdict",
            'items': {
                'type': 'string'
            }
        }
    }
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["OOC", "confidence_score", "validation_summary", "explanation"],
    "properties": {
        "OOC": {
            "type": "boolean",
            "description": "false if the caption correctly represents the image (Not Out of Context), true if it misrepresents the image (Out of Context)"
        },
        "confidence_score": {
            "type": "integer",
            "description": "0-10 (reflecting overall certainty in the verdict based on combined analysis)."
        },
        "validation_summary": {
            "type": "string",
            "description": "A brief (1-2 sentence) summary highlighting whether viewers would be misled about what they're seeing."
        },
        "explanation": {
            "type": "string",
            "description": "A detailed, evidence-based justification (max 500 words) that examines what's actually shown in the image versus what the caption claims or implies is shown."
        }
    }
}

class GPTConnector:
    def __init__(self, api_key: str, model_name: str = "gpt-4-0613"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def call_with_structured_output(
            self,
            prompt: str,
            schema: Dict[str, Any],
            image_base64: Optional[str] = None,
            system_prompt: Optional[str] = SYSTEM_PROMPT
        ) -> Dict[str, Any]:
        """
        Call GPT with function-calling capabilities and directly use JSON schema
        """
        # Prepare the messages for GPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        if image_base64:
            messages.append({"role": "user", "content": f"Image: {image_base64}"})

        # Call OpenAI GPT model
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            functions=[
                {
                    "name": "generate_response",
                    "parameters": schema
                }
            ],
            function_call={"name": "generate_response"}
        )

        # Extract the function call arguments
        function_response = response.choices[0].message.function_call.arguments
        return json.loads(function_response)