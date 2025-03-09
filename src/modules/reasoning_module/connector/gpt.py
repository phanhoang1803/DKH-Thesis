from typing import Any, Dict, Optional
import openai
import json

SYSTEM_PROMPT = """You are a highly accurate and logical fact-checking assistant. Your primary role is to evaluate the accuracy and contextual correctness of news captions using the provided information, including textual entities, external evidence, and any other relevant data. Your assessments must be objective, comprehensive, and clearly explained.

### **Responsibilities**
1. Analyze the input news caption in detail, considering both its textual content and extracted entities.
2. Validate the caption against:
   - **Textual Entities**: Check alignment for name accuracy, temporal consistency, and contextual correctness.
   - **External Evidence**: Compare the caption with external sources to verify key facts, ensure consistency, and identify any contradictions.
3. Address ambiguities or missing information explicitly. If sufficient evidence is not available, acknowledge the uncertainty and avoid speculative conclusions.
4. Provide a comprehensive final evaluation, synthesizing all available information to determine the overall accuracy and contextual validity of the caption.

### **Validation Criteria**
- **Accuracy**: Confirm factual correctness by comparing the caption with the provided inputs and evidence.
- **Consistency**: Ensure there are no contradictions, discrepancies, or temporal inconsistencies.
- **Context**: Verify that the caption aligns contextually with the entities and evidence.
- **Clarity**: Clearly highlight any uncertainties, gaps, or limitations in the available information.
"""

INTERNAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "explanation", "confidence_score"],
    "properties": {
        "verdict": {
            "type": "boolean",
            "description": "Indicates whether the internal validation passed (True) or failed (False)"
        },
        "explanation": {
            "type": "string",
            "description": "Explanation of the internal validation decision"
        },
        "confidence_score": {
            "type": "integer",
            "description": "Confidence score for the internal validation"
        }
    }
}

EXTERNAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "explanation", "confidence_score", "supporting_points"],
    "properties": {
        "verdict": {
            "type": "boolean",
            "description": "Indicates whether the external validation passed (True) or failed (False)"
        },
        "explanation": {
            "type": "string",
            "description": "Explanation of the external validation decision"
        },
        "confidence_score": {
            "type": "integer",
            "description": "Confidence score for the external validation"
        },
        "supporting_points": {
            "type": "string",
            "description": "Supporting points from external validation"
        }
    }
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["OOC", "confidence_score", "validation_summary", "explanation"],
    "properties": {
        "OOC": {
            "type": "boolean",
            "description": "\"False\" if the caption provides a fair symbolic representation of the news content, \"True\" otherwise."
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
        ) -> Dict[str, Any]:
        """
        Call GPT with function-calling capabilities and directly use JSON schema
        """
        # Prepare the messages for GPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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