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

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["final_answer", "explanation", "additional_notes", "OOC", "confidence_level"],
    "properties": {
        "final_answer": {
            "type": "string",
            "description": "The overall synthesized result based on both internal and external validation checks, summarizing the caption's accuracy."
        },
        "explanation": {
            "type": "string",
            "description": "A short, clear explanation (up to 1000 words) of the final verdict, integrating internal and external validation results."
        },
        "additional_notes": {
            "type": "string",
            "description": "Any additional remarks or observations that are not part of the main explanation but are relevant to the decision."
        },
        "OOC": {
            "type": "boolean",
            "description": "Indicates whether the caption is out of context (True) or in context (False) based on both internal and external checks."
        },
        "confidence_level": {
            "type": "integer",
            "description": "A score (0-10) reflecting the confidence level in the final decision, with higher numbers indicating greater certainty."
        }
    }
}

INTERNAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["original_news_caption", "original_entities", "supported", "explanation"],
    "properties": {
        "original_news_caption": {
            "type": "string",
            "description": "The caption being analyzed for correctness based on its alignment with the entities."
        },
        "original_entities": {
            "type": "string",
            "description": "A list of entities extracted from the content that need to be validated against the caption."
        },
        "explanation": {
            "type": "string",
            "description": "A short explanation (about 1000 words) of the decision, detailing the alignment or discrepancies between the caption and entities."
        },
        "supported": {
            "type": "boolean",
            "description": "Indicates whether the entities in the caption are correctly used and supported by the content (True or False)."
        }
    }
}

EXTERNAL_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "explanation", "confidence_score", "supporting_points"],
    "properties": {
        "verdict": {
            "type": "boolean",
            "description": "Indicates whether the caption is accurate (True) or inaccurate (False) based on external evidence."
        },
        "explanation": {
            "type": "string",
            "description": "A short, concise explanation (about 1000 words) of the decision, based on the comparison between the caption and external evidence."
        },
        "confidence_score": {
            "type": "integer",
            "description": "A score (0-10) reflecting the confidence in the decision, with higher scores indicating greater certainty."
        },
        "supporting_points": {
            "type": "string",
            "description": "Key points from the external evidence that support or refute the caption's accuracy."
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