from typing import Any, Dict, List, Optional
import google.generativeai as genai
import base64
import json

class GeminiVisionConnector:
    def __init__(self, api_key: str, model_name: str = "gemini-pro-vision"):
        """Initialize the Gemini Vision connector.
        
        Args:
            api_key (str): The API key for authentication
            model_name (str): The model name to use (defaults to gemini-pro-vision)
        """
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def typeddict_to_json_schema(self, schema_class):
        """Convert a TypedDict class to a JSON schema.
        
        Args:
            schema_class: The TypedDict class to convert
            
        Returns:
            Dict containing the JSON schema
        """
        properties = {}
        for field_name, field_type in schema_class.__annotations__.items():
            if field_type == bool:
                field_schema = {"type": "boolean"}
            elif field_type == str:
                field_schema = {"type": "string"}
            elif field_type == int:
                field_schema = {"type": "integer"}
            elif field_type == list:
                field_schema = {"type": "array", "items": {"type": "string"}}
            else:
                raise ValueError(f"Unsupported type: {field_type}")
            properties[field_name] = field_schema

        return {
            "type": "object",
            "required": list(properties.keys()),
            "properties": properties
        }

    def call_with_structured_output(
            self,
            prompt: str,
            schema: Any,
            image_base64: Optional[str] = None,
            ref_images_base64: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
        """Call Gemini Vision with structured output using the provided schema.
        
        Args:
            prompt (str): The prompt to send to the model
            schema (Any): TypedDict class defining the expected response structure
            image_base64 (Optional[str]): Base64 encoded main image
            ref_images_base64 (Optional[List[str]]): List of base64 encoded reference images
            
        Returns:
            Dict[str, Any]: Structured response following the provided schema
        """
        json_schema = self.typeddict_to_json_schema(schema)
        
        # Prepare the content parts list
        content_parts = []
        
        # Add main image if provided
        if image_base64:
            content_parts.append({
                'mime_type': 'image/jpeg',
                'data': image_base64
            })
            
        # Add reference images if provided
        if ref_images_base64:
            for ref_image in ref_images_base64:
                content_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': ref_image
                })
                
        # Add the prompt
        content_parts.append(prompt)
        
        # Generate response with structured output
        response = self.model.generate_content(
            content_parts,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
        
        # Parse and return the response
        return json.loads(response.candidates[0].content.parts[0].text)

    def generate_image_prompt(self, image_base64: str) -> str:
        """Generate a description prompt for an image.
        
        Args:
            image_base64 (str): Base64 encoded image
            
        Returns:
            str: Generated description of the image
        """
        response = self.model.generate_content(
            [
                {'mime_type': 'image/jpeg', 'data': image_base64},
                "Please provide a detailed description of this image."
            ]
        )
        return response.text