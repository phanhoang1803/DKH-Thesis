# from typing import Any, Dict, List, Optional
# import google.generativeai as genai
# import json

# class GeminiConnector:
#     def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
#         self.api_key = api_key
#         self.model_name = model_name
#         genai.configure(api_key=self.api_key)
#         self.model = genai.GenerativeModel(model_name=self.model_name)

#     def typeddict_to_json_schema(self, schema_class):
#         properties = {}
#         for field_name, field_type in schema_class.__annotations__.items():
#             if field_type == bool:
#                 field_schema = {"type": "boolean"}
#             elif field_type == str:
#                 field_schema = {"type": "string"}
#             elif field_type == int:
#                 field_schema = {"type": "integer"}
#             elif field_type == list:
#                 field_schema = {"type": "array", "items": {"type": "string"}}
#             else:
#                 raise ValueError(f"Unsupported type: {field_type}")
#             properties[field_name] = field_schema

#         return {
#             "type": "object",
#             "required": list(properties.keys()),
#             "properties": properties
#         }

#     def call_with_structured_output(
#         self,
#         prompt: str,
#         schema: Any,
#         images: Optional[List[str]] = None,
#         system_prompt: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Call Gemini with function calling capabilities, passing images as a list.
#         """
#         if system_prompt:
#             self.model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system_prompt)
        
#         # Convert schema class to JSON Schema if needed
#         if isinstance(schema, dict):
#             json_schema = schema
#         else:
#             json_schema = self.typeddict_to_json_schema(schema)
        
#         # Wrap base64 images in proper format
#         model_input = []
#         if images:
#             image_payload = [{'mime_type': 'image/jpeg', 'data': img} for img in images]
#             model_input.extend(image_payload)
        
#         model_input.append(prompt)

#         # Generate structured content
#         res = self.model.generate_content(
#             model_input,
#             generation_config=genai.GenerationConfig(
#                 temperature=0.7,
#                 top_p=0.9,
#                 top_k=40,
#                 response_mime_type="application/json", 
#                 response_schema=json_schema,
#                 candidate_count=1,
#                 max_output_tokens=2048,
#             )
#         )
#         print(res.candidates[0].content.parts[0].text)
#         return json.loads(res.candidates[0].content.parts[0].text)


from typing import Any, Dict, List, Optional
import google.generativeai as genai
import json
import re
from google.api_core import retry

class GeminiConnector:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", connector_name: str = None):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)   
        self.model = genai.GenerativeModel(model_name=self.model_name)
        self.connector_name = connector_name

    def typeddict_to_json_schema(self, schema_class):
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
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call Gemini with function calling capabilities, passing images as a list.
        Includes simple JSON parsing to handle various response formats.
        """
        # Set up the model with system prompt if provided
        
        if self.connector_name:
            print(f"Calling Gemini with connector name: {self.connector_name} and api key: {self.api_key}")
        
        if system_prompt:
            self.model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system_prompt)
        else:
            self.model = genai.GenerativeModel(model_name=self.model_name)
        
        # Convert schema class to JSON Schema if needed
        if isinstance(schema, dict):
            json_schema = schema
        else:
            json_schema = self.typeddict_to_json_schema(schema)
        
        # Wrap base64 images in proper format
        model_input = []
        if images:
            image_payload = [{'mime_type': 'image/jpeg', 'data': img} for img in images]
            model_input.extend(image_payload)
        
        model_input.append(prompt)

        # Generate structured content
        res = self.model.generate_content(
            model_input,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                response_mime_type="application/json", 
                response_schema=json_schema,
                candidate_count=1,
                max_output_tokens=2048,
            ),
            request_options={'retry': retry.Retry(initial=1, maximum=3, multiplier=1.5)}
        )
        
        # Get the response text
        response_text = res.candidates[0].content.parts[0].text
        
        # Try to parse the JSON directly
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to clean and extract valid JSON
            cleaned_text = self._clean_json_text(response_text)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Try to extract JSON objects using regex
                json_obj = self._extract_json_object(response_text)
                if json_obj:
                    return json_obj
                
                # If all parsing fails, return a default response
                return self._create_default_response(json_schema)
    
    def _clean_json_text(self, text):
        """Clean up the JSON text to make it parseable."""
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # If the text starts with ``` or ```json, remove the markdown code block markers
        if text.startswith('```'):
            lines = text.split('\n')
            if len(lines) > 1:
                # Remove the first line (```json)
                lines = lines[1:]
                # Find the closing code block marker and remove it
                if lines[-1].strip() == '```':
                    lines = lines[:-1]
                text = '\n'.join(lines)
        
        return text

    def _extract_json_object(self, text):
        """Extract a valid JSON object from the text using regex."""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)
        
        if matches:
            # Try each potential JSON object
            for json_str in matches:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return None

    def _create_default_response(self, schema):
        """Create a minimal valid response based on the schema."""
        default_response = {}
        
        # Add required properties with default values
        if 'properties' in schema:
            for key, prop_schema in schema['properties'].items():
                property_type = prop_schema.get('type')
                
                # Set appropriate default values based on type
                if property_type == 'string':
                    if 'enum' in prop_schema and prop_schema['enum']:
                        default_response[key] = prop_schema['enum'][0]
                    else:
                        default_response[key] = ""
                elif property_type in ['number', 'integer']:
                    default_response[key] = 0
                elif property_type == 'boolean':
                    default_response[key] = False
                elif property_type == 'array':
                    default_response[key] = []
                elif property_type == 'object':
                    default_response[key] = {}
        
        return default_response