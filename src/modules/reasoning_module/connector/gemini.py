from typing import Any, Dict, List, Optional
import google.generativeai as genai
import typing_extensions as typing
import json

class FinalResponse(typing.TypedDict):
    final_answer: str
    explanation: str
    additional_notes: str
    OOC: bool
    confidence_level: int

class InternalResponse(typing.TypedDict):
    original_news_caption: str
    original_entities: str
    supported: bool
    explanation: str


class GeminiConnector:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def call_with_structured_output(
        self,
        prompt: str,
        schema,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call Gemini with function calling capabilities
        """
        
        if image_base64:
            input = [{'mime_type':'image/jpeg', 'data': image_base64}, prompt]
        else:
            input = prompt

        res = self.model.generate_content(
            input,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=schema
            )
        )
        print(res)
        return json.loads(res.candidates[0].content.parts[0].text)
    
if __name__ == "__main__":
    import os

    llm_connector = GeminiConnector(
        api_key="AIzaSyChksaZVvkP4dZlN579I3RA-LzzhiJVY3g",
        model_name="gemini-1.5-flash"
    )

    prompt = """INTERNAL CHECKING: Judge whether the given entities is wrongly used in the given text.
News caption: Julian Castro at his announcement in San Antonio, Tex., on Saturday. Mr. Castro, the former secretary of housing and urban development, would be one of the youngest presidents if elected.
Possible textual entities: Mr.Castro, Julian Castro, San Antonio

Answer:
- State whether the given entities is supported by the news caption (Yes/No).  
- Provide reasoning for your judgment. """

    res = llm_connector.call_with_structured_output(
        prompt=prompt,
        schema=InExtResponse
    )
    print(res)