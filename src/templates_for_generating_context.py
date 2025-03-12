SYSTEM_PROMPT_FOR_VLM_GENERATED_CONTEXT = """
You are an assistant specialized in analyzing news images. Your task is to extract detailed information from news images including the context, key elements, people, events, and any text visible in the image. Provide comprehensive and accurate descriptions that capture both the visual elements and the news context.
"""

CONTEXT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["information", "caption", "context"],
    "properties": {
        "information": {
            "type": "string",
            "description": "Information about the image"
        },
        "caption": {
            "type": "string",
            "description": "The caption of the image"
        },
        "context": {
            "type": "string",
            "description": "Context of the image"
        }
    }
}

VLM_GENERATED_PROMPT = """TASK: Analyze the given news image and provide detailed information.

INPUT:
- News Image: The provided image you are viewing is related to a news event.
- Entities Detected: {entities}
- News Caption: {caption}
- News Content: {news_content}

INSTRUCTIONS:
1. Carefully examine the image, identifying key visual elements.
2. Review the detected entities and assess how they relate to the image.
3. Analyze the provided news caption and news content, and evaluate whether they accurately correspond to the image. If discrepancies exist, the image context should be based on image information instead of news caption and news content.
4. Extract key information from the image, determine an appropriate caption, and summarize its content based on both the visual elements and relevant accompanying text if they are relevant to the image.

NOTE: Ensure that your analysis is primarily image-driven. Use the news caption and content to enhance context only if they align with the image. If inconsistencies arise, highlight them and rely on the image for accurate interpretation
"""

VLM_OUTPUT = """\nOUTPUT REQUIRED:
- "information": Information about the image
- "caption": Caption of the image
- "context": Context of the image

Where:
- information: Information about the image
- caption: Caption of the image
- context: Context of the image (maximum 500 words)
"""

CAPTION_CONTEXT_CHECKING_PROMPT = """TASK: Evaluate whether the image caption almost aligns with the image context.

INPUT:
- Caption: {caption}
- Context: {context} (Image information)

INSTRUCTIONS:
1. Verify the context is correct
2. Compare the caption against the provided image information
3. Determine if the caption almost aligns with the image context
4. Assess if the caption correctly conveys what the image is about
5. Check for any misrepresentations or omissions of key information
6. Provide a clear verdict (TRUE/FALSE) on caption-context alignment
7. Provide a list of supporting evidences for your verdict

NOTE: Provide a detailed explanation of your reasoning for the decision.
"""

SYSTEM_PROMPT_FOR_CAPTION_CONTEXT_CHECKING = """
You are an assistant specialized in analyzing news images. Your task is to evaluate if the image caption almost aligns with the image context.
"""

CAPTION_CONTEXT_CHECKING_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "alignment_score", "confidence_score",  "explanation", "supporting_evidences"],
    "properties": {
        "verdict": {
            "type": "boolean",
            "description": "True if the caption almost aligns with the image context, False otherwise"
        },
        "alignment_score": {
            "type": "integer",
            "description": "Score between 0 and 100"
        },
        "confidence_score": {
            "type": "integer",
            "description": "Score between 0 and 10"
        },
        "explanation": {
            "type": "string",
            "description": "Detailed explanation of your reasoning for the decision"
        },
        "supporting_evidences": {
            "type": "array",
            "description": "List of supporting evidences",
            "items": {
                "type": "string",
                "description": "Evidence"
            }
        }
    }
}

CAPTION_CONTEXT_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True if the caption aligns with the image context, False otherwise
- "alignment_score": Score between 0 and 10
- "confidence_score": Score between 0 and 10
- "discrepancies": Key misalignments identified
- "explanation": Detailed explanation of your reasoning for the decision
"""

def get_context_prompt(caption: str = None, entities: str = None, news_content: str = None) -> str:
    """
    Combines the VLM prompt with the expected output format to create
    a complete context prompt for the vision model.
    
    Returns:
        str: The complete prompt to be sent to the VLM
    """
    prompt = VLM_GENERATED_PROMPT.format(caption=caption, entities=entities, news_content=news_content) + VLM_OUTPUT
    return prompt

def get_caption_context_checking_prompt(caption: str, context: str) -> str:
    prompt = CAPTION_CONTEXT_CHECKING_PROMPT.format(
        caption=caption,
        context=context
    )
    prompt += CAPTION_CONTEXT_CHECKING_OUTPUT
    return prompt






