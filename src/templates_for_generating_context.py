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





IMAGE_DESCRIPTION_SCHEMA = {
    "type": "object",
    "required": ["objective_description", "generated_caption", "confidence_score"],
    "properties": {
        "objective_description": {
            "type": "string",
            "description": "A detailed, objective description of what is visible in the image"
        },
        "generated_caption": {
            "type": "string",
            "description": "A factual caption describing only what is directly visible in the image"
        },
        "confidence_score": {
            "type": "integer",
            "description": "How confident you are in your description (1-10)"
        }
    }
}

CAPTION_VERIFICATION_SCHEMA = {
    "type": "object",
    "required": ["verdict", "alignment_score", "explanation", "supporting_evidences"],
    "properties": {
        "verdict": {
            "type": "boolean",
            "description": "Whether the caption accurately represents the image (TRUE if accurate, FALSE if misleading)"
        },
        "alignment_score": {
            "type": "integer",
            "description": "How well the caption aligns with the image content (0-100)"
        },
        "explanation": {
            "type": "string",
            "description": "Detailed explanation of your verification analysis"
        },
        "supporting_evidences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Evidence from the image analysis supporting your conclusion"
        }
    }
}

IMAGE_ANALYSIS_PROMPT = """
TASK: Analyze this news image independently and provide detailed objective information.

INSTRUCTIONS:
1. Carefully examine this image without any preconceptions.
2. Identify all key visual elements including:
   - People (their appearance, actions, expressions)
   - Objects (prominent items, vehicles, buildings)
   - Setting (location, time of day, weather conditions)
   - Text (any visible signs, captions, headlines)
   - Actions (what activities are occurring)
3. Note any clear emotional tones or atmospheres conveyed by the image.
4. Consider these detected entities that might be present: {entities}
5. Generate a caption that objectively describes ONLY what is visible in the image.

IMPORTANT: 
* Focus ONLY on what you can directly see in the image.
* Do NOT make assumptions about context beyond what is visually evident.
* Be specific and detailed about visual elements.
* Prioritize accuracy over completeness.
* If text is visible in the image, report it exactly as written.
* Be cautious about making claims about people's identities unless clearly identifiable.
"""

CAPTION_VERIFICATION_PROMPT = """
TASK: Verify if a news caption accurately represents an image based on an objective image analysis.

INPUT:
- NEWS CAPTION: "{caption}"

OBJECTIVE IMAGE ANALYSIS:
- DETAILED DESCRIPTION: {objective_description}
- INDEPENDENTLY GENERATED CAPTION: {generated_caption}
- CONFIDENCE SCORE: {confidence_score}

INSTRUCTIONS:
1. Compare the news caption against the objective image analysis.
2. Evaluate how well the caption represents the image using these categories:
   - DIRECTLY CONTRADICTED: The caption makes claims that directly contradict what's visible in the image
   - PARTIALLY SUPPORTED: The main subject/action aligns, but has additional context not visible in the image
   - FULLY SUPPORTED: The caption accurately describes what's visible in the image

3. Assess which elements of the caption are:
   a) Directly visible in the image
   b) Implied but not directly visible
   c) Neither visible nor implied

4. Evaluate if the central claim or main focus of the caption is supported, even if peripheral details aren't visible.

5. Determine an alignment score based on:
   - How well the core message of the caption aligns with the image (60% of score)
   - How many additional details are accurate vs. unverifiable (40% of score)

6. Make your final verdict, with these guidelines:
   - TRUE if the core message aligns and no elements are directly contradicted, even if some details are unverifiable
   - FALSE if key elements are directly contradicted OR if the image shows something fundamentally different from what the caption describes

IMPORTANT: News captions often include contextual information not directly visible in the image. Focus on whether the caption CONTRADICTS the image, not whether every detail is visible.
"""

def generate_image_description(image_base64, visual_entities, entities_scores, vlm_connector):
    """
    Generate an independent description of the image using the VLM.
    Uses the detailed prompt but returns a simplified schema.
    """
    
    if entities_scores is not None:
        visual_entities = [f"{entity} ({score:.2f})" for entity, score in zip(visual_entities, entities_scores)]
    
    prompt = IMAGE_ANALYSIS_PROMPT.format(entities=visual_entities)
    
    response = vlm_connector.call_with_structured_output(
        prompt=prompt,
        schema=IMAGE_DESCRIPTION_SCHEMA,
        image_base64=image_base64,
        system_prompt="You are an expert image analyst who provides objective, detailed descriptions of news images without making assumptions beyond what is visually evident."
    )
    
    return response

def verify_caption_against_description(image_description, news_caption, llm_connector):
    """
    Verify if the news caption accurately represents the image based on the independent description.
    Uses the detailed prompt but returns a simplified schema.
    """
    prompt = CAPTION_VERIFICATION_PROMPT.format(
        caption=news_caption,
        objective_description=image_description["objective_description"],
        generated_caption=image_description["generated_caption"],
        confidence_score=image_description["confidence_score"]
    )
    
    response = llm_connector.call_with_structured_output(
        prompt=prompt,
        schema=CAPTION_VERIFICATION_SCHEMA,
        system_prompt="You are a media literacy expert specialized in detecting misrepresentations in news images. Your task is to objectively assess whether captions accurately represent the images they describe."
    )
    
    return response