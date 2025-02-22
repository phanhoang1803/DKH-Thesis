# templates.py

### Internal Checking Prompt
from typing import Optional


INTERNAL_CHECKING_PROMPT = """TASK: Determine if the detected visual entities in the image are accurately represented in the given caption.

INPUT:
- News Caption: {caption}
- News Content: {content}
- Detected Visual Entities: {visual_entities}

INSTRUCTIONS:
1. Compare the caption and content against the detected visual entities.
2. Ensure spatial and temporal consistency across all elements.
3. Verify numerical claims in the text against visual evidence.
4. Assess contextual alignment between the text and the image.
5. Identify any text entities missing from the image.
6. Identify any visual entities missing from the text.
7. Evaluate relationships between entities and verify their accuracy.
8. Check if stated actions or events match the visual evidence.
9. Ensure environmental details (location, setting, time) are consistent.

NOTE: Only use provided entities for assessment. Do not assume details beyond available data.
"""


INTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence_score": 0-10
- "explanation": A concise, evidence-based analysis (max 1000 words)

Where:
- verdict: "True" if the caption correctly represents the visual entities, "False" otherwise.
- confidence_score: A confidence level (0-10) indicating certainty in the verdict.
- explanation: A detailed assessment highlighting matches, inconsistencies, and reasoning. Avoid assumptions and speculation.
"""


VISION_CHECKING_PROMPT = """TASK: Analyze the visual consistency between the news image and reference images from web results.

INPUT:
- News Image: The main image being verified (First image in the list).
- Reference Images: Images retrieved from web results.

INSTRUCTIONS:
1. Compare key visual elements (subjects, objects, composition).
2. Check for temporal consistency (lighting, seasonal elements, timestamps).
3. Verify spatial relationships, scale, and environmental details.
4. Examine lighting conditions, shadows, and camera angles.
5. Identify artifacts or signs of manipulation.
6. Assess contextual alignment between images.

NOTE: Only assess visible elements. Do not infer or assume additional context.
"""

VISION_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "alignment_score": 0-100
- "confidence_score": 0-10
- "explanation": Summary of analysis (max 500 words)
- "key_observations": List of specific matches, differences, or concerns

Where:
- verdict: "True" if the news image significantly aligns with reference images, "False" otherwise.
- alignment_score: A score (0-100) indicating the degree of visual consistency.
- confidence_score: A confidence level (0-10) for the final assessment.
- explanation: A comprehensive comparison of key visual elements, including matches, differences, and manipulation signs.
- key_observations: A bullet-point list of notable findings, highlighting specific visual consistencies or discrepancies.
"""



# Separate External Checking Templates
EXTERNAL_CHECKING_PROMPT = """TASK: Determine whether the given news caption is supported by both web search results and visual evidence.

INPUT:
- News Caption: {caption}
- Retrieved Web Results: {web_results}
- Visual Analysis Results: {vision_result}

INSTRUCTIONS:
1. Compare the caption with both textual and visual evidence.
2. Validate the caption based on:
   - Textual Evidence: Does the retrieved web content support the caption?
   - Visual Evidence: Does the image analysis align with the caption?
3. Consider key factors:
   - Factual accuracy in both text and images.
   - Alignment with verified textual sources and visual references.
   - Consistency across all types of evidence.
4. Do not assume or infer beyond what is explicitly stated in the sources.
5. Ensure judgment is strictly evidence-based, avoiding speculation.

NOTE: Use only the provided evidence. Avoid reliance on prior knowledge.
"""

EXTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence_score": 0-10
- "explanation": A detailed justification (max 1000 words)
- "supporting_points": List of direct evidence supporting the decision

Where:
- verdict: "True" if the caption is factually correct and properly used in context, "False" otherwise.
- confidence_score: A confidence level (0-10) in the final assessment.
- explanation: A detailed reasoning process using evidence from the web and image analysis. Avoid assumptions.
- supporting_points: A list of specific quotes or references from the provided evidence supporting the verdict.
"""


FINAL_CHECKING_PROMPT = """TASK: Determine if the caption accurately represents the image by analyzing internal and external validation results.

INPUT:
- Original Caption: {original_news_caption}
- Internal Check Result: {internal_result}
- External Check Result: {external_result}

INSTRUCTIONS:
1. Review both internal and external check results.
2. Compare findings against the original caption and image.
3. Identify contradictions or inconsistencies between the two checks.
4. Ensure strict evidence-based assessment.
5. Be conservative in judgment when results are ambiguous.
6. Document any limitations or uncertainties.
7. Clearly state the final decision and its rationale.
8. Flag partial verifications explicitly.
9. Indicate when either check lacks sufficient evidence.
10. Consider both the presence and absence of supporting evidence.

NOTE: Base the final verdict strictly on available data. Avoid assumptions.
"""


FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": True/False
- "confidence_score": 0-10
- "validation_summary": A concise summary of validation findings
- "explanation": Justification of the final result (max 500 words)

Where:
- OOC: "True" if the caption is out of context based on both internal and external checks, "False" otherwise.
- confidence_score: A confidence level (0-10) for the final decision.
- validation_summary: A brief summary highlighting key points of agreement or contradiction.
- explanation: A detailed reasoning process incorporating both internal and external validation. Clearly explain any discrepancies.
"""


def get_internal_prompt(caption: str, content: str, visual_entities: str) -> str:
    internal_prompt = INTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        content=content,
        visual_entities=visual_entities
    )
    internal_prompt += INTERNAL_CHECKING_OUTPUT
    return internal_prompt

def get_vision_prompt() -> str:
    """Generate vision analysis prompt with caption context."""
    vision_prompt = VISION_CHECKING_PROMPT + VISION_CHECKING_OUTPUT
    return vision_prompt

def get_external_prompt(caption: str, web_results: list, vision_result: Optional[dict] = None) -> str:
    # Format Article dataclass objects into a readable string
    results_str = ""
    for i, result in enumerate(web_results, 1):
        results_str += f"\nResult {i}:\n"
        results_str += f"Title: {result.title}\n"
        results_str += f"Caption: {result.caption}\n"
        results_str += f"Content: {result.content}\n"
        results_str += f"Domain: {result.domain}\n"
        results_str += "-" * 50 + "\n"
    
    external_prompt = EXTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        web_results=results_str,
        vision_result=vision_result
    )
    external_prompt += EXTERNAL_CHECKING_OUTPUT
    
    return external_prompt

def get_final_prompt(
    caption: str,
    internal_result: dict, 
    external_result: dict
) -> str:
    # Determine evidence type and format evidence string
    final_prompt = FINAL_CHECKING_PROMPT.format(
        original_news_caption=caption,
        internal_result=internal_result,
        external_result=external_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt