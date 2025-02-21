# templates.py

### Internal Checking Prompt
from typing import Optional


INTERNAL_CHECKING_PROMPT = """TASK: Judge whether the visual entities detected in the image are accurately represented in the given caption.

INPUT:
News Caption: {caption}
News Content: {content}
Detected Visual Entities: {visual_entities}

INSTRUCTIONS:
1. Compare entities mentioned in caption and content against those detected in the image
2. Check for spatial and temporal consistency across all elements
3. Verify any numerical claims made in caption/content against visual evidence
4. Assess contextual alignment between text descriptions and visual scene
5. Identify entities mentioned in text but not visible in image
6. Note visual entities present but not mentioned in text
7. Evaluate described relationships between entities against visual evidence
8. Check consistency of any stated actions or events with visual elements
9. Verify environmental details (location, setting, time) match across all sources

NOTE: Base assessment only on provided entities. Do not make assumptions about unseen elements.
"""

INTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 0-10,
- "explanation": The explanation of your decision (up to 1000 words),

Where:
- verdict: "True" if the caption accurately represents the visual entities, "False" otherwise.
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- explanation: A detailed analysis of how well the caption represents the visual entities, noting both matches and discrepancies. Avoid speculative statements and focus on clear visual evidence.
"""


VISION_CHECKING_PROMPT = """TASK: Analyze the visual consistency between the news image and reference images from web results.

INPUT:
News Image: The main image being verified (First image)
Reference Images: Images retrieved from web results for comparison (The rest)

INSTRUCTIONS:
1. Compare visual elements between news and reference images (composition, subjects, objects)
2. Check for matching temporal indicators and time-specific elements
3. Verify spatial relationships, scale, and environmental details
4. Examine lighting conditions, shadows, and camera angles
5. Assess image quality and potential artifacts
6. Look for contextual alignment and signs of manipulation

NOTE: Base assessment only on visible elements in provided images. Do not make assumptions about unseen content.
"""

VISION_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "alignment_score": 0-100
- "confidence_score": 0-10
- "explanation": Detailed analysis findings (max 500 words)
- "key_observations": List of specific matches, differences, or concerns

Where:
- verdict: "True" if the news image significantly aligns with reference images, "False" otherwise
- alignment_score: A score from 0 to 100 indicating how well the images align visually and contextually
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the analysis
- explanation: A comprehensive explanation of the visual comparison, including matches, differences, and potential manipulations. Focus on concrete visual evidence
- key_observations: Specific points about matching elements, discrepancies, temporal indicators, and any signs of manipulation
"""




# Separate External Checking Templates
EXTERNAL_CHECKING_PROMPT = """TASK: Judge whether the given news caption is supported by both the retrieved web search results and visual evidence.

INPUT:
News Caption: {caption}

Retrieved Web Results:
{web_results}

Visual Analysis Results:
{vision_result}

INSTRUCTIONS:
1. Carefully analyze the given news caption and compare it with both textual and visual evidence.
2. Consider the following aspects:
   - Textual Evidence: Check if the information in the caption is accurately reflected in the retrieved results
   - Visual Evidence: Verify if the visual analysis supports the caption claims
3. Consider factors such as:
   - Factual correctness in both text and images
   - Alignment with textual sources and visual references
   - Contextual accuracy across all evidence types
4. Do not assume or infer information beyond what is explicitly stated in the retrieved web results and visual analysis.
5. Your judgment should be based strictly on the provided evidence (both textual and visual) and not on prior knowledge or external assumptions.
"""

EXTERNAL_CHECKING_OUTPUT = """OUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 0-10,
- "explanation": The explanation of your decision (up to 1000 words),
-"supporting_points": List of relevant evidence that supports your decision

Where:
- verdict: "True" if caption appears accurate and properly used in context, "False" otherwise
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer
- explanation: Detailed explanation including specific points from evidence supporting your verdict. Avoid speculation.
- supporting_points: List quotes/references from the provided evidence
"""

FINAL_CHECKING_PROMPT = """Determine whether the caption accurately represents the image content by analyzing internal and external validation results.

Internal Check: Verifies if visual elements detected in the image align with the caption's description
External Check: Validates the news image against reference images and sources to ensure contextual accuracy

INPUT:
Original Caption: {original_news_caption}
Internal Check Result: {internal_result}
External Check Result: {external_result}

INSTRUCTIONS:
1. Review both internal and external check results thoroughly
2. Compare the findings against the original caption and the image
3. Identify any contradictions between the two check results
4. Apply strict evidence-based assessment
5. Maintain conservative judgment when results are ambiguous
6. Document any limitations or uncertainties in the results
7. Provide clear reasoning for the final decision
8. Flag partial verifications explicitly
9. Note when either check lacks sufficient evidence
10. Consider both presence and absence of supporting evidence

NOTE: Base assessment only on available evidence. Maintain conservative judgment when uncertain.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": True/False
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation results
- "explanation": A short explanation for the final result (up to 500 words)

Where:
- OOC: "True" if the caption is out of context based on both internal and external checks, "False" otherwise
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- validation_summary: A concise summary of validation results, highlighting key points of agreement or disagreement.
- explanation: A comprehensive explanation (about 500 words) for your final decisions based on all available evidence. Address any discrepancies between internal and external checks. Avoid speculative statements.
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