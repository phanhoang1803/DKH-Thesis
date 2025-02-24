# templates.py

### Internal Checking Prompt
from typing import Optional


INTERNAL_CHECKING_PROMPT = """TASK: Assess whether the image provides reasonable visual support for the given caption and verify the accuracy of extracted visual entities.

INPUT:
- News Caption: {caption}
- News Content: {content}
- Image: [Image attached]
- Previously Detected Visual Entities: {visual_entities}

INSTRUCTIONS:
1. Visual Entity Verification: Cross-check the detected entities against the actual image content and note any discrepancies.
2. Determine Relevance: Assess if the image aligns with the overall theme of the caption.
3. Check for Misalignment: Identify any visual elements that could contradict key claims in the caption.
4. Assess Completeness: Identify any critical missing visual elements that might weaken the caption's intended meaning.
5. Ensure Contextual Validity: Verify that the image does not distort the timeline, location, or key details of the event described.
6. Evaluate Supporting Evidence: Determine whether the image reasonably reinforces the event's portrayal rather than serving as misleading proof.
7. Identify Discrepancies: List major elements in the caption that are missing in the image and vice versa.

NOTE: The image does not need to depict every detail from the caption but should not create false implications. Focus on major inconsistencies rather than minor omissions. Consider potential image manipulation or contextual misrepresentation in the analysis.
"""


INTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence_score": 0-10
- "explanation": A concise, evidence-based analysis (max 1000 words)

Where:
- verdict: "True" if the caption reasonably aligns with the visual entities, "False" otherwise.
- confidence_score: A confidence level (0-10) indicating certainty in the verdict.
- explanation: A detailed assessment highlighting matches, inconsistencies, and reasoning. Avoid assumptions and speculation.
"""


VISION_CHECKING_PROMPT = """TASK: Analyze the visual consistency between the news image and reference images retrieved from web results, which were searched using the caption. 
Differences in visual elements do not necessarily mean the image is misleading—consider whether it still represents the same event, location, or broader context.

INPUT:
- News Image: The main image being verified (First image in the list).
- Reference Images: Images retrieved from web search results based on the caption.

INSTRUCTIONS:
1. Compare key visual elements such as main subjects, objects, and overall composition to determine similarity with reference images.
2. Check for temporal consistency by analyzing lighting conditions, seasonal elements, and timestamps to detect potential time discrepancies.
3. Verify spatial relationships, including the scale, background environment, and positioning of objects, ensuring they align with the reference images.
4. Examine lighting and camera angles to identify inconsistencies in shadows, perspectives, or unnatural lighting conditions.
5. Detect possible manipulation by looking for artifacts, irregularities, or unnatural edits that suggest digital alterations.
6. Assess whether the images, despite visual differences, still refer to the same event, location, or broader context.
7. If the news image differs significantly from the reference images and does not appear to represent the same event or context, flag it as potentially misleading or out of context.

NOTE:
Only assess visible elements. Do not infer or assume additional context beyond what is present in the images. Clearly document any inconsistencies and indicate whether the image is visually different but still contextually relevant.
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
EXTERNAL_CHECKING_PROMPT = """TASK: Analyze the visual consistency between the news image and validated reference images retrieved from web results, which were searched using the caption.

INPUT:
[Images are provided directly to the model. The first image is the news image to be verified, and the remaining images correspond to the retrieved web results in order.]
- News Caption: {caption}
- Retrieved Web Results: {web_results}

INSTRUCTIONS:

A. Reference Source Validation
1. Assess the relevance of web results to the news caption by checking temporal alignment, location consistency, event/subject match, and source credibility.
2. Flag and exclude results that describe a different event, show a different location, come from a significantly different time period, or provide conflicting context.
3. Document metadata conflicts such as inconsistent dates, conflicting locations, different event descriptions, and varying subject identifications.
4. Output validated image indices by returning a list of indices of the validated reference images that will be used in Step B.
    
B. Visual Analysis (Only analyze images from validated sources from step A)
1. Compare key visual elements, including main subjects, objects, overall scene composition, and background elements.
2. Check temporal consistency by analyzing lighting conditions, weather/seasonal indicators, time-specific elements, and chronological markers.
3. Verify spatial relationships by examining object scale, environmental context, object positioning, and perspective alignment.
4. Examine technical aspects such as lighting and shadows, camera angles, image quality, and color consistency.
5. Detect possible manipulation by identifying digital artifacts, unnatural edits, inconsistent elements, and unusual distortions.
6. Assess contextual alignment by determining whether the image accurately represents the event, location details match, time period is consistent, and the narrative aligns with reference sources.
7. Conduct a final evaluation by determining the overall visual match, assessing context alignment, identifying potential misleading elements, and verifying authenticity indicators.

NOTE: Base analysis strictly on provided evidence. Do not rely on external knowledge or assumptions. Clearly document all inconsistencies. Flag both visual and contextual discrepancies. Consider contextual alignment even when visual elements differ.
"""

EXTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
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


FINAL_CHECKING_PROMPT = """TASK: Determine whether the caption accurately represents the image, considering potential misinformation, out-of-context usage, or falsified details.

CONTEXT: 
- Internal Check: Evaluates whether the caption aligns with detected visual entities in the image, ensuring consistency in numbers, spatial relationships, and context.
- External Check: Assesses the visual consistency of the image by comparing it with reference images retrieved using the caption, analyzing spatial, temporal, and contextual alignment.

INPUT:
- News Caption: {news_caption}
- News Content: {news_content}
- News Image: [Image Attached]
- Internal Check Result: {internal_result}
- External Check Result: {external_result}

INSTRUCTIONS:   
1. Review the internal and external validation results to identify inconsistencies.
2. Compare the caption with the image and detected entities to determine if it accurately reflects the visual content.
3. If internal and external checks provide conflicting results, prioritize the most reliable evidence while noting any uncertainties.
4. Base the final decision strictly on available data; do not infer or assume missing details.
5. If evidence is inconclusive or ambiguous, flag the case for further review rather than making unsupported judgments.
6. Clearly document any limitations, including uncertainties or missing information that may impact the decision.
7. If the caption is only partially accurate, specify which elements are correct and which are not.
8. If either validation check lacks sufficient data, indicate this explicitly instead of drawing a definitive conclusion.

NOTE: Make the final verdict strictly based on the available evidence. Avoid speculation or assumptions.
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

def get_external_prompt(caption: str, web_results: list) -> str:
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
    )
    external_prompt += EXTERNAL_CHECKING_OUTPUT
    
    return external_prompt

def get_final_prompt(
    caption: str,
    content: str,
    internal_result: dict, 
    external_result: dict
) -> str:
    # Determine evidence type and format evidence string
    final_prompt = FINAL_CHECKING_PROMPT.format(
        news_caption=caption,
        news_content=content,
        internal_result=internal_result,
        external_result=external_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt