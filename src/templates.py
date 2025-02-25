# templates.py

### Internal Checking Prompt
from typing import Optional


INTERNAL_CHECKING_PROMPT = """TASK: Assess whether internet-sourced images provide reasonable visual support for the given news caption and verify the accuracy of claims based on visual evidence.

INPUT:
- News Caption: {caption}
- News Content: {content}
- Detected Visual Entities: {visual_entities}
- Image Evidences: {image_evidences}

INSTRUCTIONS:
1. Cross-Verification: Compare the news caption/content with the sourced image evidence to identify consistencies or contradictions.
2. Image Source Assessment: Evaluate the credibility of image sources (domains) and the context in which these images appear.
3. Timeline Consistency: Verify that the image evidence matches the temporal aspects mentioned in the news.
4. Entity Verification: Confirm whether visual entities detected in internet images align with those mentioned in the news text.
5. Context Analysis: Determine if the sourced images are being used in their original context or potentially repurposed.
6. Contradiction Detection: Identify specific elements in the image evidence that contradict claims made in the news text.
7. Completeness Evaluation: Assess whether critical elements of the news narrative are supported by the available image evidence.
8. Visual Manipulation Assessment: Note any signs that sourced images may have been altered or manipulated.

NOTE: Consider both what is present and what is absent in the visual evidence. Images from different sources may show different perspectives of the same event or entirely different events.
"""

INTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence_score": 0-10
- "explanation": A concise, evidence-based analysis (max 1000 words)

Where:
- verdict: "True" if the caption reasonably aligns with the visual entities and image evidences, "False" otherwise.
- confidence_score: A confidence level (0-10) indicating certainty in the verdict.
- explanation: A detailed assessment highlighting matches, inconsistencies, and reasoning between the news text and sourced images. Avoid assumptions and speculation.
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

FINAL_CHECKING_PROMPT = """TASK: Determine whether a news caption accurately represents its accompanying image, considering potential misinformation, out-of-context usage, or falsified details.

CONTEXT:
- Internal Check: Evaluates whether internet-sourced images provide reasonable visual support for the news caption by comparing caption/content with sourced image evidence.
- External Check: Assesses the visual consistency between the news image and reference images retrieved from web searches based on the caption.

INPUT:
- News Caption: {news_caption}
- News Content: {news_content}
- News Image: [Image Attached]
- Internal Check Result: {internal_result}
- External Check Result: {external_result}

INSTRUCTIONS:
1. Synthesize the internal and external validation results to form a comprehensive assessment.
2. Compare the caption with both the news image and any detected entities to determine accuracy.
3. When internal and external checks conflict, evaluate the strength of evidence from each source.
4. Base the final decision strictly on available data; avoid inferring or assuming missing details.
5. If evidence is inconclusive or ambiguous, flag the case for further review rather than making unsupported judgments.
6. Document any limitations, including uncertainties or missing information that may impact the decision.
7. For partially accurate captions, specify which elements are correct and which are misrepresented.
8. If either validation check lacks sufficient data, indicate this explicitly in your assessment.

NOTE: Make the final verdict strictly based on the available evidence. Avoid speculation or assumptions beyond what is directly observable in the provided information.
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


def get_internal_prompt(caption: str, content: str, visual_entities: str, image_evidences: str) -> str:
    internal_prompt = INTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        content=content,
        visual_entities=visual_entities,
        image_evidences=image_evidences
    )
    internal_prompt += INTERNAL_CHECKING_OUTPUT
    return internal_prompt

def get_vision_prompt() -> str:
    """Generate vision analysis prompt with caption context."""
    vision_prompt = VISION_CHECKING_PROMPT + VISION_CHECKING_OUTPUT
    return vision_prompt

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