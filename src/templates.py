# templates.py

from typing import Optional


INTERNAL_CHECKING_PROMPT_WITH_EVIDENCE = """TASK: Determine if the visual elements in the image provide enough evidence that the caption accurately represents what the image shows.

INPUT:
- Caption: {caption}
- Visual Elements Found: {visual_entities}
- Visual Descriptions: {visual_candidates}

INSTRUCTIONS:
1. Evidence Matching: Check if the descriptions of visual elements support key details mentioned in the caption.
2. Authenticity Check: Look for signs in the descriptions that might suggest the image has been altered or misrepresented.
3. Source Assessment: Consider how reliable the description sources are and if they show the image in its proper context.
4. Time and Setting Alignment: Verify that the described visuals match when and where the caption suggests the image was taken.
5. People and Object Confirmation: Make sure the people and objects described match those mentioned in the caption.
6. Evidence-Based Decision: Use the visual descriptions to determine if they support what the caption claims.
7. Inconsistency Identification: Note any differences between what the caption states and what the visual descriptions show.
8. Relevance Check: Assess whether the described visuals show the main subject of the caption rather than unrelated elements.

NOTE: Consider both what is directly described and what might be missing from the descriptions. Different sources may describe the same image differently.
"""

INTERNAL_CHECKING_PROMPT_WITHOUT_EVIDENCE = """TASK: Assess if the detected visual elements in the image align with the caption.

INPUT:
Caption: {caption}
Visual Elements Found: {visual_entities}

INSTRUCTIONS:
1. Match Check: Verify if the visual elements correspond to key components of the caption.
2. Consistency Check: Assess if the visual elements align with the caption’s descriptions.
3. Missing Details: Identify any essential elements in the caption that are absent among the detected visuals.
4. Misleading Elements: Highlight any detected elements that could mislead or contradict the caption.
5. Conclusion: Decide if the visual elements support or do not support the caption’s accuracy.

NOTE: Base your evaluation solely on the detected visual elements and the caption without additional context.
"""

INTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence": 0-10
- "explanation": A clear, evidence-based analysis (500 words maximum)
- "supporting_evidences": list of evidence that supports the verdict

Where:
- verdict: "True" if the visual evidence confirms the caption accurately represents the image without manipulation; "False" otherwise.
- confidence: A score from 0 (no confidence) to 10 (complete confidence) indicating how certain the verdict is.
- explanation: A detailed explanation based on specific visual evidence and how it relates to the caption.
- supporting_evidences: List of specific evidence that supports the verdict
"""

# Modified to incorporate direct image analysis with internal check results
FINAL_CHECKING_PROMPT = """TASK: Verify whether the news caption provides a symbolic or representative summary of the broader news content, without misleading or misrepresenting the visual content of the accompanying image.

INPUT:
- News Image: The image you are viewing directly
- News Caption: {news_caption}
- News Content: {news_content} (for context only, **do not base the final decision solely on this content**)
- Visual Check Result (Result of checking the caption with the detected visual candidates): {visual_check_result}

INSTRUCTIONS:
1. Visual Analysis: Describe key visual elements present in the image (objects, people, locations, actions, text, etc.), and identify any specific evidence that confirms or refutes elements of the caption. Base your analysis primarily on the visual content of the image.
2. Caption Claim Extraction: Identify the key claims or implications made by the caption about the broader news content. Summarize these claims in a clear and concise manner.
3. Visual Check Review: Examine the visual check result, prioritizing explanations with high confidence scores (7-10). If the confidence score is below 5, conduct an independent visual analysis without relying on the visual check result.
4. Symbolic Consistency: Assess if the image serves as a symbolic or representative visual that aligns with the broader context of the news content. The image does not need to show the exact event but should not create a misleading impression.
5. Misleading Content Detection: Determine if the caption:
    - Overstates or distorts what the image represents.
    - Selectively emphasizes certain aspects while omitting critical elements.
    - Uses the image in a way that creates a false impression, even if the details are factually correct.
6. Contradiction Analysis: Highlight any direct inconsistencies between the visual content and the caption, especially where the caption’s implications conflict with the image evidence.
7. Evidence Integration: Cross-reference the visual analysis with the visual check result, noting where the findings align or diverge to strengthen your evaluation.
8. Final Judgment: Conclude whether the caption serves as a reasonable symbolic representation of the broader news content without misleading viewers. Clearly state if the caption is misleading or not misleading, along with a brief explanation.
NOTE: The news content serves only as **background context** to understand the broader news narrative. The primary basis for your evaluation should be the **visual analysis of the image and the internal check result**.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": True/False
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation findings
- "explanation": Detailed justification of why the image is or isn't out of context, noting any specific misrepresentations or misleading elements

Where:
- OOC (Out of Context): "False" if the caption provides a fair symbolic representation of the news content, "True" otherwise.
- confidence_score: 0-10 (reflecting overall certainty in the verdict based on combined analysis).
- validation_summary: A brief (1-2 sentence) summary highlighting whether viewers would be misled about what they're seeing.
- explanation: A detailed, evidence-based justification (max 500 words) that examines what's actually shown in the image versus what the caption claims or implies is shown.
"""


def get_internal_prompt(caption: str, content: str, visual_entities: str, visual_candidates: list) -> str:
    if visual_candidates == []:
        internal_prompt = INTERNAL_CHECKING_PROMPT_WITHOUT_EVIDENCE.format(
            caption=caption,
            # content=content,
            visual_entities=visual_entities
        )
    else:
        results_str = ""
        for i, result in enumerate(visual_candidates, 1):
            results_str += f"\nVisual Candidate {i}:\n"
            results_str += f"Title: {result.title}\n"
            results_str += f"Caption: {result.caption}\n"
            results_str += f"Content: {result.content}\n"
            results_str += f"Domain: {result.domain}\n"
            results_str += "-" * 50 + "\n"
        
        internal_prompt = INTERNAL_CHECKING_PROMPT_WITH_EVIDENCE.format(
            caption=caption,
            # content=content,
            visual_entities=visual_entities,
            visual_candidates=results_str
        )
    internal_prompt += INTERNAL_CHECKING_OUTPUT
    return internal_prompt

# Modified to use only internal check results
def get_final_prompt(
    caption: str,
    content: str,
    visual_check_result: dict
) -> str:
    final_prompt = FINAL_CHECKING_PROMPT.format(
        news_caption=caption,
        news_content=content,
        visual_check_result=visual_check_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt