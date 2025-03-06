# templates.py

from typing import Optional


INTERNAL_CHECKING_PROMPT_WITH_EVIDENCE = """TASK: Assess whether internet-sourced images provide reasonable visual support for the given news caption and verify the accuracy of claims based on visual evidence.

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

INTERNAL_CHECKING_PROMPT_WITHOUT_EVIDENCE = """TASK: Assess whether the news caption accurately represents the visual content based on the detected visual entities in the image.

INPUT:
- News Caption: {caption}
- News Content: {content}
- Detected Visual Entities: {visual_entities}

INSTRUCTIONS:
1. Entity Verification: Confirm whether visual entities detected in the image align with those mentioned in the news caption and content.
2. Consistency Analysis: Check if the relationships, quantities, and descriptions in the caption match what is detected in the image.
3. Context Evaluation: Determine if the context implied by the caption matches the visual scene represented by the detected entities.
4. Completeness Assessment: Identify if any critical elements mentioned in the caption are missing from the detected visual entities.
5. Contradiction Detection: Note any specific elements in the detected visual entities that directly contradict claims in the caption.
6. Confidence Evaluation: Consider the reliability of the entity detection and how it affects verification confidence.

NOTE: Focus solely on the relationship between the caption and the detected visual entities in the image. Base your assessment only on what can be objectively determined from this comparison.
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

# Modified to incorporate direct image analysis with internal check results
FINAL_CHECKING_PROMPT = """TASK: Verify whether the news caption provides a symbolic or representative summary of the broader news content, without misleading or misrepresenting the visual content of the accompanying image.

INPUT:
- News Image: The image you are viewing directly
- News Caption: {news_caption}
- News Content: {news_content} (for context only)
- Internal Check Result: {internal_result}

INSTRUCTIONS:
1. Visual Analysis: Describe key visual elements present in the image (objects, people, locations, actions, text, etc.).
2. Claim Verification: Identify the key claims or implications made by the caption in representing the overall news content, not just the literal image.
3. Internal Check Review: Examine the internal check result, paying close attention to the confidence score and explanation.
    - Prioritize internal check results with high confidence scores (8-10).
    - If the confidence score is below 5, conduct a more independent visual analysis.
4. Symbolic Consistency: Determine if the image, as a symbolic or representative visual, aligns with the broader context of the news content without introducing misleading interpretations.
5. Misleading Content Detection: Identify if the caption:
    - Overstates or distorts what the image represents in the context of the whole news content.
    - Selectively emphasizes certain aspects while omitting critical elements.
    - Uses the image in a way that creates a false impression, even if factually related.
6. Contradiction Analysis: Highlight any direct inconsistencies between the visual content and the caption, particularly if the caption creates a false impression of the overall news narrative.
7. Evidence Integration: Cross-reference visual analysis with the internal check result, especially when findings align or diverge.
8. Final Judgment: Conclude whether the caption serves as a reasonable symbolic representation of the news content without misleading viewers.

NOTE: The news content provides essential context for the caption's symbolic meaning. Focus on whether the caption conveys a truthful impression of the broader news narrative rather than a literal match with the image.
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


def get_internal_prompt(caption: str, content: str, visual_entities: str, image_evidences: list) -> str:
    if image_evidences == []:
        internal_prompt = INTERNAL_CHECKING_PROMPT_WITHOUT_EVIDENCE.format(
            caption=caption,
            content=content,
            visual_entities=visual_entities
        )
    else:
        results_str = ""
        for i, result in enumerate(image_evidences, 1):
            results_str += f"\Evidence {i}:\n"
            results_str += f"Title: {result.title}\n"
            results_str += f"Caption: {result.caption}\n"
            results_str += f"Content: {result.content}\n"
            results_str += f"Domain: {result.domain}\n"
            results_str += "-" * 50 + "\n"
        
        internal_prompt = INTERNAL_CHECKING_PROMPT_WITH_EVIDENCE.format(
            caption=caption,
            content=content,
            visual_entities=visual_entities,
            image_evidences=results_str
        )
    internal_prompt += INTERNAL_CHECKING_OUTPUT
    return internal_prompt

# Modified to use only internal check results
def get_final_prompt(
    caption: str,
    content: str,
    internal_result: dict
) -> str:
    final_prompt = FINAL_CHECKING_PROMPT.format(
        news_caption=caption,
        news_content=content,
        internal_result=internal_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt