# templates.py

from typing import Optional


INTERNAL_CHECKING_PROMPT_WITH_EVIDENCE = """TASK: Determine if the visual elements of the image provide **direct evidence** that the caption accurately represents what the image shows and what the image is about.  

INPUT:
- Caption: {caption}
- Visual Entities Found: {visual_entities}
- Textual Descriptions: {textual_descriptions} (e.g., context, metadata, or information scraped from external sources based on the image)

INSTRUCTIONS:  

1. **Caption Matching:** Check if the caption being verified appears verbatim in any of the textual candidates, especially from reliable domains. Identical captions from reputable sources provide supporting evidence.
2. **Caption Consistency:** If the same or similar captions appear across multiple reliable sources, this strengthens verification. However, identical captions across suspicious domains might indicate coordinated misinformation.
3. **Evidence Matching:** Check if the textual descriptions **explicitly** confirm both the elements (e.g., people, event, location, date) AND the specific claims about these elements in the caption. Identifying matching elements alone is insufficient - the caption must accurately represent the actions, context, and relationships shown in the image.
4. **Authenticity Check:** Look for signs that might suggest the image has been altered or misrepresented.
5. **Source Assessment:** Evaluate the reliability of the sources describing the image. Give higher weight to established news organizations, official institutions, and verified accounts.
6. **Time and Setting Alignment:** Verify whether the descriptions explicitly confirm the date and location stated in the caption.
7. **People and Object Confirmation:** Ensure the people and objects in the image match those mentioned in the caption.
8. **Direct Evidence:** Key details **must be explicitly confirmed in the evidence, not inferred from contextual text.**
9. **Handling Missing Information:** If the textual descriptions **do not confirm** key details like date or location, mark them as “Not Fully Verified” rather than assuming correctness.
10. **Inconsistency Identification:** Note any differences or missing details between the caption and the textual descriptions. If the evidence only partially supports the caption, mark the result as "Partially Verified."

**NOTE:** 
- The final decision must be based on **verifiable visual evidence** rather than assumptions from surrounding text.
- Candidates' mere mention of objects visible in the image does not constitute verification of the caption's accuracy.
- Visual entities must independently support the caption's claims about event details, people, location, and context.
- If not explicitly confirmed through visual evidence, do not assume correctness.
"""

INTERNAL_CHECKING_PROMPT_WITHOUT_EVIDENCE = """TASK: Determine if the detected visual elements in the image provide **direct evidence** that the caption accurately represents what the image shows.  

INPUT:  
- Caption: {caption}  
- Visual Elements Found: {visual_entities}  

INSTRUCTIONS:  

1. **Evidence Matching:** Check if the detected visual elements **explicitly** confirm all key details in the caption (e.g., people, event, location, date).  
2. **Authenticity Check:** Look for any visual elements that might indicate an altered or misrepresented image.  
3. **Time and Setting Alignment:** Verify whether the detected elements confirm the **date and location** stated in the caption.  
4. **People and Object Confirmation:** Ensure that the identified people and objects match those described in the caption.  
5. **Handling Missing Information:** If **key details** like the event, date, or location **are not explicitly confirmed**, mark them as “Not Fully Verified” rather than assuming correctness.  
6. **Inconsistency Identification:** Identify any missing or contradictory elements between the caption and detected visuals. If the evidence only partially supports the caption, mark the result as **"Partially Verified."**  

**IMPORTANT:** The final decision must be based on **verifiable visual evidence** rather than assumptions. **If not explicitly confirmed, do not assume correctness.**  
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
FINAL_CHECKING_PROMPT = """TASK: Verify whether the news caption provides a symbolic or representative summary of the news content, without misleading or misrepresenting the visual content of the accompanying image.  

INPUT:
- News Image: The image you are viewing directly
- News Caption: {news_caption}
- News Content: {news_content} (for context only, **do not use as primary evidence**)
- Visual Check Result (Result of checking whether the caption accurately represents what the image shows using the detected candidates): {visual_check_result}

INSTRUCTIONS:

1. **Visual Analysis:** Describe key visual elements present in the image (objects, people, locations, actions, text, etc.), and identify any specific evidence that confirms or refutes elements of the caption. Base analysis primarily on the visual content of the image.
2. **Caption Claim Extraction:** Identify the key claims or implications made by the caption about the news content. Summarize these claims in a clear and concise manner.
3. **Visual Check Review:**:
    - Examine the visual check result carefully. 
    - **CRITICAL**: 
    - For confidence scores 8-10, final determination must base on this result. If the Visual Check Result explicitly indicates the caption does NOT accurately represent what's in the image, this MUST be classified as misleading/misrepresenting regardless of contextual alignment.
    - For confidence scores below 5, conduct a thorough independent visual analysis.
4. **Symbolic Consistency:** Check if the image represents a relevant part of the news content. The image doesn't need to show the entire story, but it should not mislead or distort key aspects of the content.
5. **Misleading Content Detection:** Determine if the caption:
   - Overstates or distorts what the image represents.
   - Selectively emphasizes certain aspects while omitting critical elements.
   - Uses the image in a way that creates a false impression, even if the details are factually correct.
6. **Contradiction Analysis:** Highlight any direct inconsistencies between the visual content and the caption, especially where the caption’s implications conflict with the image evidence.
7. **Evidence Integration:** Cross-reference independent visual analysis with the visual check result, giving priority to direct visual evidence when discrepancies exist.
8. **Final Judgment:** Based on all analysis above, determine whether the image is:
   - **NOOC (Not Out of Context): OOC = False**: The caption provides a fair symbolic representation of what's actually visible in the image
   - **OOC (Out of Context): OOC = True**: The caption misrepresents or does not match the visual content in the image, regardless of contextual accuracy

**NOTE:** The news content serves only as **background context** to understand the news narrative. The primary basis for evaluation must be the **visual analysis of the image and the visual check result**.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": False/True
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
            results_str += f"\nCandidate {i}:\n"
            results_str += f"**Title**: {result.title}\n"
            results_str += f"**Caption**: {result.caption}\n"
            results_str += f"Content: {result.content}\n"
            results_str += f"Domain: {result.domain}\n"
            results_str += "-" * 50 + "\n"
        
        internal_prompt = INTERNAL_CHECKING_PROMPT_WITH_EVIDENCE.format(
            caption=caption,
            # content=content,
            visual_entities=visual_entities,
            textual_descriptions=results_str
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