# templates.py

from typing import Optional


VISUAL_CHECKING_PROMPT_WITH_EVIDENCE = """TASK: Determine if the visual elements of the image provide **direct evidence** that the caption accurately represents what the image shows and what the image is about.  

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

# VISUAL_CHECKING_PROMPT_WITHOUT_EVIDENCE = """TASK: Determine whether the news caption reasonably represents the image by analyzing how detected visual entities align with or contradict the caption’s claims.

# INPUT:
# - News Caption: {caption}
# - News Content: {content}
# - Detected Visual Entities: {visual_entities}

# INSTRUCTIONS:

# 1. **Identify Key Elements:** Extract the main subject, location, event, and key claims from the caption.  
# 2. **Compare Visual Entities:** Assess whether detected visual entities explicitly confirm, partially align with, or contradict these key elements.  
# 3. **Evaluate Partial Alignment:** If entities do not fully match but share thematic relevance (e.g., "sports venue" for a boxing match), consider it partial support instead of outright contradiction.  
# 4. **Detect Significant Gaps:** Identify any crucial missing elements that prevent verification (e.g., an image of a house confirming real estate but lacking evidence of pricing claims).  
# 5. **Contextual Consistency:** Check if the detected entities suggest a different context from the caption, but avoid assumptions beyond explicit visual evidence.  

# ### **CONFIDENCE SCORING GUIDELINES:**  
# - **High (8-10):** Strong visual evidence fully supports or contradicts the caption.  
# - **Medium (5-7):** Some alignment exists, but key uncertainties remain.  
# - **Low (1-4):** Limited or no visual evidence to determine accuracy.  

# **NOTE:** Base your assessment strictly on comparing detected entities with the caption. Avoid assumptions beyond what the entities explicitly reveal.
# """

VISUAL_CHECKING_PROMPT_WITHOUT_EVIDENCE = """TASK: Determine whether the news caption reasonably represents the image by analyzing how well the caption aligns with the content.

INPUT:
- News Caption: {caption}
- News Content: {content}

INSTRUCTIONS:
1. Contextual Relevance: Determines whether the context and sentence are thematically connected.
2. Topical Alignment: Evaluates if the context and sentence share a common theme or topic.
3. Integration Coherence: Assesses how well the context and sentence fit together in a broader context.
4. Caption Plausibility: Determines the likelihood that the sentence could reasonably serve as a caption for the context.
5. Entity Correlation: Measures how closely the entities or events mentioned in the context and sentence relate to each other.

CONFIDENCE SCORING GUIDELINES:
- High (8-10): Strong textual evidence fully supports or contradicts the caption.
- Medium (5-7): Some alignment exists, but key uncertainties remain.
- Low (1-4): Limited or no textual evidence to determine accuracy.

NOTE: Base your assessment strictly on comparing the caption with the content. Avoid assumptions beyond what the text explicitly reveals.
"""

VISUAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
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

TEXTUAL_CHECKING_PROMPT = """TASK: Determine if textual candidates support the caption that the caption accurately represents what the image shows and what the image is about.  

INPUT:  
- Image: The image you are viewing directly
- Caption: {caption}  
- Visual Entities Found: {visual_entities}  
- Textual Descriptions: {textual_descriptions} (e.g., context, metadata, or information scraped from external sources based on the caption)

INSTRUCTIONS:  

1. **Image Similarity:** Consider the similarity between the news image and the image in the candidate. If the similarity is less than 0.8 but the caption in the candidate is very similar to the news caption, it should be considered that the news caption is misleading/misrepresenting.
2. **Visual-Caption Consistency:** Analyze whether what you can see in the image aligns with what the caption describes. Focus on **direct visual evidence**.
3. **Caption-Text Alignment:** Check if the caption appears verbatim or similarly in any textual descriptions, especially from reliable sources.
4. **Evidence Cross-Verification:** Determine if the textual descriptions **explicitly confirm** both the visual elements AND the specific claims made in the caption.
5. **Visual Entity Verification:** Verify that entities detected in the image match those mentioned in both the caption and textual descriptions.
6. **Authenticity Assessment:** Look for visual signs that might suggest image manipulation or misrepresentation.
7. **Contextual Consistency:** Evaluate whether the context implied by the caption matches what is visible in the image and described in the texts.
8. **Source Reliability:** Consider the credibility of the textual evidence sources when weighing their confirmation value.
9. **Direct Evidence Priority:** Key details **must be explicitly confirmed by the image or text evidence, not inferred**.
10. **Missing Information Handling:** If key details cannot be confirmed by either the image or text, mark them as "Not Fully Verified".
11. **Inconsistency Documentation:** Identify any contradictions between image content, caption claims, and textual descriptions.

**NOTE:** 
- The final decision must prioritize **direct evidence** when available.
- If not explicitly confirmed through combined evidence, do not assume correctness.
"""

TEXTUAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence_score": 0-10
- "explanation": A clear, evidence-based analysis (500 words maximum)
- "supporting_evidences": list of evidence that supports the verdict

Where:
- verdict: "True" if the combined visual and textual evidence confirms the caption accurately represents the image without manipulation; "False" otherwise.
- confidence_score: A score from 0 (no confidence) to 10 (complete confidence) indicating how certain the verdict is.
- explanation: A detailed explanation based on analyzing both the image and textual evidence in relation to the caption.
- supporting_evidences: List of specific evidence that supports the verdict
"""
FINAL_CHECKING_PROMPT = """TASK: Verify whether the news caption provides a symbolic or representative summary of the news content, without misleading or misrepresenting the visual content of the accompanying image.  

INPUT:
- News Caption: {news_caption}
- News Content: {news_content} (for context only, **do not use as primary evidence**)
- Check Result (Result of checking whether the caption accurately represents what the image shows): {check_result}

INSTRUCTIONS:

1. **Check Result Review:**:
    - Examine the check result carefully. 
    - **CRITICAL**: 
    - For confidence scores 7-10, final determination must base on this result. If the Check Result explicitly indicates the caption does NOT accurately represent what's in the image, this MUST be classified as misleading/misrepresenting regardless of contextual alignment.
    - For confidence scores below 7, be more cautious in your determination.
2. **Caption Claim Extraction:** Identify the key claims or implications made by the caption about the news content. Summarize these claims in a clear and concise manner.
3. **Symbolic Consistency:** Based on the check result, determine if the image represents a relevant part of the news content. The image doesn't need to show the entire story, but it should not mislead or distort key aspects of the content.
4. **Misleading Content Detection:** Determine if the caption:
   - Overstates or distorts what the image represents.
   - Selectively emphasizes certain aspects while omitting critical elements.
   - Uses the image in a way that creates a false impression, even if the details are factually correct.
5. **Contradiction Analysis:** Highlight any direct inconsistencies noted in the check result, especially where the caption's implications conflict with the image evidence.
6. **Evidence Integration:** Evaluate the evidence provided in the check result, giving priority to direct evidence when making your final determination.
7. **Final Judgment:** Based on all analysis above, determine whether the image is:
   - **NOOC (Not Out of Context): OOC = False**: The caption provides a fair symbolic representation of what's actually visible in the image
   - **OOC (Out of Context): OOC = True**: The caption misrepresents or does not match the visual content in the image, regardless of contextual accuracy

**NOTE:** The news content serves only as **background context** to understand the news narrative. The primary basis for evaluation must be the **check result**.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": False/True
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation findings
- "explanation": Detailed justification of why the image is or isn't out of context, noting any specific misrepresentations or misleading elements

Where:
- OOC (Out of Context): "False" if the caption provides a fair symbolic representation of what's actually visible in the image, "True" otherwise.
- confidence_score: 0-10 (reflecting overall certainty in the verdict based on combined analysis).
- validation_summary: A brief (1-2 sentence) summary highlighting whether viewers would be misled about what they're seeing.
- explanation: A detailed, evidence-based justification (max 500 words) that examines what's actually shown in the image versus what the caption claims or implies is shown.
"""


def get_visual_prompt(caption: str, content: str, visual_entities: str, visual_candidates: list) -> str:
    if visual_candidates == []:
        visual_prompt = VISUAL_CHECKING_PROMPT_WITHOUT_EVIDENCE.format(
            caption=caption,
            content=content,
            # visual_entities=visual_entities
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
        
        visual_prompt = VISUAL_CHECKING_PROMPT_WITH_EVIDENCE.format(
            caption=caption,
            # content=content,
            visual_entities=visual_entities,
            textual_descriptions=results_str
        )
    visual_prompt += VISUAL_CHECKING_OUTPUT
    return visual_prompt

def get_textual_prompt(caption: str, content: str, visual_entities: str, textual_candidates: list) -> str:
    results_str = ""
    for i, result in enumerate(textual_candidates, 1):
        results_str += f"\nCandidate {i}:\n"
        results_str += f"**Title**: {result.title}\n"
        results_str += f"**Caption**: {result.caption}\n"
        results_str += f"Content: {result.content}\n"
        results_str += f"Domain: {result.domain}\n"
        results_str += f"Image Similarity (The similarity between the news image and the image in the candidate): {result.similarity_score}\n"
        results_str += "-" * 50 + "\n"
        
    textual_prompt = TEXTUAL_CHECKING_PROMPT.format(
        caption=caption,
        # content=content,
        visual_entities=visual_entities,
        textual_descriptions=results_str
    )
    textual_prompt += TEXTUAL_CHECKING_OUTPUT
    return textual_prompt


def get_final_prompt(
    caption: str,
    content: str,
    check_result: dict
) -> str:
    final_prompt = FINAL_CHECKING_PROMPT.format(
        news_caption=caption,
        news_content=content,
        check_result=check_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt