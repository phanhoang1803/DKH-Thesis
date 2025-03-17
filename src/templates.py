# templates.py

from typing import Optional


# VISUAL_CHECKING_PROMPT_WITH_EVIDENCE = """TASK: Determine if the visual elements of the image provide **direct evidence** that the caption accurately represents what the image shows and what the image is about.  

# INPUT:
# - News Caption: {caption}
# - Textual Descriptions: {textual_descriptions}  (context, metadata, or information scraped from external sources)

# INSTRUCTIONS:

# 1. **Caption Matching:** If any caption in any candidate (from the reliable domains) matches the news caption, decide the caption accurately represents the image and dont need to check further.
# 2. **Evidence Matching:** Check if the textual descriptions **explicitly** confirm both the elements (e.g., people, event, location, date) AND the specific claims about these elements in the caption. Identifying matching elements alone is insufficient - the caption must accurately represent the actions, context, and relationships shown in the image.
# 3. **Authenticity Check:** Identify any signs that the image has been altered or misrepresented.
# 4. **Source Assessment:**  Give higher weight to descriptions from established news organizations, official institutions, and verified accounts.
# 5. **Time and Setting Alignment:** Verify whether the descriptions explicitly confirm the date and location stated in the caption.
# 6. **People and Object Confirmation:** Ensure the people and objects in the caption match those in the image.
# 7. **Direct Evidence:** Key details **must be explicitly confirmed in the evidences, not inferred from contextual text.**
# 8. **Handling Missing Information:** If the textual descriptions **do not confirm** key details like date or location, mark them as “Not Fully Verified” rather than assuming correctness.
# 9. **Inconsistency Identification:** Note any differences or missing details between the caption and the textual descriptions. If the evidence only partially supports the caption, mark the result as "Partially Verified."

# **NOTE:** 
# - The final decision must be based on **verifiable evidences** rather than assumptions.
# - Candidates' mere mention of objects visible in the image does not constitute verification of the caption's accuracy.
# - If not explicitly confirmed through visual evidence, do not assume correctness.
# """

VISUAL_CHECKING_PROMPT_WITH_EVIDENCE = """TASK: Determine if the related candidates of the image provide evidence that prove the caption accurately represents what the image shows and what the image is about.

INPUT:
- **News Caption:** {caption}
- **Candidate Evidences (of the image):
{textual_descriptions}

INSTRUCTIONS:
1. **Caption Matching:** Check if the caption being verified appears verbatim in any of the textual candidates, especially from reliable domains. Identical captions from reputable sources provide supporting evidence.
2. **Evidence Matching:** Check if the candidates confirm both the elements (e.g., people, event, location, date) **AND** the specific claims in the caption. Identifying matching elements alone is insufficient—the caption must accurately reflect the actions, context, and relationships in the image.  
3. **Authenticity Check:** Identify any signs of image alteration or misrepresentation.  
4. **Source Assessment:** Give higher weight to descriptions from established news organizations, official institutions, and verified sources.  
5. **Time and Setting Alignment:** Verify whether candidates confirm the date and location in the caption.  
6. **People and Object Confirmation:** Ensure the people, objects, and activities in the caption align with the image.  
7. **Evidence Requirement:** Key details **must be identified** by the candidates, not inferred.  
8. **Handling Missing Information:** If candidates **do not state** crucial details like date or location, mark the caption as “Not Fully Verified” rather than assuming correctness.  
9. **Inconsistency Identification:** If the evidence only partially supports the caption, classify the result as "Partially Verified."  

**NOTE:**  
- The final decision must be based on **verifiable evidences**, not assumptions.  
- If explicit confirmation is missing, **do not assume correctness**.  
"""

VISUAL_CHECKING_PROMPT_PRED_CHECK_WITH_EVIDENCE = """TASK: Determine if the related candidates of the image provide evidence that prove the caption accurately represents what the news image shows and what the image is about (Not candidate image).

Candidate Schema:
- Caption: Caption of the candidate image
- Image Similarity Score: Similarity score between the image in the candidate and the image in the news
- Text Similarity Score: Similarity score between the caption in the candidate and the caption in the news
- Title: Title of the candidate evidence
- Content: Content of the candidate evidence
- Domain: Domain of the candidate evidence

INPUT:
- **News Caption:** {caption}
- **Candidate Evidences (of the image):**
{textual_descriptions}

### **INSTRUCTIONS**  
1. **Analyze Image Similarity Score:**  
   - **High (≥ 0.85):** Strong indication that the images match.  
   - **Low (< 0.85):** The images do **not** match.  

2. **Analyze Caption Matching (Text Similarity Score):**  
   - **High (≥ 0.7):** Captions are highly similar.  
   - **Low (< 0.7):** Captions do **not** match.  

3. **Strict Decision Rules:**  
   - **If Image Similarity Score ≥ 0.85 AND Text Similarity Score ≥ 0.7**, **IMMEDIATELY return "True"** (caption is in-context).  
   - **If Image Similarity Score ≥ 0.85 AND Text Similarity Score < 0.7**, **IMMEDIATELY return "False"** (caption is out-of-context).  
   - **If Image Similarity Score < 0.85 AND Text Similarity Score ≥ 0.7**, **IMMEDIATELY return "False"** (caption is out-of-context).  
   - **Otherwise, if Image Similarity Score < 0.85 AND Text Similarity Score < 0.7**, do analysis based on the captions and candidate to give the final decision.  

4. **Final Decision:**  
   - If the above rules already provided a decision, return it.  
   - If explicit confirmation is missing, **DO NOT assume correctness**.  

**NOTE:** If the image similarity score is below 0.85, the caption **must be marked as out-of-context, even if the text similarity score is high**.  
"""

VISUAL_CHECKING_PROMPT_PRED_CHECK_WITH_EVIDENCE_WITH_ANALYSIS = """TASK: Determine if the related candidates of the image provide evidence that prove the caption accurately represents what the news image shows and what the image is about (Not candidate image).

Candidate Schema:
- Caption: Caption of the candidate image
- Image Similarity Score: Similarity score between the image in the candidate and the image in the news
- Text Similarity Score: Similarity score between the caption in the candidate and the caption in the news
- Title: Title of the candidate evidence
- Content: Content of the candidate evidence
- Domain: Domain of the candidate evidence

INPUT:
- **News Caption:** {caption}
- **Candidate Evidences (of the image):**
{textual_descriptions}

### **INSTRUCTIONS**  
1. **Analyze Image Similarity Score:**  
   - **High (≥ 0.85):** Strong indication that the images match.  
   - **Low (< 0.85):** The images do **not** match.  

2. **Analyze Caption Matching (Text Similarity Score):**  
   - **High (≥ 0.7):** Captions are highly similar.  
   - **Low (< 0.7):** Captions do **not** match.  

3. **Strict Decision Rules:**  
   - **If Image Similarity Score ≥ 0.85 AND Text Similarity Score ≥ 0.7**, **IMMEDIATELY return "True"** (caption is in-context).  
   - **If Image Similarity Score ≥ 0.85 AND Text Similarity Score < 0.7**, **IMMEDIATELY return "False"** (caption is out-of-context).  
   - **If Image Similarity Score < 0.85 AND Text Similarity Score ≥ 0.7**, **IMMEDIATELY return "False"** (caption is out-of-context).  
   - **Otherwise, if Image Similarity Score < 0.85 AND Text Similarity Score < 0.7**, do analysis based on the captions and candidate to give the final decision.  

4. **Final Decision:**  
   - If the above rules already provided a decision, return it.  
   - If explicit confirmation is missing, **DO NOT assume correctness**.  

### **ANALYSIS (If not decided by the rules above)**
1. **Caption Matching:** Check if the caption being verified appears verbatim in any of the textual candidates, especially from reliable domains. Identical captions from reputable sources provide supporting evidence.
2. **Evidence Matching:** Check if the candidates confirm both the elements (e.g., people, event, location, date) **AND** the specific claims in the caption. Identifying matching elements alone is insufficient—the caption must accurately reflect the actions, context, and relationships in the image.  
3. **Authenticity Check:** Identify any signs of image alteration or misrepresentation.  
4. **Source Assessment:** Give higher weight to descriptions from established news organizations, official institutions, and verified sources.  
5. **Time and Setting Alignment:** Verify whether candidates confirm the date and location in the caption.  
6. **People and Object Confirmation:** Ensure the people, objects, and activities in the caption align with the image.  
7. **Evidence Requirement:** Key details **must be identified** by the candidates, not inferred.  
8. **Handling Missing Information:** If candidates **do not state** crucial details like date or location, mark the caption as “Not Fully Verified” rather than assuming correctness.  
9. **Inconsistency Identification:** If the evidence only partially supports the caption, classify the result as "Partially Verified."
"""

VISUAL_CHECKING_PROMPT_WITHOUT_EVIDENCE = """TASK: Check if the entities found in the image and the provided content together support what the caption claims.

INPUT:  
- **Caption:** {caption}  
- **Content:** {content}
- **Detected Entities:** {visual_entities}  

INSTRUCTIONS:  
1. **Main Elements Check:** Look at whether key things shown in the image (people, objects, places) match what you'd expect based on the caption.
2. **Content and Image Match:** Compare how the text content relates to both what's in the image and what the caption says; trust what you see in the image when there are conflicts.
3. **Context Consideration:** Think about wider situations that might explain differences, such as cultural settings, specific situations, or time-related factors.
4. **Confidence Level:** Rate how well things match - from completely matching (image and content both support caption) to clearly mismatched (image shows something different from the caption).
5. **Specific Details:** Point out exact elements that either support or contradict the caption instead of making broad judgments.
"""

VISUAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence": 0-10
- "explanation": A clear, evidence-based analysis (500 words maximum)
- "supporting_evidences": list of evidence that supports the verdict

Where:
- verdict: "True" if the evidences confirms the caption accurately represents the image without manipulation; "False" otherwise.
- confidence: A score from 0 (no confidence) to 10 (complete confidence) indicating how certain the verdict is.
- explanation: A detailed explanation based on specific evidences and how it relates to the caption.
- supporting_evidences: List of specific evidences that supports the verdict
"""


VISUAL_CHECKING_WITHOUT_EVIDENCE_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False
- "confidence": 0-10
- "explanation": A clear, evidence-based analysis (500 words maximum)
- "supporting_evidences": List of detected elements that support the verdict

Where:
- verdict: "True" if the detected elements clearly support the caption’s main idea; "False" if the image contradicts or does not provide enough confirmation.  
- confidence: A score from 0 (no confidence) to 10 (complete confidence), indicating how certain the verdict is based on visible evidence.  
- explanation: A detailed analysis based on the detected elements, explaining how they align with or contradict the caption, and noting any missing or unclear details.  
- supporting_evidences: List of specific evidences that justify the verdict, such as objects, people, or settings identified in the image.  
"""

FINAL_CHECKING_PROMPT = """TASK: Verify whether the news caption provides a correct representative summary of the image content.

INPUT:
- News Image: The image you are viewing directly
- News Caption: {news_caption}
- News Content: {news_content} (for context only, **do not use as primary evidence**)
- Image Check Result (Result of checking whether the caption accurately represents what the image shows using the image candidates): {image_check_result}

INSTRUCTIONS:

1. **Image Check Review:**
    - Examine the image check result carefully. 
    - **CRITICAL**: 
    - For confidence scores 7-10, final determination "MUST" be based on the image check result, move to instruction 7 and make the decision.
    - For confidence scores below 7, conduct a thorough instructions below.
2. **Image Analysis:** Describe key elements present in the image (objects, people, locations, actions, text, etc.), and identify any specific evidence that confirms or refutes elements of the caption. Base analysis primarily on the content of the image.
3. **Caption Claim Extraction:** Identify the key claims or implications made by the caption about the news content. Summarize these claims in a clear and concise manner.
4. **Misleading Content Detection:** Determine if the caption:
   - Selectively emphasizes certain aspects while omitting critical elements.
   - Uses the image in a way that creates a false impression, even if the details are factually correct.
5. **Contradiction Analysis:** Highlight any inconsistencies between the image content and the caption, especially where the caption’s implications conflict with the image evidence.
6. **Evidence Integration:** Cross-reference independent image analysis with the image check result, giving priority to evidence when discrepancies exist.
7. **Final Judgment:** Based on all analysis above, determine whether the image is:
   - **NOOC (Not Out of Context): OOC = False**: The caption provides a correct representative summary of the image content.
   - **OOC (Out of Context): OOC = True**: The caption does not match the image content.

**NOTE:** The primary basis for evaluation should be the **the image check result**.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": Boolean - false if the caption correctly represents the image (Not Out of Context), true if it misrepresents the image (Out of Context)
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation findings
- "explanation": Detailed justification of why the image is or isn't out of context

Where:
- OOC (Out of Context): Boolean value of "false" if the caption provides a correct representation of the image content, "true" otherwise.
- confidence_score: 0-10 (reflecting overall certainty in the verdict based on combined analysis).
- validation_summary: A brief (1-2 sentence) summary highlighting whether viewers would be misled about what they're seeing.
- explanation: A detailed, evidence-based justification (max 500 words) that examines what's actually shown in the image versus what the caption claims or implies is shown.
"""


def get_visual_prompt(caption: str, content: str, visual_entities: str, visual_candidates: list, pre_check=False) -> str:
    if visual_candidates == []:
        visual_prompt = VISUAL_CHECKING_PROMPT_WITHOUT_EVIDENCE.format(
            caption=caption,
            content=content,
            visual_entities=visual_entities
        )
        visual_prompt += VISUAL_CHECKING_WITHOUT_EVIDENCE_OUTPUT
    else:        
        # Do prediction check with similarity score
        # If the similarity score is high, then the caption is in-context of the image
        # Otherwise, the caption is out-of-context of the image
        have_false_case = False
        
        if pre_check:
            for i, result in enumerate(visual_candidates, 1):
                if (result.image_similarity_score < 0.85 and result.text_similarity_score >= 0.7) or (result.image_similarity_score >= 0.85 and result.text_similarity_score < 0.7):
                    have_false_case = True
                    break
        
            results_str = ""
            for i, result in enumerate(visual_candidates, 1):
                results_str += f"\n**Candidate** {i}:\n"
                results_str += f"**Caption**: {result.caption}\n"
                results_str += f"Image Similarity Score: {result.image_similarity_score}\n"
                results_str += f"Caption Similarity Score: {result.text_similarity_score}\n"
                results_str += f"Domain: {result.domain}\n"
                results_str += "-" * 50 + "\n"
            if have_false_case:
                visual_prompt = VISUAL_CHECKING_PROMPT_PRED_CHECK_WITH_EVIDENCE.format(
                    caption=caption,
                    textual_descriptions=results_str
                )
                visual_prompt += VISUAL_CHECKING_OUTPUT
            else:
                visual_prompt = VISUAL_CHECKING_PROMPT_PRED_CHECK_WITH_EVIDENCE_WITH_ANALYSIS.format(
                    caption=caption,
                    textual_descriptions=results_str
                )
                visual_prompt += VISUAL_CHECKING_OUTPUT
        else:
            results_str = ""
            for i, result in enumerate(visual_candidates, 1):
                results_str += f"\n**Candidate** {i}:\n"
                results_str += f"**Caption**: {result.caption}\n"
                results_str += f"Title: {result.title}\n"
                results_str += f"Content: {result.content}\n"
                results_str += f"Domain: {result.domain}\n"
                results_str += "-" * 50 + "\n"
            visual_prompt = VISUAL_CHECKING_PROMPT_WITH_EVIDENCE.format(
                caption=caption,
                # content=content,
                # visual_entities=visual_entities,
                textual_descriptions=results_str
        )   
        visual_prompt += VISUAL_CHECKING_OUTPUT
    return visual_prompt

def get_final_prompt(
    caption: str,
    content: str,
    visual_check_result: dict
) -> str:
    final_prompt = FINAL_CHECKING_PROMPT.format(
        news_caption=caption,
        news_content=content,
        image_check_result=visual_check_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt