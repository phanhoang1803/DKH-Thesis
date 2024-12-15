# templates.py

INTERNAL_CHECKING_PROMPT = """INTERNAL CHECKING: Judge whether the given image is wrongly used in the given text.
News caption: {caption}
Possible textual entities: {textual_entities}

Answer: 
"""

EXTERNAL_CHECKING_PROMPT = """EXTERNAL CHECKING: 
Judge whether the given news caption is supported by the retrieved candidates.

News Caption: {caption}
Candidates (Caption/Image/Evidence): {candidates}

Answer:  
- State whether the given caption is supported by the evidence (Yes/No).  
- Provide reasoning for your judgment by comparing the caption with the candidates.
"""

FINAL_CHECKING_PROMPT = """FINAL CHECKING: Combine the results of internal and external checks to produce the final answer regarding the factuality of the news triplet (image, caption, evidence). You **must return the output in JSON format** with the following keys and corresponding values:

Image Caption:  
{image_caption}

Internal Check Result:  
{internal_result}

External Check Result:  
{external_result}
"""

def get_internal_prompt(caption: str, textual_entities: str) -> str:
    user_internal =  INTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        textual_entities=textual_entities
    )
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent assistant designed to analyze and assess the factuality of a news.",
        },
        {"role": "user", "content": user_internal},
    ]
    return messages

def get_external_prompt(caption: str, candidates: str) -> str:
    user_external = EXTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        candidates=candidates
    )
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent assistant designed to analyze and assess the factuality of a news.",
        },
        {"role": "user", "content": user_external},
    ]
    return messages

def get_final_prompt(internal_result: str, external_result: str, image_caption: str) -> str:
    final_prompt =  FINAL_CHECKING_PROMPT.format(
        internal_result=internal_result,
        external_result=external_result,
        image_caption=image_caption
    )
    final_prompt += """\nThe output **MUST** be formatted as a JSON object with the following structure:  
{
  "final_answer": "string",
  "OOC_NOOC": "string",
  "confidence_level": "string",
  "explanation": "string",
  "additional_notes": "string"
}

Where:  
1. **final_answer**: A concise conclusion about the alignment of the image, caption, and evidence. Example: "The image is wrongly used and does not align with the caption or evidence."  
2. **OOC_NOOC**:  
   - "OOC" (Out of Context) if there is any misinformation or misalignment between the image, caption, and evidence.  
   - "NOOC" (Not Out of Context) if the image, caption, and evidence align correctly.  
3. **confidence_level**: A confidence assessment, such as "High confidence," "Moderate confidence," or "Low confidence."  
4. **explanation**: Provide a clear and comprehensive explanation that integrates the internal check, external check, and the image caption. Highlight any detected inconsistencies or alignment between the triplet (image, caption, evidence) to justify the final decision.
5. **additional_notes**: Include any edge cases, ambiguities, or relevant observations that may affect the assessment.
"""

    messages = [
        {
            "role": "system",
            "content": "You are an intelligent assistant designed to analyze and assess the factuality of a news.",
        },
        {"role": "user", "content": final_prompt},
    ]
    return messages
