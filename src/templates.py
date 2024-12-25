# templates.py
### Internal Checking Prompt
INTERNAL_CHECKING_PROMPT = """TASK: Judge whether the given entities are used correctly in the given text.

INPUT:
News Caption: {caption}
Retrieved Entities from Content: {textual_entities}

INSTRUCTIONS:
1. Compare each entity with its usage in the caption.
2. Check for name/title accuracy.
3. Verify contextual correctness.
4. Ensure temporal consistency.
5. Only make a judgment based on the entities and their alignment with the caption. Do not speculate or assume information not supported by the provided content.

NOTE: If unsure about the accuracy of any entity's usage, please indicate that you are uncertain rather than providing a definitive answer.

"""

INTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 0-10,
- "explanation": "Short, clear explanation of the decision",

Where:
- verdict: "True" if the caption aligns with the external evidence, "False" otherwise.
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- explanation: A short, comprehensive explanation that integrates the caption, external evidence, and the alignment between them. Avoid speculation or unsupported conclusions.
"""

### External Checking Prompt
EXTERNAL_CHECKING_PROMPT = """TASK: Verify the accuracy of the caption against external evidence gathered from the internet.

INPUT:
Caption: {caption}
Evidence: {candidates}

VALIDATION CRITERIA:
1. Key facts should match exactly.
2. No contradictions between the caption and the external evidence.
3. Ensure temporal consistency (e.g., dates, events).
4. Verify contextual alignment (i.e., the information in the caption should match the context of the evidence).

NOTE: If there is any ambiguity or missing information in the external evidence, acknowledge it rather than making unsupported claims.
"""

EXTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 0-10,
- "explanation": "Short, detailed analysis of the alignment between the caption and the external evidence, with references to specific pieces of evidence."

Where:
- verdict: "True" if the caption aligns with the external evidence, "False" otherwise.
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- explanation: A short, detailed analysis of the alignment between the caption and external evidence, including any gaps or ambiguities.
"""

FINAL_CHECKING_PROMPT = """TASK: Synthesize the results from internal and external validation to assess the final accuracy of the caption.

INPUT:
Original News Caption: {original_news_caption}
Textual Entities (extracted from news content): {textual_entities}
External Evidence (Searched from the internet): {candidates}
Internal Check: {internal_result}
External Check: {external_result}

VALIDATION CRITERIA:
1. Confirm the final verdict based on internal and external consistency.
2. If any discrepancies or uncertainties are found in either check, state them clearly. Avoid making unsupported claims or assumptions.
3. If there is a lack of clarity or evidence, indicate uncertainty in the final result.
4. If External Evidence is a None list, the external check factor will not be included in the final answer.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": True/False
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation results from both the internal and external checks, including any unresolved discrepancies or uncertainties.
- "explanation": A short, clear, concise rationale for the final result, summarizing the synthesis of internal and external checks, and how they align or diverge.

Where:
- OOC: "True" if the caption is out of context based on both internal and external checks, "False" otherwise.
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- validation_summary: A concise summary of the internal and external validation results, highlighting key points of agreement or disagreement.
- explanation: A short but clear explanation that combines both internal and external validation results, highlighting any gaps, uncertainties, or contradictions. Avoid speculative statements.
"""

def get_internal_prompt(caption: str, textual_entities: str) -> str:
    internal_prompt = INTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        textual_entities=textual_entities
    )
    internal_prompt += INTERNAL_CHECKING_OUTPUT
    return internal_prompt

def get_external_prompt(caption: str, candidates: str) -> str:
    external_prompt = EXTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        candidates=candidates
    )
    external_prompt += EXTERNAL_CHECKING_OUTPUT
    return external_prompt

def get_final_prompt(
    caption: str,
    textual_entities: str,
    candidates: str,
    internal_result: str, 
    external_result: str
) -> str:
    final_prompt = FINAL_CHECKING_PROMPT.format(
        original_news_caption=caption,
        textual_entities=textual_entities,
        candidates=candidates,
        internal_result=internal_result,
        external_result=external_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt
