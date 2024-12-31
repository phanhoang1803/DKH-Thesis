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
- "explanation": The explanation of your decision,

Where:
- verdict: "True" if the caption aligns with the entities, "False" otherwise.
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- explanation: An explanation to your decision. Avoid speculative statements.
"""


FINAL_CHECKING_PROMPT = """TASK: Synthesize the results from internal validation to assess the final accuracy of the caption.

INPUT:
Original News Caption: {original_news_caption}
Textual Entities (extracted from news content): {textual_entities}
Internal Check Result: {internal_result}

VALIDATION CRITERIA:
2. If any discrepancies or uncertainties are found in either check, state them clearly. Avoid making unsupported claims or assumptions.
3. If there is a lack of clarity or evidence, indicate uncertainty in the final result.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": True/False
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation results from both the internal checks, including any unresolved discrepancies or uncertainties.
- "explanation": A short explanation (about 1000 words) for the final result, advoiding speculative statements.

Where:
- OOC: "True" if the caption is out of context based on both internal checks, "False" otherwise.
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- validation_summary: A concise summary of the internal validation results, highlighting key points of agreement or disagreement.
- explanation: A short explanation (about 1000 words) for your final decisions based on the input. Avoid speculative statements.
"""

def get_internal_prompt(caption: str, textual_entities: str) -> str:
    internal_prompt = INTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        textual_entities=textual_entities
    )
    internal_prompt += INTERNAL_CHECKING_OUTPUT
    return internal_prompt


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
        internal_result=internal_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt
