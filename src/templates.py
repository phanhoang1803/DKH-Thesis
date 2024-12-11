# templates.py

INTERNAL_CHECKING_PROMPT = """INTERNAL CHECKING: Judge whether the given image is wrongly used in the given text.
News caption: {caption}
Possible textual entities: {textual_entities}

Answer: 
"""

EXTERNAL_CHECKING_PROMPT = """EXTERNAL CHECKING: Judge whether the given news caption is supported by the retrieved candidates
News caption: {caption}
Candidates: {candidates}

Answer: 
"""

FINAL_CHECKING_PROMPT = """FINAL CHECKING: Combine the internal and external checking results to give the final answer.
Internal: {internal_result}
External: {external_result}

Final Answer: [Provide a comprehensive conclusion about the news factuality]
OOC/NOOC: [OOC: Out of Context, NOOC: Not Out of Context]
Explanation: [Explain the reasoning behind the final decision]
"""

def get_internal_prompt(caption: str, textual_entities: str) -> str:
    return INTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        textual_entities=textual_entities
    )

def get_external_prompt(caption: str, candidates: str) -> str:
    return EXTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        candidates=candidates
    )

def get_final_prompt(internal_result: str, external_result: str) -> str:
    return FINAL_CHECKING_PROMPT.format(
        internal_result=internal_result,
        external_result=external_result
    )