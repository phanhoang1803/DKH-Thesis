# templates_2.py

### Internal Checking Prompt remains the same
INTERNAL_CHECKING_PROMPT = """TASK: Judge whether the given entities of the caption are used correctly in the given image.

INPUT:
News Caption: {caption}
Entities: {textual_entities}

INSTRUCTIONS:
1. Analyze visual content for each entity mentioned
2. Verify entity presence and accurate representation
3. Check spatial relationships and interactions
4. Assess temporal context if relevant
5. Cross-reference caption description vs visual evidence
6. Note any visual ambiguities or occlusions
7. Flag entities not clearly visible or only partially shown

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

# Separate External Checking Templates
EXTERNAL_CHECKING_WITH_EVIDENCE_PROMPT = """TASK:  Judge whether the given news caption is supported by the retrieved evidences.

INPUT:
News Caption: {caption}
Retrieved External Evidence:
{candidates}

CONTEXT: News sites typically provide verified reports, official statements, and direct reporting. Fact-checking sites investigate viral claims, providing evidence and methodology to verify/debunk claims with accuracy ratings.

INSTRUCTIONS:
1. Compare caption directly against evidence contents
2. Look for exact quotes or paraphrased content that supports/contradicts
3. Focus on factual elements, not interpretations
4. Consider timeframe relevance
5. Note missing key details
6. Check for bias or inconsistencies
7. Differentiate between claims that are merely cited versus those actively supported or debunked by the evidence
8. Analyze the full context of quoted claims to determine if they're presented as verified facts or examples of misinformation
"""

EXTERNAL_CHECKING_WITHOUT_EVIDENCE_PROMPT = """TASK: Judge whether the given news caption is correctly used by your knowledge.

INPUT:
News Caption: {caption}

INSTRUCTIONS:
1. Assess factual plausibility using world knowledge
2. Check for logical consistency and temporal accuracy
3. Identify verifiable vs speculative elements
4. Note common misconceptions or red flags
5. Consider multiple interpretations
6. Flag claims requiring external verification
"""

EXTERNAL_CHECKING_WITH_EVIDENCE_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 0-10,
- "explanation": The explanation of your decision,
- "supporting_points": List of relevant evidence that supports your decision

Where:
- verdict: "True" if the caption appears accurate and properly used in context, "False" otherwise
- confidence_score: 8-10 (strong direct evidence), 4-7 (partial/indirect evidence), 0-3 (unrelated/impossible to assess)
- explanation: A detailed explanation including specific points from evidence supporting your verdict. Avoid speculative statements.
- supporting_points: List relevant quotes/references from the provided evidence
"""

EXTERNAL_CHECKING_WITHOUT_EVIDENCE_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 0-10,
- "explanation": The explanation of your decision,
- "supporting_points": List of relevant knowledge points used in assessment

Where:
- verdict: "True" if the caption appears plausible based on general knowledge, "False" otherwise
- confidence_score: 8-10 (aligns with established facts), 4-7 (plausible but uncertain), 0-3 (unrelated/impossible to assess)
- explanation: A detailed explanation of your reasoning and any potential concerns. Avoid speculative statements.
- supporting_points: List relevant knowledge points used in your assessment
"""

FINAL_CHECKING_PROMPT = """TASK: Verify if caption accurately represents the context by analyzing:
1. Entity alignment 
2. Internal validation
3. External validation

INPUT:
Original Caption: {original_news_caption}
Internal Check Result: {internal_result}
External Check Result: {external_result}

VALIDATION CRITERIA:
1. Evaluate both internal consistency (entity alignment) and external verification.
2. Document any discrepancies found in either check.
3. Weigh confidence scores from both internal and external checks.
4. Acknowledge uncertainty when evidence is insufficient.
5. Analyze conflicts between internal and external checks.
6. Pay high attention to cases where both checks show high confidence (8-10).
7. Cross-validate with entity analysis when internal and external disagree.

NOTE: Base assessment only on available evidence. Maintain conservative judgment when uncertain.
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": True/False
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation results
- "explanation": A detailed explanation for the final result

Where:
- OOC: "True" if the caption is out of context based on both internal and external checks, "False" otherwise
- confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
- validation_summary: A concise summary of validation results, highlighting key points of agreement or disagreement.
- explanation: A comprehensive explanation (about 1000 words) for your final decisions based on all available evidence. Address any discrepancies between internal and external checks. Avoid speculative statements.
"""

def get_internal_prompt(caption: str, textual_entities: str) -> str:
    internal_prompt = INTERNAL_CHECKING_PROMPT.format(
        caption=caption,
        textual_entities=textual_entities
    )
    internal_prompt += INTERNAL_CHECKING_OUTPUT
    return internal_prompt

def get_external_prompt(caption: str, candidates: list) -> str:
    if not candidates:
        # Use knowledge-based prompt when no evidence is available
        external_prompt = EXTERNAL_CHECKING_WITHOUT_EVIDENCE_PROMPT.format(
            caption=caption
        )
        external_prompt += EXTERNAL_CHECKING_WITHOUT_EVIDENCE_OUTPUT
    else:
        # Format Article dataclass objects into a readable string
        candidates_str = "\n".join([
            f"Source {i+1}:\nTitle: {candidate.title}\n"
            f"Description: {candidate.description}\n"
            f"Content: {candidate.content}\n"
            f"Source: {candidate.source_domain}\n"
            f"URL: {candidate.url}\n"
            for i, candidate in enumerate(candidates)
        ])
        
        external_prompt = EXTERNAL_CHECKING_WITH_EVIDENCE_PROMPT.format(
            caption=caption,
            candidates=candidates_str
        )
        external_prompt += EXTERNAL_CHECKING_WITH_EVIDENCE_OUTPUT
    
    return external_prompt

def get_final_prompt(
    caption: str,
    internal_result: dict, 
    external_result: dict
) -> str:
    # Determine evidence type and format evidence string
    final_prompt = FINAL_CHECKING_PROMPT.format(
        original_news_caption=caption,
        internal_result=internal_result,
        external_result=external_result,
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt