# # templates.py
# ### Internal Checking Prompt
# INTERNAL_CHECKING_PROMPT = """TASK: Judge whether the given entities are used correctly in the given text.

# INPUT:
# News Caption: {caption}
# Retrieved Entities from Content: {textual_entities}

# INSTRUCTIONS:
# 1. Compare each entity with its usage in the caption.
# 2. Check for name/title accuracy.
# 3. Verify contextual correctness.
# 4. Ensure temporal consistency.
# 5. Only make a judgment based on the entities and their alignment with the caption. Do not speculate or assume information not supported by the provided content.

# NOTE: If unsure about the accuracy of any entity's usage, please indicate that you are uncertain rather than providing a definitive answer.

# """

# INTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
# - "verdict": True/False,
# - "confidence_score": 0-10,
# - "explanation": The explanation of your decision,

# Where:
# - verdict: "True" if the caption aligns with the entities, "False" otherwise.
# - confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
# - explanation: An explanation to your decision. Avoid speculative statements.
# """

# # External Checking Templates
# EXTERNAL_CHECKING_PROMPT = """TASK: Evaluate whether the news caption is accurate and being used in the correct context.

# INPUT:
# News Caption: {caption}
# Retrieved External Evidence:
# {candidates}

# INSTRUCTIONS:
# IF EXTERNAL EVIDENCE IS AVAILABLE:
# 1. Compare the caption directly with the external evidence:
#    - Check if the caption matches the context from the articles
#    - Verify names, events, dates, and other factual details
#    - Look for any signs of misuse or out-of-context presentation
#    - Consider source credibility and temporal relevance
#    - Identify any contradictions or inconsistencies
# 2. Make your judgment based strictly on the evidence provided

# IF NO EXTERNAL EVIDENCE IS AVAILABLE:
# 1. Use your knowledge to assess the caption's plausibility:
#    - Check if the caption makes logical sense
#    - Verify if the described scenario is consistent with known facts
#    - Look for any obvious anachronisms or inconsistencies
#    - Consider if the caption follows typical news reporting patterns
# 2. Make your judgment based on your knowledge and understanding

# In both cases:
# - Be specific about what aspects seem accurate or inaccurate
# - Point out any red flags or concerning elements
# - Consider both factual accuracy and contextual appropriateness

# NOTE: Clearly indicate whether your assessment is based on external evidence or knowledge-based analysis.
# """

# EXTERNAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
# - "verdict": True/False,
# - "confidence_score": 0-10,
# - "explanation": The explanation of your decision,
# - "supporting_evidence": List of relevant evidence OR knowledge points that support your decision

# Where:
# - verdict: 
#   * "True" if the caption appears accurate and properly used in context
#   * "False" if the caption appears inaccurate or out of context
# - confidence_score: 
#   * For evidence-based: 5-10 based on quality of evidence
#   * For knowledge-based: 1-5 reflecting inherent uncertainty
# - explanation: A detailed explanation including:
#   * Whether assessment is evidence-based or knowledge-based
#   * Specific points supporting your verdict
#   * Any concerning elements or inconsistencies found
# - supporting_evidence: 
#   * With evidence: List relevant quotes/references
#   * Without evidence: List relevant knowledge points used in assessment
# """


# # FINAL_CHECKING_PROMPT = """TASK: Synthesize the results from internal validation to assess the final accuracy of the caption.

# # INPUT:
# # Original News Caption: {original_news_caption}
# # Textual Entities (extracted from news content): {textual_entities}
# # Internal Check Result: {internal_result}

# # VALIDATION CRITERIA:
# # 2. If any discrepancies or uncertainties are found in either check, state them clearly. Avoid making unsupported claims or assumptions.
# # 3. If there is a lack of clarity or evidence, indicate uncertainty in the final result.
# # """

# # Final Checking Templates
# FINAL_CHECKING_PROMPT = """TASK: Synthesize the results from both internal and external validation to assess the final accuracy of the caption.

# INPUT:
# Original News Caption: {original_news_caption}
# Textual Entities (extracted from news content): {textual_entities}
# Internal Check Result: {internal_result}
# External Check Result: {external_result}
# External Evidence: {candidates}

# VALIDATION CRITERIA:
# 1. Consider both internal consistency (from textual entities) and external verification.
# 2. If any discrepancies or uncertainties are found in either check, state them clearly.
# 3. Consider the confidence scores from both internal and external checks.
# 4. If there is a lack of clarity or evidence, indicate uncertainty in the final result.
# 5. Pay special attention to cases where internal and external checks might disagree.
# """

# FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
# - "OOC": True/False
# - "confidence_score": 0-10
# - "validation_summary": A concise summary of the validation results
# - "explanation": A detailed explanation for the final result

# Where:
# - OOC: "True" if the caption is out of context based on both internal and external checks, "False" otherwise.
# - confidence_score: A confidence assessment, from 0 to 10, indicating the level of certainty in the final answer.
# - validation_summary: A concise summary of both internal and external validation results, highlighting key points of agreement or disagreement.
# - explanation: A comprehensive explanation (about 1000 words) for your final decisions based on all available evidence. Address any discrepancies between internal and external checks.
# """

# def get_internal_prompt(caption: str, textual_entities: str) -> str:
#     internal_prompt = INTERNAL_CHECKING_PROMPT.format(
#         caption=caption,
#         textual_entities=textual_entities
#     )
#     internal_prompt += INTERNAL_CHECKING_OUTPUT
#     return internal_prompt

# def get_external_prompt(caption: str, candidates: list) -> str:
#     if not candidates:
#         candidates_str = "NO EXTERNAL EVIDENCE AVAILABLE - Proceed with knowledge-based assessment"
#     else:
#         # Format Article dataclass objects into a readable string
#         candidates_str = "\n".join([
#             f"Source {i+1}:\nTitle: {candidate.title}\n"
#             f"Description: {candidate.description}\n"
#             f"Content: {candidate.content}\n"
#             f"Source: {candidate.source_domain}\n"
#             f"URL: {candidate.url}\n"
#             for i, candidate in enumerate(candidates)
#         ])
    
#     external_prompt = EXTERNAL_CHECKING_PROMPT.format(
#         caption=caption,
#         candidates=candidates_str
#     )
#     external_prompt += EXTERNAL_CHECKING_OUTPUT
#     return external_prompt

# def get_final_prompt(
#     caption: str,
#     textual_entities: str,
#     candidates: list,
#     internal_result: dict, 
#     external_result: dict
# ) -> str:
#     # Format candidates string with handling for empty case
#     if not candidates:
#         candidates_str = "NO EXTERNAL EVIDENCE AVAILABLE - Assessment was knowledge-based"
#     else:
#         candidates_str = "\n".join([
#             f"Source {i+1}:\nTitle: {candidate.title}\n"
#             f"Description: {candidate.description}\n"
#             f"Content: {candidate.content}\n"
#             f"Source: {candidate.source_domain}\n"
#             f"URL: {candidate.url}\n"
#             for i, candidate in enumerate(candidates)
#         ])
    
#     final_prompt = FINAL_CHECKING_PROMPT.format(
#         original_news_caption=caption,
#         textual_entities=textual_entities,
#         internal_result=internal_result,
#         external_result=external_result,
#         candidates=candidates_str
#     )
#     final_prompt += FINAL_CHECKING_OUTPUT
    
#     return final_prompt



# templates.py

### Internal Checking Prompt remains the same
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

# Separate External Checking Templates
EXTERNAL_CHECKING_WITH_EVIDENCE_PROMPT = """TASK: Evaluate whether the news caption is accurate and being used in the correct context based on available evidence.

INPUT:
News Caption: {caption}
Retrieved External Evidence:
{candidates}

INSTRUCTIONS:
1. Compare the caption directly with the external evidence:
   - Check if the caption matches the context from the articles
   - Verify names, events, dates, and other factual details
   - Look for any signs of misuse or out-of-context presentation
   - Consider source credibility and temporal relevance
   - Identify any contradictions or inconsistencies
2. Make your judgment based strictly on the evidence provided
3. Be specific about what aspects seem accurate or inaccurate
4. Point out any red flags or concerning elements
"""

EXTERNAL_CHECKING_WITHOUT_EVIDENCE_PROMPT = """TASK: Evaluate the plausibility of the news caption based on general knowledge.

INPUT:
News Caption: {caption}

INSTRUCTIONS:
1. Use your knowledge to assess the caption's plausibility:
   - Check if the caption makes logical sense
   - Verify if the described scenario is consistent with known facts
   - Look for any obvious anachronisms or inconsistencies
   - Consider if the caption follows typical news reporting patterns
2. Make your judgment based purely on general knowledge
3. Be explicit about any assumptions or uncertainties
4. Consider both factual plausibility and contextual appropriateness
5. Maintain a conservative approach - if uncertain, indicate lower confidence
"""

EXTERNAL_CHECKING_WITH_EVIDENCE_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 5-10,
- "explanation": The explanation of your decision,
- "supporting_evidence": List of relevant evidence that supports your decision

Where:
- verdict: "True" if the caption appears accurate and properly used in context, "False" otherwise
- confidence_score: 5-10 based on quality and relevance of evidence
- explanation: A detailed explanation including specific points from evidence supporting your verdict
- supporting_points: List relevant quotes/references from the provided evidence
"""

EXTERNAL_CHECKING_WITHOUT_EVIDENCE_OUTPUT = """\nOUTPUT REQUIRED:
- "verdict": True/False,
- "confidence_score": 1-5,
- "explanation": The explanation of your decision,
- "supporting_knowledge": List of relevant knowledge points used in assessment

Where:
- verdict: "True" if the caption appears plausible based on general knowledge, "False" otherwise
- confidence_score: 1-5 reflecting inherent uncertainty of knowledge-based assessment
- explanation: A detailed explanation of your reasoning and any potential concerns
- supporting_points: List relevant knowledge points used in your assessment
"""

# Final Checking Template updated to handle both cases
# FINAL_CHECKING_PROMPT = """TASK: Synthesize all available evidence (image, internal checks, and external validation) to assess the final accuracy of the caption.

# INPUT:
# Original News Caption: {original_news_caption}
# [An image will be provided for analysis]
# Textual Entities (extracted from news content): {textual_entities}
# Internal Check Result: {internal_result}
# External Check Result: {external_result}
# Evidence Type: {evidence_type}
# {external_evidence}

# VALIDATION CRITERIA:
# 1. Consider both internal consistency (from textual entities) and external verification
# 2. Account for the type of external check performed (evidence-based vs. knowledge-based)
# 3. If any discrepancies or uncertainties are found in either check, state them clearly
# 4. Consider the confidence scores from both checks, noting that knowledge-based external checks inherently have lower confidence
# 5. Pay special attention to cases where internal and external checks might disagree
# 6. Be more conservative in final judgment when external check is knowledge-based
# """

FINAL_CHECKING_PROMPT = """TASK: Synthesize visual evidence, internal validation, and external validation to determine if the caption accurately represents the image and context.

INPUT:
Original News Caption: {original_news_caption}
Image: [A visual element is provided for analysis]
Textual Entities: {textual_entities}
Internal Check Result: {internal_result}
External Check Result: {external_result}
Evidence Type: {evidence_type}
{external_evidence}

VALIDATION CRITERIA:
1. Primary Analysis
   - Does the image content directly support or contradict the caption?
   - Are there visual signs of manipulation or miscontextualization?

2. Supporting Evidence
   - How do internal entities and external findings align with the visual evidence?
   - Consider source reliability: visual > evidence-based > knowledge-based

3. Discrepancy Check
   - Note any conflicts between visual, textual, and external evidence
   - Be conservative when relying on knowledge-based assessment
"""

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": True/False
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation results
- "explanation": A detailed explanation for the final result

Where:
- OOC: "True" if the caption is out of context based on both internal and external checks, "False" otherwise
- confidence_score: 
  * For evidence-based assessments: 0-10 based on both checks
  * For knowledge-based assessments: 0-7 reflecting inherent uncertainty
- validation_summary: A concise summary of both internal and external validation results
- explanation: A comprehensive explanation for your final decisions based on all available evidence
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
    textual_entities: str,
    candidates: list,
    internal_result: dict, 
    external_result: dict
) -> str:
    # Determine evidence type and format evidence string
    if not candidates:
        evidence_type = "KNOWLEDGE_BASED"
        external_evidence = "External Assessment: Based on general knowledge (no direct evidence available)"
    else:
        evidence_type = "EVIDENCE_BASED"
        external_evidence = "External Evidence:\n" + "\n".join([
            f"Source {i+1}:\nTitle: {candidate.title}\n"
            f"Description: {candidate.description}\n"
            f"Content: {candidate.content}\n"
            f"Source: {candidate.source_domain}\n"
            f"URL: {candidate.url}\n"
            for i, candidate in enumerate(candidates)
        ])
    
    final_prompt = FINAL_CHECKING_PROMPT.format(
        original_news_caption=caption,
        textual_entities=textual_entities,
        internal_result=internal_result,
        external_result=external_result,
        evidence_type=evidence_type,
        external_evidence=external_evidence
    )
    final_prompt += FINAL_CHECKING_OUTPUT
    
    return final_prompt