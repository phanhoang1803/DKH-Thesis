from typing import Optional, Dict

COT_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "image_description": {
            "type": "string",
            "description": "Detailed description of what is visibly present in the image"
        },
        "caption_analysis": {
            "type": "string",
            "description": "Analysis of what the caption claims or implies"
        },
        "comparison": {
            "type": "string",
            "description": "Detailed comparison between image content and caption claims"
        },
        "verdict": {
            "type": "string",
            "enum": ["VERIFIED", "PARTIALLY_VERIFIED", "NOT_VERIFIED"],
            "description": "Final verification decision"
        },
        "confidence_score": {
            "type": "integer",
            "description": "Confidence score for the verification decision (0-10)"
        },
        "explanation": {
            "type": "string",
            "description": "Explanation for the verification decision"
        }
    },
    "required": ["image_description", "caption_analysis", "comparison", "verdict", "confidence_score", "explanation"]
}

def get_cot_prompt(caption: str, content: Optional[str] = None):
    """
    Generate a Chain of Thought prompt for news caption verification
    with balanced criteria for verification decisions.
    
    Args:
        caption: The caption to verify
        content: Optional news article content for context
    
    Returns:
        Chain of Thought prompt string
    """
    prompt = f"""
    Task: Verify if the news image caption accurately represents what is shown in the image.

INPUT:
- News Caption: "{caption}"
- News Content: "{content}"

Verification Steps:
1. Describe the Image:
    - Identify key people, objects, and actions.
    - Note any visible text or signs.
    - Focus only on what is clearly seen.
2. Analyze the Caption:
    - Identify main subjects and claims.
    - Separate direct statements from implications.
3. Compare Image vs. Caption:
    - Confirm which claims are directly supported.
    - Identify unverifiable or missing details.
    - Flag contradictions or misleading elements.
4. Decide the Verdict:
    - VERIFIED: Image fully supports the caption.
    - PARTIALLY VERIFIED: Some claims lack full visual proof.
    - NOT VERIFIED: Caption distorts, contradicts, or misrepresents.

Explain the Decision:
    - Provide clear examples from the image.
    - Keep reasoning specific and evidence-based.
    """
    
    return prompt

def get_cot_system_prompt():
    """
    Returns a balanced system prompt for the VLM that encourages considering
    reasonable inferences along with direct visual evidence.
    """
    return """
    You are an expert news verification system designed to determine if image captions 
    accurately represent image content. Analyze both the image and caption carefully, 
    using a balanced approach that considers both what is directly visible and what 
    can be reasonably inferred according to journalistic standards.
    
    Your analysis should be rigorous yet fair, recognizing that news captions often 
    include some context not directly visible in the image while still ensuring 
    the main claims are supported by visual evidence.
    """
    
# The expected output format for final checking
FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "OOC": {
            "type": "boolean",
            "description": "True if the caption is Out of Context, False if it correctly represents the image (Not Out of Context)"
        },
        "confidence_score": {
            "type": "integer",
            "description": "Confidence score for the verification decision (0-10)"
        },
        "validation_summary": {
            "type": "string",
            "description": "A concise summary of the validation findings"
        },
        "explanation": {
            "type": "string",
            "description": "Detailed justification of why the image is or isn't out of context"
        }
    },
    "required": ["OOC", "confidence_score", "validation_summary", "explanation"]
}

FINAL_CHECKING_OUTPUT = """\nOUTPUT REQUIRED:
- "OOC": Boolean - false if the caption correctly represents the image (Not Out of  ontext), true if it misrepresents the image (Out of Context)
- "confidence_score": 0-10
- "validation_summary": A concise summary of the validation findings
- "explanation": Detailed justification of why the image is or isn't out of context

Where:
- OOC (Out of Context): Boolean value of "false" if the caption provides a correct representation of the image content, "true" otherwise.
- confidence_score: 0-10 (reflecting overall certainty in the verdict based on combined analysis).
- validation_summary: A brief (1-2 sentence) summary highlighting whether viewers would be misled about what they're seeing.
- explanation: A detailed, evidence-based justification (max 500 words) that examines what's actually shown in the image versus what the caption claims or implies is shown.
"""

def get_cot_final_prompt(caption: str, content: str, cot_result: Dict):
    """
    Generate a simplified final checking prompt for Chain of Thought verification results.
    
    Args:
        caption: The news caption
        content: The news content
        cot_result: The structured CoT verification result
        
    Returns:
        Final checking prompt string
    """
    prompt = f"""Task: Confirm if the caption accurately represents the image.

Input:
- Image (directly analyzed)
- Caption: "{caption}"
- News Content: "{content}" (for background, not primary evidence)
- CoT Verification Findings: {cot_result}

Final Check Steps:

1. Review CoT findings to confirm if the image fully supports the caption.
2. Assess Misleading Potential:
    - If the main claim is visually supported, mark Not Out of Context (OOC = False).
    - If the caption distorts or misrepresents, mark Out of Context (OOC = True).
3. Make the Decision:
    - Would a viewer misunderstand the image based on the caption?
    - Does the caption add or remove critical context?
    
Now, let’s finalize the verification.
"""

    prompt += FINAL_CHECKING_OUTPUT  # Add the output format requirements
    
    return prompt


def cot_final_check(data: Dict, image_base64: str, cot_result: Dict, vlm_connector) -> Dict:
    """
    Perform the final checking for Chain of Thought verification results.
    
    Args:
        data: Dictionary containing caption and content
        image_base64: Base64 encoded image
        cot_result: The CoT verification result
        vlm_connector: Connector to vision-language model
        
    Returns:
        Final verification result
    """
    # Generate the CoT-specialized final checking prompt
    final_prompt = get_cot_final_prompt(
        caption=data["caption"],
        content=data.get("content", ""),
        cot_result=cot_result
    )
    
    # Call VLM with the final checking prompt
    final_result = vlm_connector.call_with_structured_output(
        prompt=final_prompt,
        schema=FINAL_RESPONSE_SCHEMA, 
        image_base64=image_base64
    )
    
    return final_result