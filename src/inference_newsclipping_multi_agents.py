# inference_use_retrieved_evidences.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional, Union
import openai
from modules import EntitiesModule, GPTConnector, GeminiConnector, TextEvidencesModule, ImageEvidencesModule, GeminiVisionConnector
from dataloaders import cosmos_dataloader
from mdatasets.newsclipping_datasets import MergedBalancedNewsClippingDataset
from dotenv import load_dotenv
import argparse
from huggingface_hub import login
import torch
import json

import time
from src.utils.utils import process_results, NumpyJSONEncoder
import google

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_dataset", 
                       help="")
    parser.add_argument("--entities_path", type=str, default="test_dataset/links_test.json")
    parser.add_argument("--image_evidences_path", type=str, default="queries_dataset/merged_balanced/inverse_search/test/test.json", 
                        help="")
    parser.add_argument("--text_evidences_path", type=str, default="queries_dataset/merged_balanced/direct_search/test/test.json", 
                        help="")
    parser.add_argument("--rewritten_evidence_dir_path", type=str, default="queries_dataset/merged_balanced/r_evidence/test")
    parser.add_argument("--img_des_dir_path", type=str, default="queries_dataset/merged_balanced/consistency_check/test")
    parser.add_argument("--random_index_path", type=str, default=None)
    
    parser.add_argument("--gemini_api_key", type=str, default=None)
    parser.add_argument("--gemini_vlm_api_key", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default="gemini", choices=["gpt", "gemini", "fireworks"])
    parser.add_argument("--vlm_model", type=str, default="gemini", choices=["gpt", "gemini", "fireworks"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--output_dir_path", type=str, default="./result_multi_agents/")
    parser.add_argument("--errors_dir_path", type=str, default="./errors_ranking_consistency_check/")
    
    # Integrated similarity weights
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for visual similarity (S_visual)")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for textual similarity (S_textual)")
    parser.add_argument("--gamma", type=float, default=0.2, help="Weight for interaction term (S_visual * S_textual)")
    
    # Dataloader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_shuffle", action='store_false')
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    
    # Model configs
    parser.add_argument("--ner_model", type=str, default="dslim/bert-large-NER")
    parser.add_argument("--blip_model", type=str, default="Salesforce/blip2-opt-2.7b")
    
    return parser.parse_args()

import time
import os
import json
from typing import Optional, Union, List, Dict, Any

def inference(entities_module: EntitiesModule,
             image_evidences_module: ImageEvidencesModule, 
             text_evidences_module: TextEvidencesModule,
             llm_connector: GPTConnector,
             vlm_connector: Optional[Union[GPTConnector, GeminiConnector]],
             data: dict,
             idx: int,
             rewritten_evidence_dir_path: str,
             img_des_dir_path: str,
             alpha: float = 0.5,
             beta: float = 0.5,
             gamma: float = 0.2):
    """
    Inference function for verification of news images with integrated multi-agent analysis.
    
    The function implements a multi-agent process:
    1. Internal checking agent for initial assessment
    2. Rewrite evidence agent for organizing evidence
    3. Dual retrieval agents (visual and textual)
    4. Detective agent for detailed investigation
    5. Analyst agent for final judgment and reporting
    
    Args:
        entities_module: Module for entity extraction
        image_evidences_module: Module for image evidence retrieval
        text_evidences_module: Module for text evidence retrieval
        llm_connector: Connector for LLM API
        vlm_connector: Connector for Vision Language Model API
        data: Input data containing image and metadata
        idx: Index for current item
        rewritten_evidence_dir_path: Path to save context information
        img_des_dir_path: Path to save image descriptions
        alpha: Weight for visual similarity (S_visual)
        beta: Weight for textual similarity (S_textual)
        gamma: Weight for interaction term (S_visual * S_textual)
        
    Returns:
        Processed verification results
    """
    start_time = time.time()
    
    # Get base64 encoded image
    image_base64 = data["image_base64"]
    
    # Get visual entities from the image
    visual_entities = image_evidences_module.get_entities_by_index(idx)
    
    # Log the entities found
    print(f"Found {len(visual_entities)} visual entities: {', '.join(visual_entities[:5])}...")
    
    # Get evidence using combined similarity scoring
    image_evidence = image_evidences_module.get_evidence_by_index(
        idx, 
        query=data["caption"], 
        reference_image=image_base64, 
        max_results=1, 
        a=alpha,    # Weight for visual similarity
        b=beta,     # Weight for text similarity
        c=gamma     # Weight for interaction term
    )
    
    text_evidence = text_evidences_module.get_evidence_by_index(
        idx, 
        query=data["caption"], 
        reference_image=image_base64, 
        max_results=1, 
        a=alpha,    # Weight for visual similarity
        b=beta,     # Weight for text similarity  
        c=gamma     # Weight for interaction term
    )
    
    # Select the best evidence based on combined score
    if image_evidence == [] and text_evidence == []:
        evidence = None
        print("No evidence found. Falling back to context-based analysis.")
    elif image_evidence == []:
        evidence = text_evidence[0]
        print(f"Using text evidence with score {evidence.combined_score}: {evidence.title}")
    elif text_evidence == []:
        evidence = image_evidence[0]
        print(f"Using image evidence with score {evidence.combined_score}: {evidence.title}")
    else:
        if image_evidence[0].combined_score > text_evidence[0].combined_score:
            evidence = image_evidence[0]
            print(f"Using image evidence with score {evidence.combined_score} (vs text: {text_evidence[0].combined_score}): {evidence.title}")
        else:
            evidence = text_evidence[0]
            print(f"Using text evidence with score {evidence.combined_score} (vs image: {image_evidence[0].combined_score}): {evidence.title}")
    
    # Prepare result structure
    result = {
        "caption": data["caption"],
        "ground_truth": data["label"],
        "visual_entities": visual_entities,
        "inference_time": 0.0
    }
    
    if not evidence:
        return None
    
    # STEP 1: Internal Checking Agent - Basic assessment of news image and caption
    print("STEP 1: Internal Checking Agent...")
    internal_check = internal_checking_agent(
        vlm_connector=vlm_connector,
        image_base64=image_base64,
        caption=data["caption"],
        visual_entities=visual_entities,
        img_des_dir_path=img_des_dir_path,
        idx=idx
    )
    # STEP 2: Rewrite Evidence Agent - Rewrite and structure the evidence
    print("STEP 2: Rewrite Evidence Agent...")
    rewritten_evidence = rewrite_evidence_agent(
        llm_connector=llm_connector,
        evidence=evidence,
        rewritten_evidence_dir_path=rewritten_evidence_dir_path,
        idx=idx,
        visual_entities=visual_entities
    )
    
    # STEP 3: Retrieval Agents - Compare visual and textual elements
    print("STEP 3: Retrieval Agents...")
    # Visual Retrieval Agent
    visual_comparison = visual_retrieval_agent(
        vlm_connector=vlm_connector,
        news_image_base64=image_base64,
        evidence_image_base64=evidence.image_data,
        visual_entities=visual_entities
    )
    
    # Textual Retrieval Agent
    textual_comparison = textual_retrieval_agent(
        llm_connector=llm_connector,
        news_caption=data["caption"],
        news_content=data["content"],
        evidence_content=rewritten_evidence["content"],
        key_factors=rewritten_evidence["key_factors"]
    )
    
    # STEP 4: Detective Agent - Detailed investigation of key elements
    print("STEP 4: Detective Agent...")
    detective_analysis = detective_agent(
        llm_connector=llm_connector,
        visual_comparison=visual_comparison,
        textual_comparison=textual_comparison
    )
    
    # STEP 5: Analyst Agent - Final judgment and reporting
    print("STEP 5: Analyst Agent...")
    verification_result = analyst_agent(
        vlm_connector=vlm_connector,
        image_base64=image_base64,
        evidence=evidence,
        rewritten_evidence=rewritten_evidence,
        caption=data["caption"],
        internal_check=internal_check,
        visual_comparison=visual_comparison,
        textual_comparison=textual_comparison,
        detective_analysis=detective_analysis
    )
    
    # Calculate total inference time
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Prepare final result
    result.update({
        "evidence": evidence.to_dict(),
        "rewritten_evidence": rewritten_evidence,
        "visual_comparison": visual_comparison,
        "textual_comparison": textual_comparison,
        "detective_analysis": detective_analysis,
        "verification_result": verification_result,
        "inference_time": float(inference_time)
    })
    
    return process_results(result)

def internal_checking_agent(vlm_connector, image_base64, caption, visual_entities, img_des_dir_path, idx):
    image_analysis_system_prompt = """
    You are an expert in fact-checking.
    """
    
    image_analysis_prompt = f"""
    Some news captions and accompanying images are inconsistent in terms of key news elements (5W1H) because rumor mongers have taken images from other news and used them as illustrations for current news to make up multimodal misinformation.
    
    Given the image and its caption, analyze if they are consistent with each other. Focus on the following key news elements:
    1. Who (person)
    2. What (event/action)
    3. When (time)
    4. Where (location)
    5. Why (reason)
    6. How (method)
    7. Artwork/objects
    
    For each element that appears in the image or caption, determine if there's consistency or inconsistency.
    
    The news caption is: '{caption}' 
    The detected visual entities include: {', '.join(visual_entities[:15] if len(visual_entities) > 15 else visual_entities)}
    
    Provide a thorough analysis of possible inconsistencies between the image and caption.
    """

    image_analysis = None
    file_path = os.path.join(img_des_dir_path, f"{idx}.json")
    
    if os.path.exists(file_path):
        print(f"Loading existing image description from {file_path}")
        with open(file_path, "r") as f:
            image_analysis = json.load(f)
    else:
        image_analysis = vlm_connector.call_with_structured_output(
            prompt=image_analysis_prompt,
            schema={
                "type": "object",
                "properties": {
                    "consistency_assessment": {
                        "type": "string",
                        "description": "Overall assessment of consistency between image and caption with supporting/contradicting elements"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"],
                        "description": "Confidence level in the analysis"
                    }
                },
                "required": ["consistency_assessment", "confidence"]
            },
            image_base64=image_base64,
            system_prompt=image_analysis_system_prompt
        )

        os.makedirs(img_des_dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(image_analysis, f, indent=2, ensure_ascii=False)
    
    return image_analysis

def rewrite_evidence_agent(llm_connector, evidence, rewritten_evidence_dir_path, idx, visual_entities):
    """
    Rewrite evidence agent that structures evidence into a coherent format.
    
    Args:
        llm_connector: Large language model connector
        evidence: Evidence object containing text from a scraped web source
        visual_entities: List of detected visual entities from the news image
        
    Returns:
        Dictionary containing rewritten evidence
    """
    
    file_path = os.path.join(rewritten_evidence_dir_path, f"{idx}.json")
        
    if os.path.exists(file_path):
        print(f"Loading existing evidence rewrite from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            rewritten_evidence = json.load(f)
            return rewritten_evidence
    
    # Extract evidence information
    evidence_caption = evidence.caption if evidence.caption else ""
    
    if evidence.content != None:
        evidence_content = evidence.content[:2000]
    else:
        evidence_content = ""
    
    evidence_text = f"Title: {evidence.title} \n\n Image Caption: {evidence_caption}" + f"\n\nContext: {evidence_content}"

    # Prompt to rewrite the evidence text into a coherent form
    rewrite_prompt = f"""
    Please analyze and rewrite the following evidence content related to a news item:
    
    EVIDENCE CONTENT:
    {evidence_text}
    
    Generate a concise summary and identify key factors important for verification (people, locations, dates, events). You should include the Image caption to the final content.
    """
    
    # Get rewritten evidence
    rewritten_evidence = llm_connector.call_with_structured_output(
        prompt=rewrite_prompt,
        schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "A coherent summary of the evidence"
                },
                "key_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key factors for news verification"
                }
            },
            "required": ["content", "key_factors"]
        },
        system_prompt="You are an expert in organizing evidence for news verification."
    )
    
    rewritten_evidence["original"] = evidence_text
    
    os.makedirs(rewritten_evidence_dir_path, exist_ok=True)
    with open(os.path.join(rewritten_evidence_dir_path, f"{idx}.json"), "w", encoding="utf-8") as f:
        json.dump(rewritten_evidence, f, indent=2, ensure_ascii=False)
    
    return rewritten_evidence


def visual_retrieval_agent(vlm_connector, news_image_base64, evidence_image_base64, visual_entities):
    """
    Visual retrieval agent that compares the news image with the evidence image.
    
    Args:
        vlm_connector: Vision language model connector
        news_image_base64: Base64 encoded news image
        evidence_image_base64: Base64 encoded evidence image
        visual_entities: List of detected visual entities from the news image
        
    Returns:
        Visual comparison result
    """
    visual_comparison_prompt = f"""
    Compare these two images:
    1. FIRST image: NEWS IMAGE being verified
    2. SECOND image: EVIDENCE IMAGE for comparison
    
    Focus on: visual similarity, subjects, environment, time indicators, and event context.
    
    Detected entities in news image: {', '.join(visual_entities[:10] if len(visual_entities) > 10 else visual_entities)}
    """
    
    visual_comparison = vlm_connector.call_with_structured_output(
        prompt=visual_comparison_prompt,
        schema={
            "type": "object",
            "properties": {
                "visual_similarity_assessment": {
                    "type": "string",
                    "description": "Assessment of visual similarity"
                },
                "key_similarities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key similarities (max 3)"
                },
                "key_differences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key differences (max 3)"
                },
                "same_event": {
                    "type": "boolean",
                    "description": "Whether images appear to be from the same event"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"],
                    "description": "Confidence level"
                }
            },
            "required": ["visual_similarity_assessment", "same_event", "confidence"]
        },
        image_base64=news_image_base64,
        ref_images_base64=evidence_image_base64,
        system_prompt="You are an expert in forensic image analysis."
    )
    
    return visual_comparison


def textual_retrieval_agent(llm_connector, news_caption, news_content, evidence_content, key_factors):
    """
    Textual retrieval agent that compares the news text with the evidence text.
    
    Args:
        llm_connector: Language model connector
        news_caption: News caption
        news_content: News content
        evidence_content: Evidence content
        key_factors: Key factors from rewritten evidence
        
    Returns:
        Textual comparison result
    """
    textual_comparison_prompt = f"""
    Compare:
    
    NEWS CAPTION: {news_caption}
    
    EVIDENCE CONTENT: {evidence_content}
    
    KEY FACTORS FROM EVIDENCE: {', '.join(key_factors)}
    
    Analyze the textual consistency between news and evidence.
    """
    
    textual_comparison = llm_connector.call_with_structured_output(
        prompt=textual_comparison_prompt,
        schema={
            "type": "object",
            "properties": {
                "textual_consistency_assessment": {
                    "type": "string",
                    "description": "Assessment of textual consistency"
                },
                "key_consistencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key consistencies (max 3)"
                },
                "key_inconsistencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key inconsistencies (max 3)"
                },
                "factually_consistent": {
                    "type": "boolean",
                    "description": "Whether the texts are high factually consistent"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"],
                    "description": "Confidence level"
                }
            },
            "required": ["textual_consistency_assessment", "factually_consistent", "confidence"]
        },
        system_prompt="You are an expert in textual analysis and fact-checking."
    )
    
    return textual_comparison


def detective_agent(llm_connector, visual_comparison, textual_comparison):
    """
    Detective agent that conducts detailed investigation of key elements.
    
    Args:
        llm_connector: Language model connector
        visual_comparison: Visual comparison result
        textual_comparison: Textual comparison result
        
    Returns:
        Detective analysis result
    """
    detective_prompt = f"""
    Based on previous analyses:
    
    1. VISUAL COMPARISON: {visual_comparison["visual_similarity_assessment"]} 
       Same event: {"Yes" if visual_comparison["same_event"] else "No"} (Confidence: {visual_comparison["confidence"]})
    
    2. TEXTUAL COMPARISON: {textual_comparison["textual_consistency_assessment"]}
       Factually consistent: {"Yes" if textual_comparison["factually_consistent"] else "No"} (Confidence: {textual_comparison["confidence"]})
    
    IMPORTANT: When the text closely matches but the images don't match (means they are not from the same event), this is STRONG evidence of multimodal misinformation.
    It means someone has taken an unrelated image and used it to illustrate a real news story.
    
    Investigate key elements (time, place, person, event, object) to detect contradictions.
    """
    
    detective_analysis = llm_connector.call_with_structured_output(
        prompt=detective_prompt,
        schema={
            "type": "object",
            "properties": {
                "investigation_summary": {
                    "type": "string",
                    "description": "Summary of investigation findings"
                },
                "key_contradictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key contradictions found (if any)"
                },
                "overall_consistency": {
                    "type": "boolean",
                    "description": "Whether the news is consistent with evidence"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"],
                    "description": "Confidence level"
                }
            },
            "required": ["investigation_summary", "overall_consistency", "confidence"]
        },
        system_prompt="You are a forensic detective specializing in news verification."
    )
    
    return detective_analysis


def analyst_agent(vlm_connector, image_base64, evidence, rewritten_evidence, 
                 caption, internal_check,
                 visual_comparison, textual_comparison, detective_analysis):
    """
    Analyst agent that provides the final judgment and comprehensive verification report.
    
    Args:
        vlm_connector: Vision language model connector
        llm_connector: Language model connector
        image_base64: Base64 encoded news image
        evidence: Evidence object
        rewritten_evidence: Rewritten evidence
        caption: News caption
        content: News content
        visual_entities: Visual entities
        internal_check: Internal check result
        visual_comparison: Visual comparison result
        textual_comparison: Textual comparison result
        detective_analysis: Detective analysis result
        
    Returns:
        Final verification result
    """
    # Key contradictions from detective agent
    contradictions_text = "\n".join([f"- {contradiction}" for contradiction in detective_analysis.get("key_contradictions", [])])
    if not contradictions_text:
        contradictions_text = "No key contradictions identified."
    
    evidence_reliable = visual_comparison["same_event"] or textual_comparison["factually_consistent"]
    
    analyst_prompt = f"""
    News captions and accompanying images are sometimes inconsistent in terms of key news elements (5W1H: Who, What, When, Where, Why, How) 
    because rumor mongers have taken images from other news events and used them as illustrations for current news to create 
    multimodal misinformation. 
    
    Provide a verification report on whether this news image is used in its correct context. 
    
    IMPORTANT: When analyzing news content, consider these scenarios:
    1. MISLEADING CONTEXT: When the text closely matches verified facts but the image is from a different event than claimed, this is STRONG evidence of multimodal misinformation. It suggests someone has taken an unrelated image and used it to illustrate a real news story without proper disclosure.
    2. REPRESENTATIVE CONTEXT: Some legitimate news may use images illustratively rather than literally - these are typically clearly labeled as "file photo," "stock image," "illustration," or otherwise indicated as representational of a broader event rather than the specific incident. This is acceptable journalistic practice when properly disclosed.
    
    {"CRITICAL NOTE: Both visual and textual evidence comparisons are unreliable in this case. You should rely more heavily on the direct internal check of the image-caption pair." if not evidence_reliable else ""}
    
    SUMMARY OF PREVIOUS ANALYSES:
    {"- Internal check: " + internal_check["consistency_assessment"] + " (Confidence: " + internal_check["confidence"] + ")" if not evidence_reliable else ""}
    - Visual comparison: {visual_comparison["visual_similarity_assessment"]} (Same event: {"Yes" if visual_comparison["same_event"] else "No"}. Confidence: {visual_comparison["confidence"]})
    - Textual comparison: {textual_comparison["textual_consistency_assessment"]} (Factually consistent: {"Yes" if textual_comparison["factually_consistent"] else "No"}. Confidence: {textual_comparison["confidence"]})
    - Detective investigation: {detective_analysis["investigation_summary"]} (Overall consistency: {"Yes" if detective_analysis["overall_consistency"] else "No"}. Confidence: {detective_analysis["confidence"]}
    
    KEY CONTRADICTIONS:
    {contradictions_text}
    
    NEWS CAPTION:
    {caption}
    
    EVIDENCE SUMMARY:
    {rewritten_evidence["content"][:500]}
    
    Based on all analyses, provide a final verification report. {"Since both visual and textual evidence comparisons are unreliable, place more weight on the direct internal check of the image-caption pair." if not evidence_reliable else ""}
    """
    
    verification_schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Executive summary of verification findings"
            },
            "is_authentic": {
                "type": "boolean",
                "description": "Whether the image itself is authentic"
            },
            "is_in_context": {
                "type": "boolean",
                "description": "Whether the image is used in appropriate context"
            },
            "context_classification": {
                "type": "string",
                "enum": ["direct_context", "representative_context", "misleading_context"],
                "description": "How the image relates to the claimed context"
            },
            "explanation": {
                "type": "string",
                "description": "Explanation of the verification result"
            },
            "confidence": {
                "type": "string",
                "enum": ["High", "Medium", "Low"],
                "description": "Overall confidence in verification"
            }
        },
        # "context_classification",
        "required": ["summary", "is_authentic", "is_in_context", "context_classification", "explanation", "confidence"]
    }

    verification_result = vlm_connector.call_with_structured_output(
        prompt=analyst_prompt,
        schema=verification_schema,
        image_base64=image_base64,
        ref_images_base64=evidence.image_data,
        system_prompt="You are an expert in fact-checking and news image verification, specializing in detecting multimodal misinformation."
    )

    # Include a simplified record of all agent assessments
    verification_result["agent_assessments"] = {
        "internal_check": {
            "consistency": internal_check["consistency_assessment"],
            "confidence": internal_check["confidence"]
        },
        "visual_comparison": {
            "similarity": visual_comparison["visual_similarity_assessment"],
            "same_event": visual_comparison["same_event"],
            "confidence": visual_comparison["confidence"]
        },
        "textual_comparison": {
            "consistency": textual_comparison["textual_consistency_assessment"],
            "factually_consistent": textual_comparison["factually_consistent"],
            "confidence": textual_comparison["confidence"]
        },
        "detective_analysis": {
            "overall_consistency": detective_analysis["overall_consistency"],
            "investigation_summary": detective_analysis["investigation_summary"],
            "confidence": detective_analysis["confidence"]
        }
    }
    
    return verification_result

def get_transform():
    return None

def main():
    args = arg_parser()
    
    # Setup environment
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])
    
    # Make results and error folders
    if not os.path.exists(args.output_dir_path):
        os.makedirs(args.output_dir_path)
    if not os.path.exists(args.errors_dir_path):
        os.makedirs(args.errors_dir_path)
    
    print(args.gemini_api_key)
    
    print("Connecting to LLM Model...")
    if args.llm_model == "gpt":
        llm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.llm_model == "gemini":
        llm_connector = GeminiConnector(
            api_key=args.gemini_api_key if args.gemini_api_key else os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.0-flash-001"
        )
    else:
        raise ValueError(f"Invalid LLM model: {args.llm_model}")
    print("LLM Model Connected")
        
    print("Connecting to VLM Model...")
    if args.vlm_model == "gpt":
        vlm_connector = GPTConnector(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4o-mini-2024-07-18"
        )
    elif args.vlm_model == "gemini":
        if args.gemini_vlm_api_key:
            api_key = args.gemini_vlm_api_key
        else:
            api_key = args.gemini_api_key if args.gemini_api_key  else os.environ["GEMINI_API_KEY"],
        
        vlm_connector = GeminiVisionConnector(
            api_key=api_key,
            model_name="gemini-2.0-flash-001"
            # model_name="gemini-1.5-flash"
            # model_name="gemini-2.5-pro-exp-03-25"
        )
    else:
        raise ValueError(f"Invalid VLM model: {args.vlm_model}")
    print("VLM Model Connected")
        
    # Initialize modules
    print("Initializing modules...")
    entities_module = EntitiesModule(args.entities_path)
    image_evidences_module = ImageEvidencesModule(args.image_evidences_path)
    text_evidences_module = TextEvidencesModule(args.text_evidences_path)
    print("Modules initialized")
    
    # Load dataset
    dataset = MergedBalancedNewsClippingDataset(args.data_path)
    
    # Determine indices to process
    start_idx = args.start_idx if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else len(dataset) - 1
    
    # Load random indices if specified
    if args.random_index_path is not None:    
        try:
            with open(args.random_index_path, "r") as f:
                random_index = [int(line.strip()) for line in f.readlines()]
        except Exception as e:
            raise e
    else:
        random_index = list(range(start_idx, end_idx + 1))
        
    # Select indices between start_idx and end_idx
    indices = [idx for idx in random_index if start_idx <= idx <= end_idx]
    
    # Validate indices
    if start_idx >= len(dataset):
        raise ValueError(f"Start index {start_idx} is out of range for dataset of length {len(dataset)}")
    if end_idx >= len(dataset):
        end_idx = len(dataset) - 1
    if start_idx > end_idx:
        raise ValueError(f"Start index {start_idx} is greater than end index {end_idx}")
    
    # Process data and save results
    results = []
    error_items = []
    total_start_time = time.time()
    
    print(f"Processing {len(indices)} items from index {start_idx} to {end_idx}")

    for i in range(len(indices)):
        idx = indices[i]
        retry_count = 0
        max_retries = 3
    
        while retry_count <= max_retries:
            try:
                print(f"\n--- Processing item {idx} (Attempt {retry_count + 1}/{max_retries + 1}) ---")
                res_path = os.path.join(args.output_dir_path, f"result_{idx}.json")
                
                # Skip if result already exists and skip_existing is set
                if args.skip_existing and os.path.exists(res_path):
                    print(f"Skipping existing result for index {idx}")
                    break
                
                # Get dataset item
                item = dataset[idx]
                
                # Run inference
                result = inference(
                    entities_module=entities_module,
                    image_evidences_module=image_evidences_module,
                    text_evidences_module=text_evidences_module,
                    llm_connector=llm_connector,
                    vlm_connector=vlm_connector,
                    data=item,
                    idx=idx,
                    rewritten_evidence_dir_path=args.rewritten_evidence_dir_path,
                    img_des_dir_path=args.img_des_dir_path,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma
                )
                
                # Save result
                with open(res_path, "w", encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
                
                results.append(result)
                print(f"Saved result to {res_path}")
                break  # Success - exit the retry loop
                
            except KeyError as e:
                print(f"KeyError processing item {idx}: {e}")
                break  # Don't retry for these errors
            except openai.BadRequestError as e:
                print(f"BadRequestError processing item {idx}: {e}")
                break  # Don't retry for these errors
            except json.decoder.JSONDecodeError as e:
                print(f"JSONDecodeError processing item {idx}: {e}")
                break  # Don't retry for these errors
            except UnicodeEncodeError as e:
                print(f"UnicodeEncodeError processing item {idx}: {e}")
                break  # Don't retry for these errors
            except google.api_core.exceptions.ResourceExhausted as e:
                retry_count += 1
                wait_time = 30
                print(f"Gemini quota exceeded for item {idx} (Attempt {retry_count}/{max_retries + 1})")
                print(f"Waiting {wait_time} seconds before retry...")
                
                if retry_count <= max_retries:
                    time.sleep(wait_time)  # Wait before retry
                else:
                    print(f"Max retries ({max_retries}) exceeded for item {idx}, moving to next item")
                    
            except Exception as e:
                with open(os.path.join(args.errors_dir_path, f"error_{idx}.json"), "w") as f:
                    error_item = {
                        "error": str(e),
                    }
                    json.dump(error_item, f, indent=2, ensure_ascii=False)
                error_items.append(error_item)
                print(f"Error processing item {idx}: {e}")
                # raise e
                
    total_time = time.time() - total_start_time
    
if __name__ == "__main__":
    main()