# inference_use_retrieved_evidences.py

from datetime import datetime
from typing import Optional, Union
import numpy as np
import openai
from modules import EntitiesModule, GPTConnector, GeminiConnector, ExternalRetrievalModule, TextEvidencesModule, Evidence, ImageEvidencesModule
from dataloaders import cosmos_dataloader
from mdatasets.newsclipping_datasets import MergedBalancedNewsClippingDataset
from src.modules.evidence_retrieval_module.scraper.scraper import Article
from templates import get_visual_prompt, get_final_prompt
from templates_for_generating_context import CAPTION_CONTEXT_CHECKING_RESPONSE_SCHEMA, CONTEXT_RESPONSE_SCHEMA, get_context_prompt, get_caption_context_checking_prompt, SYSTEM_PROMPT_FOR_VLM_GENERATED_CONTEXT, SYSTEM_PROMPT_FOR_CAPTION_CONTEXT_CHECKING
import os
from dotenv import load_dotenv
import argparse
from huggingface_hub import login
from torchvision import transforms
from typing_extensions import TypedDict
import torch
import json
import time
from src.config import NEWS_SITES, FACT_CHECKING_SITES
from src.utils.utils import process_results, NumpyJSONEncoder, EvidenceCache
from src.modules.reasoning_module.connector.gpt import VISUAL_RESPONSE_SCHEMA, FINAL_RESPONSE_SCHEMA
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
    parser.add_argument("--context_dir_path", type=str, default="queries_dataset/merged_balanced/context/test")
    parser.add_argument("--img_des_dir_path", type=str, default="queries_dataset/merged_balanced/image_description/test")
    parser.add_argument("--random_index_path", type=str, default=None)
    
    parser.add_argument("--gemini_api_key", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default="gemini", choices=["gpt", "gemini", "fireworks"])
    parser.add_argument("--vlm_model", type=str, default="gemini", choices=["gpt", "gemini", "fireworks"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--output_dir_path", type=str, default="./result_ranking_lion/")
    parser.add_argument("--errors_dir_path", type=str, default="./errors_ranking_lion/")
    
    # Integrated similarity weights
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for visual similarity (S_visual)")
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

def inference(entities_module: EntitiesModule,
             image_evidences_module: ImageEvidencesModule, 
             text_evidences_module: TextEvidencesModule,
             llm_connector: GPTConnector,
             vlm_connector: Optional[Union[GPTConnector, GeminiConnector]],
             data: dict,
             idx: int,
             context_dir_path: str,
             img_des_dir_path: str,
             alpha: float = 0.7,
             beta: float = 0.3,
             gamma: float = 0.5):
    """
    Inference function for verification of news images with integrated similarity scoring.
    
    The function implements a two-step process:
    1. Rewriting evidence using Q-former for image descriptions and evidence text
    2. Providing explanation using image, descriptions, claim, and evidence
    
    Args:
        entities_module: Module for entity extraction
        image_evidences_module: Module for image evidence retrieval
        text_evidences_module: Module for text evidence retrieval
        llm_connector: Connector for LLM API
        vlm_connector: Connector for Vision Language Model API
        data: Input data containing image and metadata
        idx: Index for current item
        context_dir_path: Path to save context information
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
    
    # If no evidence found, generate context directly
    if evidence is None:
        return generate_context_based_analysis(
            vlm_connector=vlm_connector,
            llm_connector=llm_connector,
            data=data,
            visual_entities=visual_entities,
            image_base64=image_base64,
            idx=idx,
            context_dir_path=context_dir_path,
            result=result,
            start_time=start_time
        )
    
    # STEP 1: Rewriting evidence
    # Use Q-former to get image descriptions, and use the evidence text to get the final content
    print("STEP 1: Rewriting evidence...")
    rewritten_evidence = rewrite_evidence(
        llm_connector=llm_connector,
        image_base64=image_base64,
        evidence=evidence,
        visual_entities=visual_entities
    )
    
    # STEP 2: Give explanation
    # Use Q-former to get the news image descriptions and verify the content
    print("STEP 2: Generating explanation...")
    verification_result = generate_explanation(
        vlm_connector=vlm_connector,
        llm_connector=llm_connector,
        image_base64=image_base64,
        evidence=evidence,
        rewritten_evidence=rewritten_evidence,
        caption=data["caption"],
        content=data["content"],
        visual_entities=visual_entities,
        img_des_dir_path=img_des_dir_path,
        idx=idx
    )
    
    # Calculate total inference time
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Prepare final result
    result.update({
        "evidence": evidence.to_dict(),
        "rewritten_evidence": rewritten_evidence,
        "verification_result": verification_result,
        "inference_time": float(inference_time)
    })
    
    return process_results(result)


def rewrite_evidence(llm_connector, image_base64, evidence, visual_entities):
    """
    Rewrite evidence into a coherent, contextually attuned format.
    
    This function focuses on formatting the evidence text (caption, content) into a
    coherent and structured form, not analyzing the news image.
    
    Args:
        llm_connector: Large language model connector
        image_base64: Base64 encoded image (the news image to verify)
        evidence: Evidence object containing text from a scraped web source
        visual_entities: List of detected visual entities from the news image
        
    Returns:
        Dictionary containing rewritten evidence
    """
    # Extract evidence information
    evidence_caption = evidence.caption if evidence.caption else ""
    evidence_content = evidence.content
    
    # if evidence.content and evidence_text == "":
    #     evidence_text += "\n\n" + evidence.content[:2000]  # Limit content length
    
    # System prompt for evidence rewriting
    system_prompt = """
    You are an expert assistant that specializes in organizing and rewriting evidence for news verification. 
    """
    
    # Prompt to rewrite the evidence text into a coherent form
    rewrite_prompt = f"""
    Now I give you the evidence content.
    
    EVIDENCE CONTENT:
    {evidence_caption}
    
    Please help me generate a coherent and contextually attuned content without change the core information of the textual evidence.
    """
    
    # Get rewritten evidence
    rewritten_evidence = llm_connector.call_with_structured_output(
        prompt=rewrite_prompt,
        schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "A coherent and contextually attuned content"
                }
            },
            "required": ["content"]
        },
        system_prompt=system_prompt
    )
    
    rewritten_evidence["original"] = evidence_caption
    
    return rewritten_evidence


def generate_explanation(vlm_connector, llm_connector, image_base64, evidence, 
                         rewritten_evidence, caption, content, visual_entities, img_des_dir_path, idx):
    """
    Generate a comprehensive explanation and verification report by comparing
    the news image with the evidence.
    
    Args:
        vlm_connector: Vision language model connector
        llm_connector: Language model connector
        image_base64: Base64 encoded image (the news image to verify)
        evidence: Evidence object containing text from a scraped web source
        rewritten_evidence: Rewritten evidence from step 1
        caption: News image caption to verify
        content: News content
        visual_entities: List of detected visual entities from the news image
        
    Returns:
        Verification result with detailed analysis
    """
    # First, analyze the news image itself
    # System prompt for image analysis
    image_analysis_system_prompt = """
    You are an expert visual analyst specializing in news image verification. 
    Analyze thoroughly but maintain objectivity. Your goal is to create a factual foundation for verification.
    """
    
    image_analysis_prompt = f"""
    Analyze this news image in detail. Consider the following queries to guild your analysis:
    - What is happening in this image?
    - Who are the main subjects visible?
    - What is the setting or location?
    - What actions are being performed?
    - What emotions are displayed?
    - What visual cues suggest this is a news event?
    - How are the subjects positioned or arranged?
    - What contextual details provide information about when this occurred?
    - What notable objects or symbols are present?
    - How does this image relate to current events?
    
    Known entities in the image: {', '.join(visual_entities[:10] if len(visual_entities) > 10 else visual_entities)}
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
                    "detailed_description": {
                        "type": "string",
                        "description": "A comprehensive description of the image contents"
                    },
                    "key_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of important elements in the image"
                    },
                    "possible_context": {
                        "type": "string",
                        "description": "Possible context or situation depicted in the image"
                    }
                },
                "required": ["detailed_description", "key_elements"]
            },
            image_base64=image_base64,
            system_prompt=image_analysis_system_prompt
        )

        os.makedirs(img_des_dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(image_analysis, f, indent=2, ensure_ascii=False)
    
    # Now create a verification prompt that compares the news image with the evidence
    # System prompt for verification reporting
    verification_system_prompt = """
    You are a forensic image verification expert specializing in news content authentication. 
    Your verification reports should be comprehensive, balanced, and clearly articulate your reasoning process and confidence level. Avoiding make assumptions.
    """
    
    verification_prompt = f"""
    I need a comprehensive verification report for a news image. Here's all the available information:
    
    NEWS IMAGE ANALYSIS:
    {image_analysis['detailed_description']}
    
    KEY ELEMENTS IN NEWS IMAGE:
    {', '.join(image_analysis['key_elements'])}
    
    CLAIM (CAPTION):
    {caption}
    
    VISUAL SIMILARITY SCORE BETWEEN THE NEWS IMAGE AND THE IMAGE THAT THE EVIDENCE IS REFERRING TO: {evidence.image_similarity_score}
    
    EVIDENCE CONTENT:
    {rewritten_evidence['content']}
    
    
    DETECTED ENTITIES IN NEWS IMAGE:
    {', '.join(visual_entities[:15] if len(visual_entities) > 15 else visual_entities)}
    
    IMPORTANT: 
    1. The visual similarity score indicates how closely the evidence image matches the news image.
    
    Based on the comparison between the news image and the evidence, as well as the provided information above, provide a detailed verification report.
    Focus on:
    1. Whether the image is rightly used by the claim/caption
    2. The authenticity of the image (real, altered, AI-generated)
    3. Source verification (where and when the image originated)
    4. Contextual accuracy, including:
    - Whether the evidence supports that the news image depicts the claimed event
    - Consider whether the claim is about a different image or event than the news image based on evidence and visual similarity score
    - If the evidence appears unrelated to both the news image and the claim, base verification primarily on the image analysis and clearly state that the provided evidence doesn't address this specific news item.
    
    Remember to separate facts from speculation and clearly indicate your confidence level in different aspects of your analysis.
    """
    
    # Define a comprehensive schema for verification reporting
    verification_schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Executive summary of verification findings"
            },
            "content_classification": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relevant tags (platforms, people, topics)"
            },
            "source_details": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Where content originated"},
                    "location": {"type": "string", "description": "Geographical context"},
                    "time_period": {"type": "string", "description": "When image was created"},
                    "entities_involved": {"type": "array", "items": {"type": "string"}, "description": "Key people/organizations"},
                    "possible_intent": {"type": "string", "description": "Likely purpose of content"}
                }
            },
            "authenticity_assessment": {
                "type": "object",
                "properties": {
                    "is_authentic": {"type": "boolean", "description": "Whether content (image) is authentic"},
                    "modification_type": {"type": "string", "description": "Type of modification if not authentic"},
                    "verification_methods": {"type": "array", "items": {"type": "string"}, "description": "Methods used"},
                    "noted_artifacts": {"type": "array", "items": {"type": "string"}, "description": "Any detected anomalies"}
                }
            },
            "contextual_accuracy": {
                "type": "object",
                "properties": {
                    "in_context": {"type": "boolean", "description": "Does the news image is rightly used by the news claim (caption) based on the contextual accuracy analysis?"},
                    "explanation": {"type": "string", "description": "Explain the conclusion"},
                },
                "required": ["in_context", "explanation"]
            },
            "supporting_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional evidence supporting conclusions"
            },
            "confidence_level": {
                "type": "string",
                "enum": ["High", "Medium", "Low"],
                "description": "Overall confidence in verification results"
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recommendations for readers/users"
            }
        },
        "required": ["summary", "content_classification", "source_details", 
                    "authenticity_assessment", "contextual_accuracy", "confidence_level"]
    }
    
    # Get the verification result
    verification_result = vlm_connector.call_with_structured_output(
        prompt=verification_prompt,
        schema=verification_schema,
        image_base64=image_base64,
        system_prompt=verification_system_prompt
    )
    
    # Include image analysis in the result for transparency
    verification_result["image_analysis"] = {
        "description": image_analysis["detailed_description"],
        "key_elements": image_analysis["key_elements"]
    }
    
    return verification_result


def generate_context_based_analysis(vlm_connector, llm_connector, data, visual_entities, 
                                   image_base64, idx, context_dir_path, result, start_time):
    """
    Generate analysis based on context when no suitable evidence is found.
    
    Args:
        vlm_connector: Vision language model connector
        llm_connector: Language model connector
        data: Input data
        visual_entities: Detected visual entities
        image_base64: Base64 encoded image
        idx: Current item index
        context_dir_path: Path to save context
        result: Current result structure
        start_time: Start time for inference
        
    Returns:
        Updated result with context-based analysis
    """
    # Check if context already exists
    context_result = None
    context_file_path = os.path.join(context_dir_path, f"{idx}.json")
    
    if os.path.exists(context_file_path):
        print(f"Loading existing context from {context_file_path}")
        with open(context_file_path, "r") as f:
            context_result = json.load(f)
    else:
        # Generate context from image
        print("Generating context from image (no evidence found)")
        
        context_prompt = f"""
        Analyze this image thoroughly and provide comprehensive context.
        
        I can see these potential entities in the image: {', '.join(visual_entities[:15] if len(visual_entities) > 15 else visual_entities)}
        
        The image has this caption: "{data['caption']}"
        
        Additional content: "{data['content'][:500] if 'content' in data and data['content'] else 'No additional content provided.'}"
        
        Please provide:
        1. A detailed description of what you see
        2. Historical or cultural context related to the image
        3. Potential sources or origins for this type of image
        4. Any technical observations about the image qualities
        5. Assessment of whether the caption accurately describes the image
        """
        
        context_schema = {
            "type": "object",
            "properties": {
                "image_description": {
                    "type": "string",
                    "description": "Detailed description of image contents"
                },
                "historical_context": {
                    "type": "string",
                    "description": "Relevant historical or cultural context"
                },
                "potential_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Possible sources or origins"
                },
                "technical_assessment": {
                    "type": "string",
                    "description": "Technical observations about image qualities"
                },
                "caption_accuracy": {
                    "type": "object",
                    "properties": {
                        "is_accurate": {"type": "boolean"},
                        "explanation": {"type": "string"}
                    }
                }
            },
            "required": ["image_description", "historical_context", "potential_sources", "caption_accuracy"]
        }
        
        context_result = vlm_connector.call_with_structured_output(
            prompt=context_prompt,
            schema=context_schema,
            image_base64=image_base64
        )
        
        # Save the context result
        os.makedirs(context_dir_path, exist_ok=True)
        with open(context_file_path, "w") as f:
            json.dump(context_result, f, indent=2, ensure_ascii=False)
    
    # Generate verification report based on context
    print("Generating verification report based on context...")
    verification_prompt = f"""
    I need to verify a news image but have no external evidence. I'll use direct image analysis.
    
    CLAIM (CAPTION): 
    {data['caption']}
    
    IMAGE DESCRIPTION:
    {context_result['image_description']}
    
    HISTORICAL CONTEXT:
    {context_result['historical_context']}
    
    POTENTIAL SOURCES:
    {', '.join(context_result['potential_sources'])}
    
    CAPTION ACCURACY ASSESSMENT:
    {'Accurate' if context_result['caption_accuracy']['is_accurate'] else 'Inaccurate'}: {context_result['caption_accuracy']['explanation']}
    
    Based solely on the image analysis, provide a verification report that acknowledges the limitations of not having external evidence.
    """
    
    # Same schema as in generate_explanation
    verification_schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Executive summary of verification findings"
            },
            "content_classification": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relevant tags (platforms, people, topics)"
            },
            "source_details": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Where content originated"},
                    "location": {"type": "string", "description": "Geographical context"},
                    "time_period": {"type": "string", "description": "When image was created"},
                    "entities_involved": {"type": "array", "items": {"type": "string"}, "description": "Key people/organizations"},
                    "possible_intent": {"type": "string", "description": "Likely purpose of content"}
                }
            },
            "authenticity_assessment": {
                "type": "object",
                "properties": {
                    "is_authentic": {"type": "boolean", "description": "Whether content is authentic"},
                    "modification_type": {"type": "string", "description": "Type of modification if not authentic"},
                    "verification_methods": {"type": "array", "items": {"type": "string"}, "description": "Methods used"},
                    "noted_artifacts": {"type": "array", "items": {"type": "string"}, "description": "Any detected anomalies"}
                }
            },
            "contextual_accuracy": {
                "type": "object",
                "properties": {
                    "caption_matches_image": {"type": "boolean", "description": "Does caption match image?"},
                    "misrepresentation_details": {"type": "string", "description": "Any misrepresentation details"},
                    "missing_context": {"type": "string", "description": "Important context that's missing"}
                }
            },
            "supporting_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional evidence supporting conclusions"
            },
            "confidence_level": {
                "type": "string",
                "enum": ["High", "Medium", "Low"],
                "description": "Overall confidence in verification results"
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recommendations for readers/users"
            }
        },
        "required": ["summary", "content_classification", "source_details", 
                    "authenticity_assessment", "contextual_accuracy", "confidence_level"]
    }
    
    verification_result = vlm_connector.call_with_structured_output(
        prompt=verification_prompt,
        schema=verification_schema,
        image_base64=image_base64
    )
    
    # Calculate inference time
    inference_time = time.time() - start_time
    print(f"Context-based analysis completed in {inference_time:.2f} seconds")
    
    # Update result
    result.update({
        "evidence": None,
        "context_analysis": context_result,
        "verification_result": verification_result,
        "inference_time": float(inference_time)
    })
    
    return process_results(result)

def get_transform():
    return None

def main():
    args = arg_parser()
    
    # Setup environment
    load_dotenv()
    
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
        vlm_connector = GeminiConnector(
            api_key=args.gemini_api_key if args.gemini_api_key else os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.0-flash-001"
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
                    context_dir_path=args.context_dir_path,
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