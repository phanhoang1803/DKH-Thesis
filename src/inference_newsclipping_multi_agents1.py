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
import time
import os
import json
from typing import Optional, Union, List, Dict, Any

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
    parser.add_argument("--output_dir_path", type=str, default="./result_used_html_to_rewrite_evidence/")
    parser.add_argument("--errors_dir_path", type=str, default="./errors_used_html_to_rewrite_evidence/")
    
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
        use_filter_by_domains=False,
        use_filter_non_captions=True,
        a=0.6,    # Weight for visual similarity
        b=0.3,    # Weight for text similarity
        c=0.2     # Weight for interaction term
    )
    
    text_evidence = text_evidences_module.get_evidence_by_index(
        idx, 
        query=data["caption"], 
        reference_image=image_base64, 
        max_results=1, 
        a=0.3,    # Weight for visual similarity
        b=0.6,     # Weight for text similarity  
        c=0.2     # Weight for interaction term
    )
    
    result = {
        "caption": data["caption"],
        "ground_truth": data["label"],
        "visual_entities": visual_entities,
        "inference_time": 0.0,
        "image_evidence": None,
        "text_evidence": None
    }
    
    # Check if we have image evidence
    image_evidence_obj = None
    if image_evidence:
        image_evidence_obj = image_evidence[0]
        print(f"Found image evidence with score {image_evidence_obj.combined_score}: {image_evidence_obj.title}")
        result["image_evidence"] = image_evidence_obj.to_dict()
    else:
        print("No image evidence found.")
    
    # Check if we have text evidence
    text_evidence_obj = None
    if text_evidence:
        text_evidence_obj = text_evidence[0]
        print(f"Found text evidence with score {text_evidence_obj.combined_score}: {text_evidence_obj.title}")
        result["text_evidence"] = text_evidence_obj.to_dict()
    else:
        print("No text evidence found.")
    
    # Check if we have any evidence at all
    if not image_evidence_obj and not text_evidence_obj:
        print("No evidence found. Unable to perform verification.")
        result["has_evidence"] = False
        # return None
    
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
    
    # Initialize for storing rewritten evidence
    image_rewritten_evidence = None
    text_rewritten_evidence = None
    
    # STEP 2: Rewrite Evidence Agent for both image and text evidence
    print("STEP 2: Rewrite Evidence Agents...")
    
    # Process image evidence if available
    if image_evidence_obj:
        image_rewritten_evidence = rewrite_evidence_agent(
            llm_connector=llm_connector,
            evidence=image_evidence_obj,
            rewritten_evidence_dir_path=os.path.join(rewritten_evidence_dir_path, "html_image"),
            idx=idx,
            visual_entities=visual_entities
        )
        result["image_rewritten_evidence"] = image_rewritten_evidence
    
    # Process text evidence if available
    if text_evidence_obj:
        text_rewritten_evidence = rewrite_evidence_agent(
            llm_connector=llm_connector,
            evidence=text_evidence_obj,
            rewritten_evidence_dir_path=os.path.join(rewritten_evidence_dir_path, "html_text"),
            idx=idx,
            visual_entities=visual_entities
        )
        result["text_rewritten_evidence"] = text_rewritten_evidence
       
    # STEP 4: Detective Agent - Detailed investigation of key elements
    print("STEP 4: Detective Agent...")
    detective_analysis = detective_agent(
        llm_connector=llm_connector,
        image_rewritten_evidence=image_rewritten_evidence,
        text_rewritten_evidence=text_rewritten_evidence,
    )
    result["detective_analysis"] = detective_analysis
    
    # STEP 5: Analyst Agent - Final judgment and reporting
    print("STEP 5: Analyst Agent...")
    verification_result = analyst_agent(
        vlm_connector=vlm_connector,
        image_base64=image_base64,
        image_evidence=image_evidence_obj,
        text_evidence=text_evidence_obj,
        image_rewritten_evidence=image_rewritten_evidence,
        text_rewritten_evidence=text_rewritten_evidence,
        caption=data["caption"],
        internal_check=internal_check,
        detective_analysis=detective_analysis
    )
    result["verification_result"] = verification_result
    
    # Calculate total inference time
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    result["inference_time"] = float(inference_time)
    
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
    
    evidence_content = evidence.html_content[:2000]

    evidence_text = f"Title: {evidence.title} \n\n Image Caption: {evidence_caption}" + f"\n\nArticle content: {evidence_content}"

    # Prompt to rewrite the evidence text into a coherent form
    rewrite_prompt = f"""
    Please analyze and rewrite the following evidence information related to a news item:
    
    EVIDENCE INFORMATION:
    {evidence_text}
    
    Generate a concise summary and identify key factors important for verification (people, locations, dates, events). If the article have multiple claims, you should write them all.
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

def detective_agent(llm_connector, image_rewritten_evidence, text_rewritten_evidence):
    # if not image_rewritten_evidence and not text_rewritten_evidence:
    #     return None

    # Prepare summary of available evidence
    evidence_available = []
    if image_rewritten_evidence:
        image_evidence_summary = f"Image evidence: Content: {image_rewritten_evidence.get('content', 'Not available')}\n Key factors: {', '.join(image_rewritten_evidence.get('key_factors', []))}\n"
        evidence_available.append("Image evidence")
    if text_rewritten_evidence:
        text_evidence_summary = f"Text evidence: Content: {text_rewritten_evidence.get('content', 'Not available')}\n Key factors: {', '.join(text_rewritten_evidence.get('key_factors', []))}\n"
        evidence_available.append("Text evidence")
        
    evidence_summary = "Available evidence: " + ", ".join(evidence_available)
    
    # Prepare detective prompt
    detective_prompt = f"""
    Your task is to detect contradictions between the image evidence and text evidence.
    
    {evidence_summary}
    
    {image_evidence_summary if image_rewritten_evidence else ''}
    {text_evidence_summary if text_rewritten_evidence else ''}
    
    Investigate key elements (time, place, people, event, objects) across all available evidence to detect contradictions. 
    Consider the reliable of the evidences because this is news verification. Can omit if it not really reliable.
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
                    "description": "Whether the image evidence and text evidence are consistent"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"],
                    "description": "Confidence level in this assessment"
                }
            },
            "required": ["investigation_summary", "overall_consistency", "confidence"]
        },
        system_prompt="You are a forensic detective who is expert in detecting contradictions between image evidence and text evidence for news verification."
    )
    
    return detective_analysis

def analyst_agent(vlm_connector, image_base64, image_evidence, text_evidence,
                         image_rewritten_evidence, text_rewritten_evidence, caption,
                         internal_check, detective_analysis):
    # Build analyst prompt with all available information
    analyst_prompt = f"""
    News captions and accompanying images are sometimes inconsistent because rumor mongers have taken images from other news events and used them as illustrations for current news to create multimodal misinformation.
    Provide a final verification report on whether this news image is used in its correct context, based on all available evidence and previous analyses.
    
    IMPORTANT: When analyzing news content, consider these scenarios:
    1. MISLEADING CONTEXT: When the text closely matches verified facts but the image is from a different event than claimed, this is STRONG evidence of multimodal misinformation. It suggests someone has taken an unrelated image and used it to illustrate a real news story without proper disclosure.
    2. REPRESENTATIVE CONTEXT: Some legitimate news may use images illustratively rather than literally - these are typically clearly labeled as "file photo," "stock image," "illustration," or otherwise indicated as representational of a broader event rather than the specific incident. This is acceptable journalistic practice when properly disclosed.
    
    *** News information ***
    News caption:
    {caption}
    
    Image related information:
    {image_rewritten_evidence}
    
    Caption related information:
    {text_rewritten_evidence}
    
    *** Analyses ***
    {f"Detective investigation: {detective_analysis}" if detective_analysis else ""}
    
    *** More analysis ***
    
    1. Check whether the image related information is the same as the image information because the image related information is retrieved from the internet through the image search engine.
    2. Check whether the caption related information is the same as the caption information because the caption related information is retrieved from the internet through the text search engine.
    3. Check whether the image is used in the correct context.
    
    Based on all analyses, provide a final verification report with clear reasoning.
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
                "description": "Whether the image itself appears authentic (not manipulated)"
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
            "key_evidence_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key evidence points supporting this conclusion (max 3)"
            },
            "verification_reasoning": {
                "type": "string",
                "description": "Detailed reasoning behind the verification result"
            },
            "confidence": {
                "type": "string",
                "enum": ["High", "Medium", "Low"],
                "description": "Overall confidence in verification"
            }
        },
        "required": ["summary", "is_authentic", "is_in_context", "context_classification", "verification_reasoning", "confidence"]
    }

    verification_result = vlm_connector.call_with_structured_output(
        prompt=analyst_prompt,
        schema=verification_schema,
        image_base64=image_base64,
        system_prompt="You are an expert in fact-checking and news image verification, specializing in detecting multimodal misinformation."
    )

    # Include a simplified record of all agent assessments
    verification_result["agent_assessments"] = {
        "internal_check": {
            "consistency": internal_check["consistency_assessment"],
            "confidence": internal_check["confidence"]
        },
        "detective_analysis": {
            "investigation_summary": detective_analysis["investigation_summary"],
            "overall_consistency": detective_analysis["overall_consistency"],
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
            # model_name="gemini-2.5-pro-exp-03-25"
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
                wait_time = 60
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
                raise e
                
    total_time = time.time() - total_start_time
    
if __name__ == "__main__":
    main()