from typing import Dict, List

class ReportGenerator:
    def __init__(self, vlm_connector):
        self.vlm_connector = vlm_connector

    async def generate_structured_report(self, caption: str, news_content: str, evidence: Dict, 
                                  retrieval_result: Dict, final_verdict: Dict, 
                                  no_evidence: bool,
                                  debate_history: List[Dict]) -> str:
        """Generate a structured report using LLM in the required format"""
        
        # Prepare all available data for the LLM
        report_data = {
            "caption": caption,
            "news_content": news_content,
            "final_verdict": final_verdict,
            "evidence": evidence,
            "retrieval_result": retrieval_result,
            "no_evidence": no_evidence,
            "debate_rounds": len(debate_history) - 1,
            "agent_consensus": self._check_agent_consensus(debate_history),
            "debate_summary": self._get_debate_summary(debate_history)
        }
        
        return await self._call_report_generator(report_data)
    
    async def _call_report_generator(self, report_data: Dict) -> str:
        """Generate structured report using LLM"""
    
        if not self.vlm_connector:
            return self._generate_fallback_report(report_data)
        
        # Prepare evidence summary for LLM
        evidence_summary = self._prepare_evidence_for_llm(report_data["evidence"])
        debate_summary = report_data["debate_summary"]
        
        system_prompt = """You are an expert fact-checker and report writer specializing in misinformation verification. 
        You create comprehensive, professional verification reports following a specific structured format.
        Your reports are detailed, evidence-based, and follow journalistic standards for fact-checking."""
        
        # Define the schema for structured output
        report_schema = {
            "type": "object",
            "properties": {
                "case_summary": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "verified": {"type": "string", "enum": ["Verified", "Not Verified", "Partially Verified"]},
                        "geolocation": {"type": "string"},
                        "date": {"type": "string"}
                    },
                    "required": ["summary", "verified", "geolocation", "date"]
                },
                "content_classification": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["tags"]
                },
                "verified_evidence": {
                    "type": "object",
                    "properties": {
                        "source_details": {"type": "string"},
                        "location": {"type": "string"},
                        "time": {"type": "string"},
                        "entities_involved": {"type": "string"},
                        "motivation_intent": {"type": "string"}
                    },
                    "required": ["source_details", "location", "time", "entities_involved", "motivation_intent"]
                },
                "forensic_analysis": {
                    "type": "object",
                    "properties": {
                        "authenticity_assessment": {"type": "string"},
                        "verification_tools_methods": {"type": "array", "items": {"type": "string"}},
                        "synthetic_type": {"type": "string"},
                        "other_artifacts": {"type": "string"}
                    },
                    "required": ["authenticity_assessment", "verification_tools_methods", "synthetic_type", "other_artifacts"]
                },
                "other_evidence_findings": {
                    "type": "object",
                    "properties": {
                        "supporting_sources": {"type": "array", "items": {"type": "string"}},
                        "cross_checking_info": {"type": "string"},
                        "other_info": {"type": "string"}
                    },
                    "required": ["supporting_sources", "cross_checking_info", "other_info"]
                }
            },
            "required": ["case_summary", "content_classification", "verified_evidence", "forensic_analysis", "other_evidence_findings"]
        }
        
        report_prompt = f"""
        Create a comprehensive misinformation verification report based on the following analysis data:

        **CAPTION TO VERIFY:** {report_data["caption"]}
        
        **NEWS CONTENT/CONTEXT:** {report_data["news_content"]}
        
        **FINAL VERDICT:** 
        - Classification: {"Misinformation" if report_data["final_verdict"]["verdict"] == "YES" else "Authentic Content"}
        - Confidence: {report_data["final_verdict"]["confidence"]:.3f}
        - Reasoning: {report_data["final_verdict"]["reason"]}
        - Detailed Explanation: {report_data["final_verdict"].get("detailed_explanation", "N/A")}
        
        **EVIDENCE ANALYSIS:**
        {evidence_summary}
        
        **DEBATE ANALYSIS:**
        - Rounds conducted: {report_data["debate_rounds"]}
        - Agent consensus: {"Achieved" if report_data["agent_consensus"] else "Not achieved"}
        - Debate summary: {debate_summary}
        
        **METADATA:**
        - Evidence availability: {"Limited" if report_data["no_evidence"] else "Comprehensive"}

        Provide a structured verification report with the following components:

        1. **Case Summary**: Comprehensive summary of verification findings, verification status, geolocation, and date
        2. **Content Classification**: Relevant tags based on platforms, people, brands, topics found
        3. **Verified Evidence**: Source details, location info, temporal info, entities involved, and motivation analysis
        4. **Forensic Analysis**: Authenticity assessment, verification methods used, synthetic content detection, and technical findings
        5. **Other Evidence & Findings**: Supporting sources, cross-checking information, and additional relevant details

        IMPORTANT INSTRUCTIONS:
        1. If specific information is not available, use phrases like "Not determined from available evidence" or "Pending further analysis"
        2. Be factual and specific when evidence supports conclusions
        3. Include actual evidence details when available rather than generic statements
        4. For coordinates, dates, and specific details - only include if actually found in evidence
        5. Make the report professional and suitable for fact-checking publication
        6. If evidence is limited, acknowledge this but still provide whatever analysis is possible
        """

        try:
            # Call LLM to generate the structured report with schema
            response = await self.vlm_connector.call_with_structured_output(
                prompt=report_prompt,
                schema=report_schema,
                system_prompt=system_prompt
            )
            
            # Convert structured response to formatted report string
            return self._format_structured_response_to_report(response)
            
        except Exception as e:
            print(f"Error in LLM report generation: {e}")
            return self._generate_fallback_report(report_data)
    
    def _format_structured_response_to_report(self, structured_response: Dict) -> str:
        """Convert structured response to formatted report string"""
    
        try:
            case_summary = structured_response.get("case_summary", {})
            content_classification = structured_response.get("content_classification", {})
            verified_evidence = structured_response.get("verified_evidence", {})
            forensic_analysis = structured_response.get("forensic_analysis", {})
            other_evidence = structured_response.get("other_evidence_findings", {})
            
            report = f"""# Case Summary
{case_summary.get("summary", "No summary available")}
- Verified: {case_summary.get("verified", "Not determined")}
- Geolocation: {case_summary.get("geolocation", "Not determined")}
- Date: {case_summary.get("date", "Not determined")}

# Content Classification
- **Tags:** {", ".join(content_classification.get("tags", ["No tags identified"]))}

# Verified Evidence
- **Source Details:** 
{verified_evidence.get("source_details", "No source details available")}

- **Where? (Location):** {verified_evidence.get("location", "Geographic context not established")}

- **When? (Time):** {verified_evidence.get("time", "Temporal context not established")}

- **Who? (Entities Involved):** {verified_evidence.get("entities_involved", "Key entities not clearly identified")}

- **Why? (Motivation or Intent):** {verified_evidence.get("motivation_intent", "Requires further investigation")}

# Forensic Analysis
- **Authenticity Assessment:** {forensic_analysis.get("authenticity_assessment", "Assessment pending")}

- **Verification Tools & Methods:** {", ".join(forensic_analysis.get("verification_tools_methods", ["Methods not specified"]))}

- **Synthetic Type (if applicable):** {forensic_analysis.get("synthetic_type", "No synthetic content detected")}

- **Other Artifacts:** {forensic_analysis.get("other_artifacts", "No additional artifacts identified")}

# Other Evidence & Findings
- **Supporting Sources:** {", ".join(other_evidence.get("supporting_sources", ["No supporting sources identified"]))}

- **Cross-Checking Information:** {other_evidence.get("cross_checking_info", "Cross-verification pending")}

- **Other Info:** {other_evidence.get("other_info", "No additional information available")}
    """
            
            return report
            
        except Exception as e:
            print(f"Error formatting structured response: {e}")
            return "Error formatting report. Please check the structured response format."
    
    
    def _prepare_evidence_for_llm(self, evidence: Dict) -> str:
        """Prepare evidence data in a readable format for LLM"""
        evidence_text = []
        
        # Visual entities
        if evidence.get("visual_entities"):
            evidence_text.append(f"Visual Entities Detected: {', '.join(evidence['visual_entities'])}")
        
        # Textual entities  
        if evidence.get("textual_entities"):
            evidence_text.append(f"Textual Entities Detected: {', '.join(evidence['textual_entities'])}")
        
        # Evidence summary
        if evidence.get("summary"):
            evidence_text.append(f"Evidence Summary: {evidence['summary']}")
        
        # Top evidence sources
        if evidence.get("reranked_evidences"):
            evidence_text.append("\nTop Evidence Sources:")
            for i, ev in enumerate(evidence["reranked_evidences"][:5], 1):
                evidence_text.append(f"{i}. Title: {ev.get('title', 'N/A')}")
                evidence_text.append(f"   Source: {ev.get('source', 'N/A')}")
                evidence_text.append(f"   Domain: {ev.get('domain', 'N/A')}")
                evidence_text.append(f"   Caption: {ev.get('caption', 'N/A')}")
                if ev.get('content'):
                    content_preview = ev['content'][:500] + "..." if len(ev['content']) > 500 else ev['content']
                    evidence_text.append(f"   Content: {content_preview}")
                evidence_text.append(f"   Similarity Score: {ev.get('combined_score', 'N/A')}")
                evidence_text.append("")
        
        return "\n".join(evidence_text) if evidence_text else "No specific evidence details available"
    
    def _get_debate_summary(self, debate_history: List[Dict]) -> str:
        """Get a summary of the debate between agents"""
        if not debate_history:
            return "No debate conducted"
        
        summary_parts = []
        
        # Initial positions
        if len(debate_history) > 0:
            initial = debate_history[0]
            summary_parts.append(f"Initial positions - Agent1: {initial['agent1']['opinion']}, Agent2: {initial['agent2']['opinion']}")
        
        # Final positions
        if len(debate_history) > 1:
            final = debate_history[-1]
            summary_parts.append(f"Final positions - Agent1: {final['agent1']['opinion']} (confidence: {final['agent1']['confidence']:.2f}), Agent2: {final['agent2']['opinion']} (confidence: {final['agent2']['confidence']:.2f})")
        
        # Key reasoning points
        reasoning_points = []
        for round_data in debate_history[-2:]:  # Last 2 rounds
            if 'agent1' in round_data and round_data['agent1'].get('reasoning'):
                reasoning_points.append(f"Agent1: {round_data['agent1']['reasoning'][:200]}...")
            if 'agent2' in round_data and round_data['agent2'].get('reasoning'):
                reasoning_points.append(f"Agent2: {round_data['agent2']['reasoning'][:200]}...")
        
        if reasoning_points:
            summary_parts.append("Key reasoning: " + " | ".join(reasoning_points))
        
        return ". ".join(summary_parts)
    
    def _generate_fallback_report(self, report_data: Dict) -> str:
        """Generate fallback report when LLM is not available"""
        verdict_text = "misinformation" if report_data["final_verdict"]["verdict"] == "YES" else "authentic content"
        confidence = report_data["final_verdict"]["confidence"]
        
        return f"""# Case Summary  
        After multi-agent analysis, this image-caption pair has been classified as {verdict_text} with confidence {confidence:.3f}.
        - Verified: {"Not Verified" if report_data["final_verdict"]["verdict"] == "YES" else "Verified"}
        - Geolocation: Not determined from available evidence
        - Date: Not determined from available evidence

        # Content Classification  
        - **Tags:** Pending analysis

        # Verified Evidence  
        - **Source Details:** {len(report_data["evidence"].get("reranked_evidences", []))} evidence sources analyzed
        - **Where? (Location):** Pending geolocation analysis
        - **When? (Time):** Pending temporal analysis  
        - **Who? (Entities Involved):** {', '.join(report_data["evidence"].get("visual_entities", [])[:5])}
        - **Why? (Motivation or Intent):** {report_data["final_verdict"].get("detailed_explanation", "Requires further analysis")}

        # Forensic Analysis  
        - **Authenticity Assessment:** {"Content shows signs of manipulation" if report_data["final_verdict"]["verdict"] == "YES" else "Content appears authentic"}
        - **Verification Tools & Methods:** Multi-agent debate system, Evidence aggregation, Similarity analysis
        - **Synthetic Type (if applicable):** Not determined
        - **Other Artifacts:** Confidence score: {confidence:.3f}

        # Other Evidence & Findings  
        - **Supporting Sources:** {len(report_data["evidence"].get("evidences", []))} sources processed
        - **Cross-Checking Information:** Multi-agent verification completed
        - **Other Info:** Evidence availability: {"Limited" if report_data["no_evidence"] else "Comprehensive"}
        """
    
    def _check_agent_consensus(self, debate_history: List[Dict]) -> bool:
        """Check if agents reached consensus in final round"""
        if not debate_history:
            return False
        
        last_round = debate_history[-1]
        agent1_opinion = last_round.get("agent1", {}).get("opinion")
        agent2_opinion = last_round.get("agent2", {}).get("opinion")
        
        return agent1_opinion == agent2_opinion
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp for report"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")