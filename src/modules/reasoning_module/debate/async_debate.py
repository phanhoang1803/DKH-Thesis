import asyncio
import json
import os
from typing import Dict, List
from modules.evidence_module import ImageEvidencesModule, TextEvidencesModule, EvidenceAggregator, EvidenceSummarizer
from modules.reasoning_module.debate.debate_agent import DebateAgent
from modules.reasoning_module.debate.retrieval_agent import RetrievalAgent
from modules.reasoning_module.debate.report_generator import ReportGenerator
from modules.reasoning_module.debate.debate_judger import DebateJudger

class AsyncDebate:
    """Manages the asynchronous debate between agents"""
    
    def __init__(self, 
                 image_evidences_module: ImageEvidencesModule, 
                 text_evidences_module: TextEvidencesModule, 
                 max_rounds: int = 3, 
                 primary_vlm_connectors=None, 
                 fallback_vlm_connector=None,
                 image_information_save_dir: str = None):
        self.max_rounds = max_rounds
        self.primary_vlm_connectors = primary_vlm_connectors
        self.fallback_vlm_connector = fallback_vlm_connector
        self.image_information_save_dir = image_information_save_dir
        
        print(f"Primary VLM connectors: {primary_vlm_connectors}")
        print(f"Fallback VLM connector: {fallback_vlm_connector}")
        print(f"There are {len(primary_vlm_connectors)} primary VLM connectors")
        
        assert len(primary_vlm_connectors) >= 2, "There must be at least 2 primary VLM connectors"
        assert fallback_vlm_connector != None, "There must be exactly 1 fallback VLM connector"
        
        self.evidence_aggregator = EvidenceAggregator(
            image_evidences_module, 
            text_evidences_module, 
            self._get_connector(2)
        )
        self.evidence_summarizer = EvidenceSummarizer(self._get_connector(3), self._get_connector(2)) # Use connector of EvidenceReranker because now it's not using
        self.retrieval_agent = RetrievalAgent(self._get_connector(4))
        self.report_generator = ReportGenerator(self._get_connector(5))
        self.debate_judger = DebateJudger(self._get_connector(6))
        
        agent1_system_prompt = """You are an evidence-based analyst for image-caption pairs.
        Determine if an image-caption pair is misinformation by examining evidence and visual content.

        Provide your confidence (0.0-1.0) and clear reasoning. Focus on evidence over opinion.
        Respond to counterarguments logically and be willing to adjust your position when needed.
        """

        agent2_system_prompt = """You are a semantic analyst for image-caption pairs.
        Determine if an image-caption pair is misinformation by scrutinizing evidence and looking for inconsistencies.

        Provide your confidence (0.0-1.0) and detailed reasoning. Challenge weak arguments.
        Consider alternative interpretations while maintaining evidence-based analysis.
        """

        self.agent1 = DebateAgent("Agent1", vlm_connector=self._get_connector(0), system_prompt=agent1_system_prompt, stance="YES")
        self.agent2 = DebateAgent("Agent2", vlm_connector=self._get_connector(1), system_prompt=agent2_system_prompt, stance="NO")
        self.debate_history = []
    
    def _get_connector(self, index: int):
        """Return primary connector if available, otherwise fallback"""
        return self.primary_vlm_connectors[index] if len(self.primary_vlm_connectors) > index else self.fallback_vlm_connector
    
    def update_vlm_connector(self, vlm_connector1, vlm_connector2):
        self.vlm_connector1 = vlm_connector1
        self.vlm_connector2 = vlm_connector2
        self.agent1.update_vlm_connector(vlm_connector1)
        self.agent2.update_vlm_connector(vlm_connector2)
        self.retrieval_agent.update_vlm_connector(vlm_connector1)
    
    def _evidence_to_dict(self, evidence_obj):
        """Converts an Evidence object to a dictionary."""
        return {
            "title": evidence_obj.title,
            "caption": evidence_obj.caption,
            "content": evidence_obj.content,
            "domain": evidence_obj.domain,
            "source": evidence_obj.source,
            "image_similarity_score": evidence_obj.image_similarity_score,
            "text_similarity_score": evidence_obj.text_similarity_score,
            "combined_score": evidence_obj.combined_score
        }
    
    async def run_debate(self, index: int, image_base64: str, caption: str, news_content: str):
        """Run the complete debate process and generate structured report"""
        
        # Clear the debate history
        self.agent1.reset()
        self.agent2.reset()
        self.debate_history = []
        
        # 1. Collect evidence
        print("1. Collecting evidence")
        evidence_result = await self.evidence_aggregator.get_aggregated_evidence(index, caption, image_base64)
        
        # If no evidence found, try to get evidence with VLM ranking
        no_evidence = False
        if evidence_result["reranked_evidences"] == []:
            no_evidence = True

        ## Summarize the evidence using VLM
        print("2. Summarizing evidence")
        summary = await self.evidence_summarizer.summarize_evidence(evidences=evidence_result["reranked_evidences"], 
                                                                    image_base64=image_base64, 
                                                                    caption=caption,
                                                                    visual_entities=evidence_result["visual_entities"], 
                                                                    image_information_save_dir=self.image_information_save_dir, 
                                                                    index=index)
    
        ## To list for textual entities
        textual_entities = []
        for entity in evidence_result["textual_entities"]:
            textual_entities.append(entity["word"])
        
        # Convert evidence objects to dictionaries
        evidences = [self._evidence_to_dict(e) for e in evidence_result["evidences"]]
        reranked_evidences = [self._evidence_to_dict(e) for e in evidence_result["reranked_evidences"]]
        
        evidence = {
            "visual_entities": evidence_result["visual_entities"],
            "textual_entities": textual_entities,
            "summary": summary,
            "evidences": evidences,
            "reranked_evidences": reranked_evidences
        }
        
        # 2. Retrieval Agent
        print("3. Analyzing retrieval information")
        retrieval_result = await self.retrieval_agent.analyze(caption=caption, news_content=news_content, evidence=evidence)
    
        # 3. Both agents form initial opinions
        print("4. Forming initial opinions")
        
        agent1_task = self.agent1.form_initial_opinion(caption, evidence, image_base64=image_base64, retrieval_result=retrieval_result)
        agent2_task = self.agent2.form_initial_opinion(caption, evidence, image_base64=image_base64, retrieval_result=retrieval_result)
        
        agent1_opinion, agent2_opinion = await asyncio.gather(agent1_task, agent2_task)
        
        self.debate_history.append({
            "round": 0,
            "agent1": agent1_opinion,
            "agent2": agent2_opinion
        })
        
        # 3. Debate rounds
        print("5. Debating")
        round_num = 1
        while round_num <= self.max_rounds:
            # Get the last response of agents
            last_agent1_response = self.debate_history[-1]["agent1"]
            last_agent2_response = self.debate_history[-1]["agent2"]
            
            if last_agent1_response["opinion"] == last_agent2_response["opinion"]:
                print(f"Debate converged at round {round_num - 1}")
                break
            
            print(f"Debate round {round_num}")
            agent1_response_task = self.agent1.debate_round(
                last_agent2_response, caption, evidence, image_base64, round_num
            )
            agent2_response_task = self.agent2.debate_round(
                last_agent1_response, caption, evidence, image_base64, round_num
            )
            
            agent1_response, agent2_response = await asyncio.gather(agent1_response_task, agent2_response_task)
            
            # Save this round's debate
            round_data = {
                "round": round_num,
                "agent1": agent1_response,
                "agent2": agent2_response
            }
            self.debate_history.append(round_data)
            
            round_num += 1
        
        if round_num > self.max_rounds:
            print(f"Reached maximum rounds ({self.max_rounds}) without convergence")
        
        # 4. Determine final verdict
        print("6. Determining final verdict")
        final_verdict = await self.debate_judger.determine_final_verdict(self.debate_history)
        print(f"Final verdict: {final_verdict['verdict']} with confidence {final_verdict['confidence']:.2f}")
        
        # 5. Generate structured report
        print("7. Generating structured report")
        structured_report = await self.report_generator.generate_structured_report(
            caption=caption,
            news_content=news_content,
            evidence=evidence,
            retrieval_result=retrieval_result,
            final_verdict=final_verdict,
            no_evidence=no_evidence,
            debate_history=self.debate_history
        )
        
        return {
            "debate_history": self.debate_history,
            "retrieval_result": retrieval_result,
            "verdict": final_verdict,
            "evidence": evidence,
            "no_evidence": no_evidence,
            "structured_report": structured_report
        }

    