from typing import Dict, List
from modules.evidence_module import ImageEvidencesModule, TextEvidencesModule, EvidenceAggregator
from modules.reasoning_module.debate.debate_agent import DebateAgent
from modules.reasoning_module.debate.retrieval_agent import RetrievalAgent

class AsyncDebate:
    """Manages the asynchronous debate between agents"""
    
    def __init__(self, image_evidences_module: ImageEvidencesModule, text_evidences_module: TextEvidencesModule, max_rounds: int = 3, vlm_connector1=None, vlm_connector2=None, vlm_connector3=None):
        self.max_rounds = max_rounds
        self.vlm_connector1 = vlm_connector1
        self.vlm_connector2 = vlm_connector2
        self.vlm_connector3 = vlm_connector3
        self.evidence_aggregator = EvidenceAggregator(image_evidences_module, text_evidences_module, vlm_connector3 if vlm_connector3 else vlm_connector1)
        
        agent1_system_prompt = """You are an analytical fact-checker for image-caption pairs.
        Determine if an image-caption pair is misinformation by examining evidence and visual content.

        Provide your confidence (0.0-1.0) and clear reasoning. Focus on evidence over opinion.
        Respond to counterarguments logically and be willing to adjust your position when needed.
        """

        agent2_system_prompt = """You are a critical fact-checker for image-caption pairs.
        Determine if an image-caption pair is misinformation by scrutinizing evidence and looking for inconsistencies.

        Provide your confidence (0.0-1.0) and detailed reasoning. Challenge weak arguments.
        Consider alternative interpretations while maintaining evidence-based analysis.
        """

        self.agent1 = DebateAgent("Agent1", vlm_connector=vlm_connector1, system_prompt=agent1_system_prompt)
        self.agent2 = DebateAgent("Agent2", vlm_connector=vlm_connector2, system_prompt=agent2_system_prompt)
        self.retrieval_agent = RetrievalAgent(vlm_connector=vlm_connector3 if vlm_connector3 else vlm_connector1)
        self.debate_history = []
    
    def update_vlm_connector(self, vlm_connector1, vlm_connector2):
        self.vlm_connector1 = vlm_connector1
        self.vlm_connector2 = vlm_connector2
        self.agent1.update_vlm_connector(vlm_connector1)
        self.agent2.update_vlm_connector(vlm_connector2)
        self.retrieval_agent.update_vlm_connector(vlm_connector1)
    
    def run_debate(self, index: int, image_base64: str, caption: str):
        """Run the complete debate process"""
        
        # Clear the debate history
        self.agent1.reset()
        self.agent2.reset()
        self.debate_history = []
        
        # 1. Collect evidence
        print("1. Collecting evidence")
        evidence_result = self.evidence_aggregator.get_aggregated_evidence(index, caption, image_base64)
        
        if evidence_result["evidences"] == []:
            return {
                "debate_history": self.debate_history,
                "retrieval_result": None,
                "verdict": None,
                "evidence": None
            }
        
        ## Summarize the evidence using VLM
        print("2. Summarizing evidence")
        summary = self._summarize_evidence(evidence_result["evidences"], image_base64)
    
        ## To list for textual entities
        textual_entities = []
        for entity in evidence_result["textual_entities"]:
            textual_entities.append(entity["word"])
        
        ## To list for evidences
        evidences = []
        for evidence in evidence_result["evidences"]:
            evidences.append({
                "title": evidence.title,
                "caption": evidence.caption,
                "content": evidence.content,
                "domain": evidence.domain,
                "source": evidence.source,
                "image_similarity_score": evidence.image_similarity_score,
                "text_similarity_score": evidence.text_similarity_score,
                "combined_score": evidence.combined_score
            })
        
        ## To list for reranked evidences
        reranked_evidences = []
        for evidence in evidence_result["reranked_evidences"]:
            reranked_evidences.append({
                "title": evidence.title,
                "caption": evidence.caption,
                "content": evidence.content,
                "domain": evidence.domain,
                "source": evidence.source,
                "image_similarity_score": evidence.image_similarity_score,
                "text_similarity_score": evidence.text_similarity_score,
                "combined_score": evidence.combined_score
            })
        
        evidence = {
            "visual_entities": evidence_result["visual_entities"],
            "textual_entities": textual_entities,
            "summary": summary,
            "evidences": evidences,
            "reranked_evidences": reranked_evidences
        }
        
        # 2. Retrieval Agent
        print("3. Analyzing retrieval information")
        retrieval_result = self.retrieval_agent.analyze(caption, evidence)
    
        # 3. Both agents form initial opinions
        print("4. Forming initial opinions")
        agent1_opinion = self.agent1.form_initial_opinion(caption, evidence, image_base64=image_base64, retrieval_result=retrieval_result)
        agent2_opinion = self.agent2.form_initial_opinion(caption, evidence, image_base64=image_base64, retrieval_result=retrieval_result)
        
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
            
            print(f"Debate round {round_num}")
            agent1_response = self.agent1.debate_round(
                last_agent2_response, caption, evidence, image_base64, round_num
            )
            agent2_response = self.agent2.debate_round(
                last_agent1_response, caption, evidence, image_base64, round_num
            )
            
            # Save this round's debate
            round_data = {
                "round": round_num,
                "agent1": agent1_response,
                "agent2": agent2_response
            }
            self.debate_history.append(round_data)
            
            # Check if convergence
            if agent1_response["opinion"] == agent2_response["opinion"]:
                print(f"Debate converged after {round_num} rounds")
                break
            
            round_num += 1
        
        if round_num > self.max_rounds:
            print(f"Reached maximum rounds ({self.max_rounds}) without convergence")
        
        # 4. Determine final verdict
        print("6. Determining final verdict")
        final_verdict = self._determine_final_verdict()
        print(f"Final verdict: {final_verdict['verdict']} with confidence {final_verdict['confidence']:.2f}")
        
        return {
            "debate_history": self.debate_history,
            "retrieval_result": retrieval_result,
            "verdict": final_verdict,
            "evidence": evidence
        }
    
    
    def _determine_final_verdict(self) -> Dict:
        """Determine the final verdict of the debate with detailed explanation using a judge"""
        # Get the last opinions from both agents
        agent1_opinion = None
        agent2_opinion = None
        agent1_confidence = 0.0
        agent2_confidence = 0.0
        agent1_reasoning = ""
        agent2_reasoning = ""
        
        # Find the most recent opinion and reasoning from each agent
        for round_data in reversed(self.debate_history):
            if "agent1" in round_data and agent1_opinion is None:
                agent1_opinion = round_data["agent1"]["opinion"]
                agent1_confidence = round_data["agent1"]["confidence"]
                agent1_reasoning = round_data["agent1"]["reasoning"]
            
            if "agent2" in round_data and agent2_opinion is None:
                agent2_opinion = round_data["agent2"]["opinion"]
                agent2_confidence = round_data["agent2"]["confidence"]
                agent2_reasoning = round_data["agent2"]["reasoning"]
            
            if agent1_opinion is not None and agent2_opinion is not None:
                break
        
        # Format the debate history for the judge
        debate_history_text = self._format_debate_history_for_judge()
        
        # Call the judge to make a final determination
        judge_verdict = self._call_judge(
            agent1_opinion=agent1_opinion,
            agent1_confidence=agent1_confidence,
            agent1_reasoning=agent1_reasoning,
            agent2_opinion=agent2_opinion,
            agent2_confidence=agent2_confidence,
            agent2_reasoning=agent2_reasoning,
            debate_history=debate_history_text
        )
        
        # Add the agents' opinions to the verdict
        judge_verdict["agent1_opinion"] = {
            "verdict": agent1_opinion,
            "confidence": agent1_confidence,
            "reasoning": agent1_reasoning
        }
        judge_verdict["agent2_opinion"] = {
            "verdict": agent2_opinion,
            "confidence": agent2_confidence,
            "reasoning": agent2_reasoning
        }
        
        return judge_verdict

    def _format_debate_history_for_judge(self) -> str:
        """Format the debate history for the judge"""
        history_text = "# Complete Debate History\n\n"
        
        for entry in self.debate_history:
            round_num = entry.get('round', 0)
            if round_num == 0:
                history_text += "## Initial Positions\n"
                history_text += f"Agent1 initial opinion: {entry['agent1']['opinion']}\n"
                history_text += f"Agent1 reasoning: {entry['agent1']['reasoning']}\n\n"
                history_text += f"Agent2 initial opinion: {entry['agent2']['opinion']}\n"
                history_text += f"Agent2 reasoning: {entry['agent2']['reasoning']}\n\n"
            else:
                history_text += f"## Round {round_num}\n"
                history_text += f"Agent1: {entry['agent1']['opinion']} (Confidence: {entry['agent1']['confidence']:.2f})\n"
                history_text += f"Agent1 reasoning: {entry['agent1']['reasoning']}\n\n"
                history_text += f"Agent2: {entry['agent2']['opinion']} (Confidence: {entry['agent2']['confidence']:.2f})\n"
                history_text += f"Agent2 reasoning: {entry['agent2']['reasoning']}\n\n"
        
        return history_text

    def _call_judge(self, agent1_opinion, agent1_confidence, agent1_reasoning, 
                    agent2_opinion, agent2_confidence, agent2_reasoning, debate_history) -> Dict:
        """
        Call a judge to make a final determination
        
        Args:
            agent1_opinion: Opinion of agent 1 (REAL/FAKE)
            agent1_confidence: Confidence of agent 1
            agent1_reasoning: Reasoning of agent 1
            agent2_opinion: Opinion of agent 2 (REAL/FAKE)
            agent2_confidence: Confidence of agent 2
            agent2_reasoning: Reasoning of agent 2
            debate_history: Formatted debate history
        
        Returns:
            Dictionary containing the judge's verdict
        """
        if not self.vlm_connector2:
            # If no VLM connector available, fall back to confidence-based decision
            if agent1_opinion == agent2_opinion:
                verdict = agent1_opinion
                confidence = (agent1_confidence + agent2_confidence) / 2
                reason = "Both agents agree."
                explanation = f"Both agents agreed on {verdict}. Agent 1 confidence: {agent1_confidence:.2f}, Agent 2 confidence: {agent2_confidence:.2f}"
            else:
                if agent1_confidence > agent2_confidence:
                    verdict = agent1_opinion
                    confidence = agent1_confidence
                    reason = "Agent 1 has higher confidence."
                else:
                    verdict = agent2_opinion
                    confidence = agent2_confidence
                    reason = "Agent 2 has higher confidence."
                explanation = f"Agents disagreed. Agent 1: {agent1_opinion} ({agent1_confidence:.2f}), Agent 2: {agent2_opinion} ({agent2_confidence:.2f})"
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
                "detailed_explanation": explanation
            }
        
        # Define the schema for the judge's response
        schema = {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["YES", "NO"],
                    "description": "The final verdict on whether the image-caption pair constitutes misinformation"
                },
                "confidence": {
                    "type": "number",
                    "description": "The confidence in the verdict (0.0 to 1.0)"
                },
                "reason": {
                    "type": "string",
                    "description": "A concise reason for the verdict"
                },
                "detailed_explanation": {
                    "type": "string",
                    "description": "A detailed explanation of the verdict, analyzing both agents' arguments"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Key points that led to the final verdict"
                }
            },
            "required": ["verdict", "confidence", "reason", "detailed_explanation", "key_points"]
        }
        
        # Construct the prompt for the judge
        judge_prompt = f"""
        You are a neutral judge evaluating a debate between two AI agents about whether an image-caption pair constitutes misinformation.
        
        # Final Positions
        
        ## Agent 1
        Verdict: {agent1_opinion}
        Confidence: {agent1_confidence:.2f}
        Reasoning: {agent1_reasoning}
        
        ## Agent 2
        Verdict: {agent2_opinion}
        Confidence: {agent2_confidence:.2f}
        Reasoning: {agent2_reasoning}
        
        # Debate History
        {debate_history}
        
        As a judge, your task is to:
        1. Carefully evaluate both agents' arguments
        2. Consider the quality of reasoning, use of evidence, and logical coherence
        3. Determine which verdict (YES or NO) is best supported by the evidence
        4. Provide a confidence score for your verdict
        5. Explain your reasoning in detail
        
        Note that your verdict should NOT be based simply on which agent showed higher confidence or which agent spoke last.
        Instead, focus on the quality of arguments and evidence presented.
        
        Provide your final verdict with a detailed explanation that carefully weighs the strengths and weaknesses of both positions.
        """
        
        system_prompt = """
        You are an expert judge specializing in evaluating debates about misinformation. 
        You carefully analyze arguments from both sides and make an impartial determination based on the quality of reasoning and evidence.
        Your verdicts are thorough, fair, and emphasize critical thinking.
        """
        
        try:
            # Call the VLM to act as judge
            vlm_connector = self.vlm_connector3 if self.vlm_connector3 else self.vlm_connector2
            
            response = vlm_connector.call_with_structured_output(
                prompt=judge_prompt,
                schema=schema,
                system_prompt=system_prompt
            )
            
            # Format the response
            verdict = response.get("verdict", "UNKNOWN")
            confidence = response.get("confidence", 0.5)
            reason = response.get("reason", "")
            detailed_explanation = response.get("detailed_explanation", "")
            key_points = response.get("key_points", [])
            
            formatted_explanation = detailed_explanation + "\n\n**Key Points:**\n" + "\n".join([f"- {point}" for point in key_points])
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
                "detailed_explanation": formatted_explanation
            }
            
        except Exception as e:
            print(f"Error in judge determination: {e}")
            # Fallback to confidence-based decision
            if agent1_opinion == agent2_opinion:
                verdict = agent1_opinion
                confidence = (agent1_confidence + agent2_confidence) / 2
                reason = "Both agents agree."
                explanation = f"Both agents agreed on {verdict}. Agent 1 confidence: {agent1_confidence:.2f}, Agent 2 confidence: {agent2_confidence:.2f}"
            else:
                if agent1_confidence > agent2_confidence:
                    verdict = agent1_opinion
                    confidence = agent1_confidence
                    reason = "Agent 1 has higher confidence."
                else:
                    verdict = agent2_opinion
                    confidence = agent2_confidence
                    reason = "Agent 2 has higher confidence."
                explanation = f"Agents disagreed. Agent 1: {agent1_opinion} ({agent1_confidence:.2f}), Agent 2: {agent2_opinion} ({agent2_confidence:.2f})"
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
                "detailed_explanation": explanation
            }

    def _summarize_evidence(self, evidences: List[Dict], image_base64: str) -> str:
        """Summarize the evidence using VLM"""
        
        text = ""
        for evidence in evidences:
            evidence_caption = evidence.caption if evidence.caption else ""
            if evidence.content != None:
                evidence_content = evidence.content[:2000]
            else:
                evidence_content = ""
    
            evidence_text = f"Title: {evidence.title} \n\n Image Caption: {evidence_caption}" + f"\n\nContent: {evidence_content}"
            text += evidence_text + "\n\n"
        
        text = text.strip()
        
        # System prompt for evidence rewriting
        system_prompt = """
        You are a helpful assistant that rewrites the textual evidence into a coherent form.
        """
        
        # Prompt to rewrite the evidence text into a coherent form
        rewrite_prompt = f"""
        Now I give you the authentic image and its textual evidence.
        
        TEXTUAL EVIDENCE:
        {text}
        
        REMEMBER: DO NOT describe the image, should be only the information from the textual evidence. Pay more attention to caption but don't forget the title and content.
        Please help me generate a coherent and contextually attuned content from the textual evidence. 
        """
        
        # Get rewritten evidence
        vlm_connector = self.vlm_connector3 if self.vlm_connector3 else self.vlm_connector1
        if vlm_connector:
            rewritten_evidence = vlm_connector.call_with_structured_output(
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
            images=[image_base64],
            system_prompt=system_prompt
        )
        
        return rewritten_evidence
