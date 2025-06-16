from typing import Dict, List

class DebateJudger:
    def __init__(self, vlm_connector):
        self.vlm_connector = vlm_connector

    async def determine_final_verdict(self, debate_history: List[Dict]) -> Dict:
        """Determine the final verdict of the debate with detailed explanation using a judge"""
        # Get the last opinions from both agents
        agent1_opinion = None
        agent2_opinion = None
        agent1_confidence = 0.0
        agent2_confidence = 0.0
        agent1_reasoning = ""
        agent2_reasoning = ""
        
        # Find the most recent opinion and reasoning from each agent
        for round_data in reversed(debate_history):
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
        debate_history_text = self._format_debate_history_for_judge(debate_history)
        
        # Call the judge to make a final determination
        judge_verdict = await self._call_judge(
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

    def _format_debate_history_for_judge(self, debate_history: List[Dict]) -> str:
        """Format the debate history for the judge"""
        history_text = "# Complete Debate History\n\n"
        
        for entry in debate_history:
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

    async def _call_judge(self, agent1_opinion, agent1_confidence, agent1_reasoning, 
                    agent2_opinion, agent2_confidence, agent2_reasoning, debate_history) -> Dict:
        """
        Call a judge to make a final determination
        
        Args:
            agent1_opinion: Opinion of agent 1 (YES/NO)
            agent1_confidence: Confidence of agent 1
            agent1_reasoning: Reasoning of agent 1
            agent2_opinion: Opinion of agent 2 (YES/NO)
            agent2_confidence: Confidence of agent 2
            agent2_reasoning: Reasoning of agent 2
            debate_history: Formatted debate history
        
        Returns:
            Dictionary containing the judge's verdict
        """
        if not self.vlm_connector:
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
            response = await self.vlm_connector.call_with_structured_output(
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
