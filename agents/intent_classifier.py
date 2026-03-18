import json
import os
from typing import Dict, Tuple

import requests

from models.intent_state import IntentSignal


class IntentClassifier:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        self.intent_prompt_template = """You are an expert sales psychologist analyzing customer messages to determine their buying intent.

Analyze the following customer message and classify their intent stage:

INTENT STAGES:
1. EXPLORING: Customer is learning, asking "what is", seeking educational content
2. COMPARING: Customer is evaluating options, asking about differences, features, or alternatives
3. DECISION_READY: Customer is asking about pricing, timelines, demos, or next steps
4. UNCERTAIN: Message doesn't clearly indicate any stage

CUSTOMER MESSAGE: "{message}"

CONVERSATION HISTORY (last 3 turns):
{history}

Analyze:
1. What specific signals indicate their intent stage?
2. What is their confidence level (0.0 to 1.0)?
3. What evidence supports this classification?

Respond ONLY with valid JSON in this exact format:
{{
    "stage": "exploring|comparing|decision_ready|uncertain",
    "confidence": 0.0-1.0,
    "signal_type": "brief description of the signal",
    "evidence": "specific phrases or context from the message",
    "reasoning": "why you classified it this way"
}}"""

    def classify_intent(self, message: str, conversation_history: list) -> Tuple[IntentSignal, str]:
        """Classify user intent from message"""
        
        # Format history
        history_text = "\n".join([
            f"User: {turn['user']}\nAgent: {turn['agent']}"
            for turn in conversation_history[-3:]
        ])
        
        prompt = self.intent_prompt_template.format(
            message=message,
            history=history_text if history_text else "No previous conversation"
        )
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "format": "json"
                },
                timeout=30
            )
            
            result = response.json()
            analysis = json.loads(result["response"])
            
            signal = IntentSignal(
                signal_type=analysis["signal_type"],
                confidence=float(analysis["confidence"]),
                evidence=analysis["evidence"]
            )
            
            return signal, analysis["reasoning"]
            
        except Exception as e:
            print(f"Intent classification error: {e}")
            # Fallback to basic classification
            return self._fallback_classification(message)
    
    def _fallback_classification(self, message: str) -> Tuple[IntentSignal, str]:
        """Simple rule-based fallback"""
        message_lower = message.lower()
        
        decision_keywords = ["price", "cost", "demo", "schedule", "buy", "purchase", "timeline", "start"]
        comparing_keywords = ["compare", "vs", "versus", "difference", "alternative", "better", "feature"]
        exploring_keywords = ["what is", "how does", "tell me", "explain", "learn", "understand"]
        
        if any(kw in message_lower for kw in decision_keywords):
            return IntentSignal(
                signal_type="decision_keywords_detected",
                confidence=0.6,
                evidence=message[:100]
            ), "Fallback: Decision keywords detected"
            
        elif any(kw in message_lower for kw in comparing_keywords):
            return IntentSignal(
                signal_type="comparison_keywords_detected",
                confidence=0.6,
                evidence=message[:100]
            ), "Fallback: Comparison keywords detected"
            
        elif any(kw in message_lower for kw in exploring_keywords):
            return IntentSignal(
                signal_type="exploration_keywords_detected",
                confidence=0.6,
                evidence=message[:100]
            ), "Fallback: Exploration keywords detected"
            
        else:
            return IntentSignal(
                signal_type="uncertain",
                confidence=0.3,
                evidence=message[:100]
            ), "Fallback: No clear signals"

