import json
import os
from typing import Dict, Optional

import requests
from tools.calendly_connector import CalendlyConnector
from tools.frappe_connector import FrappeConnector

from agents.intent_classifier import IntentClassifier
from agents.knowledge_retriever import KnowledgeRetriever
from models.intent_state import ConversationContext, IntentStage


class SalesOrchestrator:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.knowledge_retriever = KnowledgeRetriever()
        self.frappe = FrappeConnector()
        self.calendly = CalendlyConnector()
        
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = "llama3.2"
        
        # Conversation store (in production, use Redis)
        self.conversations: Dict[str, ConversationContext] = {}
    
    def process_message(self, conversation_id: str, user_message: str, user_metadata: Optional[Dict] = None) -> Dict:
        """Main orchestration method"""
        
        # Get or create conversation context
        context = self.conversations.get(conversation_id)
        if not context:
            context = ConversationContext(conversation_id=conversation_id)
            if user_metadata:
                context.lead_email = user_metadata.get("email")
                context.lead_name = user_metadata.get("name")
                context.company = user_metadata.get("company")
            self.conversations[conversation_id] = context
        
        # Step 1: Classify Intent
        intent_signal, reasoning = self.intent_classifier.classify_intent(
            user_message,
            context.conversation_history
        )
        
        # Update intent state
        context.intent_state.update_scores(intent_signal)
        
        # Step 2: Retrieve relevant knowledge
        retrieved_docs = self.knowledge_retriever.retrieve(
            user_message,
            context.intent_state.stage.value,
            top_k=3
        )
        
        # Step 3: Generate contextual response
        agent_response = self._generate_response(
            user_message,
            context,
            retrieved_docs,
            intent_signal
        )
        
        # Step 4: Determine if action should be taken
        action_taken = None
        if context.intent_state.stage == IntentStage.DECISION_READY and \
           context.intent_state.confidence_score >= context.intent_state.DECISION_THRESHOLD:
            action_taken = self._execute_action(context, agent_response)
        
        # Step 5: Update conversation
        context.add_turn(user_message, agent_response, intent_signal)
        if retrieved_docs:
            context.recommended_content.extend([
                {
                    "timestamp": context.updated_at.isoformat(),
                    "citation": doc["citation"],
                    "relevance": doc["relevance_score"]
                }
                for doc in retrieved_docs
            ])
        
        if action_taken:
            context.actions_taken.append(action_taken)
        
        # Step 6: Update CRM
        self._update_crm(context)
        
        return {
            "response": agent_response,
            "intent_stage": context.intent_state.stage.value,
            "confidence": context.intent_state.confidence_score,
            "action_taken": action_taken,
            "citations": [doc["citation"] for doc in retrieved_docs]
        }
    
    def _generate_response(self, user_message: str, context: ConversationContext, 
                          retrieved_docs: list, intent_signal) -> str:
        """Generate contextual agent response"""
        
        # Build context for LLM
        knowledge_context = "\n\n".join([
            f"[Source: {doc['citation']['source']}]\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        history_context = "\n".join([
            f"User: {turn['user']}\nAgent: {turn['agent']}"
            for turn in context.conversation_history[-3:]
        ])
        
        prompt = f"""You are Revinova's AI Sales Consultant. Your role is to guide prospects naturally through their buying journey.

CURRENT INTENT STAGE: {context.intent_state.stage.value}
CONFIDENCE: {context.intent_state.confidence_score:.2f}
DETECTED SIGNAL: {intent_signal.signal_type}

CONVERSATION HISTORY:
{history_context if history_context else "This is the start of the conversation"}

RELEVANT KNOWLEDGE:
{knowledge_context if knowledge_context else "No specific resources retrieved"}

USER MESSAGE: "{user_message}"

GUIDELINES:
1. If EXPLORING (confidence < 0.65): Provide educational value, share resources, ask discovery questions
2. If COMPARING (0.65-0.74): Highlight differentiation, share spec sheets, address specific concerns
3. If DECISION_READY (0.75+): You may suggest scheduling a demo, but don't be pushy
4. ALWAYS cite your sources using [Source: filename]
5. Keep responses conversational and helpful, not salesy
6. If you don't have information, admit it rather than making claims

Generate a helpful, natural response:"""

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=30
            )
            
            result = response.json()
            return result["response"].strip()
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Could you rephrase that?"
    
    def _execute_action(self, context: ConversationContext, agent_response: str) -> Optional[Dict]:
        """Execute action when ready (e.g., offer Calendly link)"""
        
        # Check if we haven't already offered scheduling
        if any(action.get("type") == "calendly_offered" for action in context.actions_taken):
            return None
        
        # Generate Calendly link
        calendly_link = self.calendly.generate_link(
            email=context.lead_email,
            name=context.lead_name
        )
        
        return {
            "type": "calendly_offered",
            "timestamp": context.updated_at.isoformat(),
            "calendly_link": calendly_link,
            "intent_confidence": context.intent_state.confidence_score
        }
    
    def _update_crm(self, context: ConversationContext):
        """Update Frappe CRM with conversation data"""
        
        lead_data = {
            "email_id": context.lead_email,
            "lead_name": context.lead_name or "Unknown",
            "company_name": context.company,
            "custom_intent_stage": context.intent_state.stage.value,
            "custom_intent_confidence": context.intent_state.confidence_score,
            "custom_last_interaction": context.updated_at.isoformat(),
            "custom_conversation_turns": len(context.conversation_history),
            "notes": f"Latest intent signals: {', '.join([s.signal_type for s in context.intent_state.signals[-3:]])}"
        }
        
        self.frappe.update_lead(context.lead_email, lead_data)

