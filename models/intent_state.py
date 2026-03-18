from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IntentStage(str, Enum):
    EXPLORING = "exploring"
    COMPARING = "comparing"
    DECISION_READY = "decision_ready"
    UNCERTAIN = "uncertain"

class IntentSignal(BaseModel):
    signal_type: str
    confidence: float
    evidence: str
    timestamp: datetime = Field(default_factory=datetime.now)

class IntentState(BaseModel):
    stage: IntentStage = IntentStage.UNCERTAIN
    confidence_score: float = 0.0
    signals: List[IntentSignal] = []
    
    # Stage-specific scores
    exploring_score: float = 0.0
    comparing_score: float = 0.0
    decision_ready_score: float = 0.0
    
    # Thresholds
    EXPLORING_THRESHOLD: float = 0.6
    COMPARING_THRESHOLD: float = 0.65
    DECISION_THRESHOLD: float = 0.75
    
    def update_scores(self, new_signal: IntentSignal):
        """Update intent scores based on new signals"""
        self.signals.append(new_signal)
        
        # Decay older signals (recency bias)
        decay_factor = 0.9
        for i, signal in enumerate(reversed(self.signals[-10:])):
            weight = decay_factor ** i
            
            if "price" in signal.signal_type or "cost" in signal.signal_type:
                self.decision_ready_score += signal.confidence * weight * 0.3
                self.comparing_score += signal.confidence * weight * 0.2
                
            elif "compare" in signal.signal_type or "vs" in signal.signal_type:
                self.comparing_score += signal.confidence * weight * 0.4
                self.decision_ready_score += signal.confidence * weight * 0.1
                
            elif "how does" in signal.signal_type or "what is" in signal.signal_type:
                self.exploring_score += signal.confidence * weight * 0.4
                
            elif "schedule" in signal.signal_type or "demo" in signal.signal_type:
                self.decision_ready_score += signal.confidence * weight * 0.5
                
            elif "timeline" in signal.signal_type or "when" in signal.signal_type:
                self.decision_ready_score += signal.confidence * weight * 0.3
                self.comparing_score += signal.confidence * weight * 0.2
        
        # Normalize scores
        total = self.exploring_score + self.comparing_score + self.decision_ready_score
        if total > 0:
            self.exploring_score /= total
            self.comparing_score /= total
            self.decision_ready_score /= total
        
        # Determine stage
        self._update_stage()
    
    def _update_stage(self):
        """Determine current stage based on scores"""
        max_score = max(self.exploring_score, self.comparing_score, self.decision_ready_score)
        
        if self.decision_ready_score >= self.DECISION_THRESHOLD:
            self.stage = IntentStage.DECISION_READY
            self.confidence_score = self.decision_ready_score
        elif self.comparing_score >= self.COMPARING_THRESHOLD:
            self.stage = IntentStage.COMPARING
            self.confidence_score = self.comparing_score
        elif self.exploring_score >= self.EXPLORING_THRESHOLD:
            self.stage = IntentStage.EXPLORING
            self.confidence_score = self.exploring_score
        else:
            self.stage = IntentStage.UNCERTAIN
            self.confidence_score = max_score

class ConversationContext(BaseModel):
    conversation_id: str
    lead_email: Optional[str] = None
    lead_name: Optional[str] = None
    company: Optional[str] = None
    
    intent_state: IntentState = Field(default_factory=IntentState)
    conversation_history: List[Dict] = []
    recommended_content: List[Dict] = []
    actions_taken: List[Dict] = []
    
    metadata: Dict = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_turn(self, user_message: str, agent_response: str, intent_signal: Optional[IntentSignal] = None):
        """Add a conversation turn and update intent"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "agent": agent_response,
            "intent_stage": self.intent_state.stage.value,
            "confidence": self.intent_state.confidence_score
        })
        
        if intent_signal:
            self.intent_state.update_scores(intent_signal)
        
        self.updated_at = datetime.now()

