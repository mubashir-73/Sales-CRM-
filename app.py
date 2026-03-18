from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.sales_orchestrator import SalesOrchestrator

app = FastAPI(title="Revinova Sales Agent API")

orchestrator = SalesOrchestrator()

class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    company: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent_stage: str
    confidence: float
    action_taken: Optional[dict] = None
    citations: list

@app.on_event("startup")
async def startup_event():
    """Index knowledge base on startup"""
    print("Indexing knowledge base...")
    try:
        orchestrator.knowledge_retriever.index_knowledge_base()
        print("Agent ready!")
    except Exception as e:
        print(f"WARNING: Could not index knowledge base on startup: {e}")
        print("Agent starting without pre-indexed knowledge. Will retry on first request.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message"""
    try:
        user_metadata = {
            "email": request.user_email,
            "name": request.user_name,
            "company": request.company
        }
        
        result = orchestrator.process_message(
            request.conversation_id,
            request.message,
            user_metadata
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation state"""
    context = orchestrator.conversations.get(conversation_id)
    if not context:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "intent_state": {
            "stage": context.intent_state.stage.value,
            "confidence": context.intent_state.confidence_score,
            "signals": [s.dict() for s in context.intent_state.signals[-5:]]
        },
        "history": context.conversation_history[-10:],
        "recommended_content": context.recommended_content,
        "actions_taken": context.actions_taken
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

