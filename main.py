# FastAPI REST API — the bridge between the frontend chat UI and the AI engine.
# Run with: uvicorn main:app --reload
#
# Endpoints:
#   POST /chat            → Send a message, get AI response
#   POST /conversation    → Start a new conversation
#   GET  /conversations/{user_id} → Get all sessions
#   GET  /history/{conv_id}       → Get message history
#   GET  /tickets/{user_id}       → Get support tickets
#   POST /rebuild-kb              → Rebuild knowledge base (admin)

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from chatbot_engine import SupportChatbotEngine
from intent_detector import IntentDetector
from database import (
    init_db, create_conversation, save_message,
    get_conversation_history, get_all_conversations,
    create_ticket, get_user_tickets
)

load_dotenv()


# ── Request/Response schemas ──────────────────────────────────────────────────
# Pydantic models validate incoming request data automatically

class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str

class NewConversationRequest(BaseModel):
    user_id: str

class ChatResponse(BaseModel):
    answer: str
    intent: str
    sentiment: str
    escalated: bool
    ticket_id: Optional[str] = None
    conversation_id: str


# ── App lifecycle ─────────────────────────────────────────────────────────────

# Global instances — created once at startup, reused for every request
engine = None
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup — initializes all heavy resources."""
    global engine, detector
    print("Starting up...")
    
    init_db()                          # Create SQLite tables
    engine = SupportChatbotEngine()    # Load/build ChromaDB knowledge base
    detector = IntentDetector()        # Initialize intent classifier
    
    print("All systems ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="AI Support Chatbot API",
    description="Customer support chatbot with RAG + Intent Detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allows the frontend HTML to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend HTML as a static file
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serves the chat UI when you open http://localhost:8000"""
    return FileResponse("frontend/index.html")


@app.post("/conversation")
async def new_conversation(request: NewConversationRequest):
    """Creates a new conversation session."""
    conv_id = create_conversation(request.user_id)
    return {"conversation_id": conv_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — the heart of the API.
    
    Flow:
    1. Detect intent + sentiment from user message
    2. Load conversation history from SQLite
    3. Run RAG pipeline → get AI answer
    4. Save both messages to DB
    5. If escalation needed → create ticket
    6. Return response with all metadata
    """
    if not engine or not detector:
        raise HTTPException(500, "Engine not initialized")
    
    # Step 1: Detect intent and sentiment
    intent_result = detector.detect(request.message)
    intent = intent_result["intent"]
    sentiment = intent_result["sentiment"]
    needs_escalation = intent_result["needs_escalation"]
    priority = intent_result["priority"]
    
    # Step 2: Load conversation history
    history = get_conversation_history(request.conversation_id)
    
    # Step 3: Save user message to DB
    save_message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.message,
        intent=intent,
        sentiment=sentiment
    )
    
    ticket_id = None
    
    # Step 4: Generate response
    if needs_escalation:
        # Bot can't handle this — create a ticket and inform user
        ticket_id = create_ticket(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            issue_summary=request.message,
            intent=intent,
            priority=priority
        )
        answer = (
            f"I understand this is important and I want to make sure you get the best help. "
            f"I've created a support ticket ({ticket_id}) and a human agent will reach out "
            f"within 24 hours (or sooner for urgent issues). "
            f"Is there anything else I can try to help with in the meantime?"
        )
    else:
        # Let the RAG bot answer
        result = engine.answer(request.message, history, intent)
        answer = result["answer"]
    
    # Step 5: Save AI response to DB
    save_message(
        conversation_id=request.conversation_id,
        role="assistant",
        content=answer
    )
    
    return ChatResponse(
        answer=answer,
        intent=intent,
        sentiment=sentiment,
        escalated=needs_escalation,
        ticket_id=ticket_id,
        conversation_id=request.conversation_id
    )


@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str):
    """Returns all conversation sessions for a user."""
    return get_all_conversations(user_id)


@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
    """Returns full message history for a conversation."""
    return get_conversation_history(conversation_id, limit=100)


@app.get("/tickets/{user_id}")
async def get_tickets(user_id: str):
    """Returns all support tickets for a user."""
    return get_user_tickets(user_id)


@app.post("/rebuild-kb")
async def rebuild_knowledge_base():
    """Admin endpoint to rebuild the knowledge base after updating FAQs."""
    engine.rebuild_knowledge_base()
    return {"message": "Knowledge base rebuilt successfully."}