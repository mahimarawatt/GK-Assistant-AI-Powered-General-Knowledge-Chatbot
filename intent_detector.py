# Detects WHAT the user wants (intent) and HOW they feel (sentiment).
# Uses the LLM for this — it's fast and free with Groq.
#
# Why intent detection?
# - Route to the right FAQ category for better RAG retrieval
# - Detect frustrated users early → proactive escalation
# - Log intent distribution → understand what users need most

import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages  import HumanMessage, SystemMessage
import config


class IntentDetector:
    
    SYSTEM_PROMPT = """You are an intent and sentiment classifier for a customer support chatbot.
    
Analyze the user message and return ONLY a JSON object (no other text) with:
- "intent": one of {intents}
- "sentiment": one of ["positive", "neutral", "negative", "frustrated"]  
- "confidence": float between 0.0 and 1.0
- "needs_escalation": boolean (true if user is very frustrated, issue is urgent/complex, or explicitly asks for human)
- "priority": one of ["low", "medium", "high"] based on urgency

Example output:
{{"intent": "billing", "sentiment": "frustrated", "confidence": 0.92, "needs_escalation": true, "priority": "high"}}""".format(
        intents=config.INTENTS
    )
    
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=config.GROQ_MODEL,
            temperature=0,      # Deterministic for classification
            max_tokens=150      # Classification output is tiny
        )
    
    def detect(self, user_message: str) -> dict:
        """
        Classifies user message intent and sentiment.
        
        Returns dict:
        {
            "intent": str,
            "sentiment": str,
            "confidence": float,
            "needs_escalation": bool,
            "priority": str
        }
        """
        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"User message: {user_message}")
            ])
            
            # Parse JSON from LLM response
            result = json.loads(response.content.strip())
            
            # Escalate if confidence is too low regardless of intent
            if result.get("confidence", 1.0) < config.ESCALATION_THRESHOLD:
                result["needs_escalation"] = True
            
            return result
        
        except (json.JSONDecodeError, Exception) as e:
            # Fallback if parsing fails — safe defaults
            print(f"Intent detection error: {e}")
            return {
                "intent": "general_query",
                "sentiment": "neutral",
                "confidence": 0.5,
                "needs_escalation": False,
                "priority": "medium"
            }