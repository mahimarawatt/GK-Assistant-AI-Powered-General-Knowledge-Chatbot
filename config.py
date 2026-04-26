# config.py
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 3
CHROMA_DIR = "./chroma_store"
DB_PATH = "./chatbot.db"
COMPANY_NAME = "GKnowledge"   # Change to your company name

# Intent categories the bot can detect

INTENTS = [
    "science", "history", "geography", "mathematics",
    "technology", "sports", "arts_culture", "general_query"
]

# Confidence threshold — below this, escalate to human
ESCALATION_THRESHOLD = 0.4