# 🧠 GK Assistant — AI-Powered General Knowledge Chatbot

A full-stack AI chatbot that answers general knowledge questions using **RAG (Retrieval-Augmented Generation)**, **LLaMA 3.3 via Groq**, and a **ChromaDB** vector store. Built with FastAPI, LangChain, and a clean vanilla JS frontend.

---

## 🚀 Features

- 🔍 **RAG Pipeline** — Retrieves relevant knowledge from a local vector database before answering
- 🤖 **LLaMA 3.3 (Groq)** — Fast, free LLM inference for both answers and intent detection
- 🎯 **Intent & Sentiment Detection** — Automatically classifies every message (science, history, geography, etc.)
- 🗂️ **Conversation History** — All chats stored in SQLite and reloadable from the sidebar
- 🎫 **Ticket Escalation** — Creates support tickets for frustrated or complex queries
- 💬 **Clean Chat UI** — Sidebar with chat history, typing indicators, quick-action buttons
- ⚡ **Persistent Vector Store** — ChromaDB built once, reloaded instantly on every restart

---

## 🗂️ Project Structure

```
GK_Assistant/
│
├── main.py                  # FastAPI app — all API endpoints
├── chatbot_engine.py        # RAG engine — builds KB, runs retrieval + LLM
├── intent_detector.py       # Classifies intent and sentiment using LLM
├── database.py              # SQLite — conversations, messages, tickets
├── config.py                # All settings (model names, paths, constants)
│
├── knowledge_base/
│   ├── faq.json             # General knowledge Q&A pairs
│   └── *.txt                # Optional topic text files (science.txt, etc.)
│
├── frontend/
│   └── index.html           # Chat UI (vanilla HTML/CSS/JS)
│
├── .env                    
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gk-assistant.git
cd gk-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Activate it:
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: [https://console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
uvicorn main:app --reload --port 8000
```

Open your browser at: **http://localhost:8000**

> ⏳ The first run will take ~30 seconds to build the ChromaDB vector store. Every run after that is instant.

---

## 🧠 How It Works

```
User Message
     │
     ▼
Intent Detector (LLaMA 3.3)
     │  → intent: "science", sentiment: "neutral"
     ▼
ChromaDB Similarity Search
     │  → Top 4 relevant chunks from knowledge base
     ▼
LLM (LLaMA 3.3 via Groq)
     │  → System prompt + context + history
     ▼
Answer returned to frontend
```

---

## 📚 Adding More Knowledge

### Option A — Add Q&As to `faq.json`

```json
{
  "question": "Your question here?",
  "answer": "Your detailed answer here.",
  "category": "science"
}
```

Available categories: `science`, `history`, `geography`, `mathematics`, `technology`, `arts_culture`, `sports`, `general_query`

### Option B — Add `.txt` files to `knowledge_base/`

Create any `.txt` file (e.g. `knowledge_base/space.txt`) and write freely. It will be auto-ingested on the next rebuild.

### After adding content — rebuild the vector store:

```bash
# Delete old ChromaDB
# Windows:
rmdir /s /q chroma_db
# Mac/Linux:
rm -rf chroma_db

# Restart the server
uvicorn main:app --reload --port 8000
```

---

## 🔧 Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model for answers |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Text chunk size for splitting |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Number of chunks retrieved per query |
| `CHROMA_DIR` | `chroma_db` | Vector store directory |
| `DB_PATH` | `support.db` | SQLite database path |

---

## 📦 Requirements

```
fastapi
uvicorn
langchain
langchain-community
langchain-groq
langchain-text-splitters
langchain-core
chromadb
sentence-transformers
python-dotenv
pydantic
```

Generate with:
```bash
pip freeze > requirements.txt
```

---

## 🔒 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key |

---

## 🛠️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the chat UI |
| `POST` | `/conversation` | Start a new conversation |
| `POST` | `/chat` | Send a message, get AI response |
| `GET` | `/conversations/{user_id}` | Get all sessions for a user |
| `GET` | `/history/{conversation_id}` | Get full message history |
| `GET` | `/tickets/{user_id}` | Get all support tickets |
| `POST` | `/rebuild-kb` | Rebuild the knowledge base (admin) |

---

## 🙌 Credits

- [Groq](https://groq.com) — Ultra-fast LLM inference
- [LangChain](https://langchain.com) — RAG framework
- [ChromaDB](https://trychroma.com) — Vector database
- [HuggingFace](https://huggingface.co) — Embedding models

---

## 📃 License

MIT License — free to use, modify, and distribute.
