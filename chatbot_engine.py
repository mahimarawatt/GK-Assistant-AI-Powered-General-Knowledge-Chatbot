# The RAG engine adapted for customer support:
# - Ingests knowledge base (FAQ + policies) once at startup
# - Answers questions using retrieved context + conversation history
# - Generates a special system prompt that makes the LLM act as a support agent

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import config

load_dotenv()


SUPPORT_SYSTEM_PROMPT = """You are a helpful, knowledgeable AI assistant that answers general knowledge questions accurately.

RULES:
1. Answer based on the provided context when available.
2. If the context doesn't cover the question, use your own knowledge to answer — do NOT say you don't know if you actually do.
3. Be clear and educational. Use examples where helpful.
4. For factual topics, be precise. For open-ended topics, be balanced.
5. Always end with "Do you have any other questions?"

CONTEXT FROM KNOWLEDGE BASE:
{context}

CONVERSATION HISTORY:
{history}
""".format(company="", context="{context}", history="{history}")

class SupportChatbotEngine:
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=config.GROQ_MODEL,
            temperature=0.3,
            max_tokens=512
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.vector_store = None
        self._load_or_build_knowledge_base()
    
    
    def _load_or_build_knowledge_base(self):
        """
        Loads ChromaDB from disk if it exists, otherwise builds it from scratch.
        This means the first run takes ~30 seconds; every run after is instant.
        """
        chroma_path = Path(config.CHROMA_DIR)
        
        if chroma_path.exists() and any(chroma_path.iterdir()):
            print("Loading existing knowledge base from ChromaDB...")
            self.vector_store = Chroma(
                persist_directory=config.CHROMA_DIR,
                embedding_function=self.embeddings
            )
            print(f"Loaded {self.vector_store._collection.count()} chunks.")
        else:
            print("Building knowledge base from scratch...")
            self._build_knowledge_base()
    
    
    def _build_knowledge_base(self):
        """
        Reads all files from the knowledge_base/ directory,
        converts them to Documents, chunks them, embeds them, stores in ChromaDB.
        """
        documents = []
        kb_path = Path("knowledge_base")
        
        # Load FAQ JSON
        faq_file = kb_path / "faq.json"
        if faq_file.exists():
            with open(faq_file) as f:
                faqs = json.load(f)
            for item in faqs:
                # Combine Q+A into one document for better retrieval
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                documents.append(Document(
                    page_content=text,
                    metadata={"source": "faq", "category": item.get("category", "general")}
                ))
            print(f"Loaded {len(faqs)} FAQ entries.")
        
        # Load all .txt files
        for txt_file in kb_path.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8")
            documents.append(Document(
                page_content=text,
                metadata={"source": txt_file.name}
            ))
            print(f"Loaded: {txt_file.name}")
        
        if not documents:
            raise RuntimeError("No documents found in knowledge_base/ directory!")
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
        
        # Build ChromaDB vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_DIR
        )
        print("Knowledge base built and saved to disk.")
    
    
    def answer(self, user_message: str, conversation_history: list,
               intent: str = "general_query") -> dict:
        """
        Main method — takes user message + history, returns AI answer.
        
        Process:
        1. Add intent to search query for better retrieval
        2. Retrieve top-K relevant chunks from ChromaDB
        3. Build prompt with context + history
        4. Call Groq LLM
        5. Return answer + sources
        
        Args:
            user_message: Current user question
            conversation_history: List of {"role": ..., "content": ...} dicts
            intent: Detected intent (helps filter retrieval)
        
        Returns:
            {"answer": str, "sources": list, "retrieved_chunks": int}
        """
        # Step 1: Semantic search — find relevant chunks
        # Adding intent to query improves retrieval accuracy
        search_query = f"{intent}: {user_message}"
        retrieved_docs = self.vector_store.similarity_search(
            search_query, 
            k=config.TOP_K_RESULTS
        )
        
        # Step 2: Format retrieved context
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step 3: Format conversation history for the prompt
        history_text = ""
        for msg in conversation_history[-6:]:   # Last 3 exchanges
            role = "Customer" if msg["role"] == "user" else "Agent"
            history_text += f"{role}: {msg['content']}\n"
        
        # Step 4: Build messages for LLM
        system_content = SUPPORT_SYSTEM_PROMPT.format(
            context=context if context else "No relevant information found.",
            history=history_text if history_text else "This is the start of the conversation."
        )
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message)
        ]
        
        # Step 5: Call LLM
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs],
            "retrieved_chunks": len(retrieved_docs)
        }
    
    
    def rebuild_knowledge_base(self):
        """Clears and rebuilds ChromaDB — useful after updating FAQ."""
        import shutil
        if Path(config.CHROMA_DIR).exists():
            shutil.rmtree(config.CHROMA_DIR)
        self._build_knowledge_base()