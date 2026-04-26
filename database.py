# SQLite database for storing all conversations and support tickets.
# Using raw sqlite3 (no ORM) keeps it simple and dependency-free.
# In production you'd use PostgreSQL + SQLAlchemy, but SQLite is perfect here.

import sqlite3
import uuid
import json
from datetime import datetime
import config


def get_connection():
    """
    Returns a SQLite connection.
    check_same_thread=False allows FastAPI's async workers to share the connection.
    """
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # Makes rows behave like dicts
    return conn


def init_db():
    """
    Creates all tables if they don't exist yet.
    Called once at app startup.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: conversations — one row per chat session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            message_count INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0    -- 0 = open, 1 = resolved
        )
    """)
    
    # Table 2: messages — every single message in every conversation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,           -- 'user' or 'assistant'
            content TEXT NOT NULL,
            intent TEXT,                  -- detected intent (user messages only)
            sentiment TEXT,               -- positive / neutral / negative
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    
    # Table 3: support tickets — created when bot can't help
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            issue_summary TEXT NOT NULL,
            intent TEXT,
            priority TEXT DEFAULT 'medium',   -- low / medium / high
            status TEXT DEFAULT 'open',       -- open / in_progress / resolved
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")


# ── Conversation functions ────────────────────────────────────────────────────

def create_conversation(user_id: str) -> str:
    """Creates a new conversation session. Returns the conversation ID."""
    conv_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    conn = get_connection()
    conn.execute(
        "INSERT INTO conversations VALUES (?, ?, ?, ?, 0, 0)",
        (conv_id, user_id, now, now)
    )
    conn.commit()
    conn.close()
    return conv_id


def save_message(conversation_id: str, role: str, content: str,
                 intent: str = None, sentiment: str = None):
    """Saves a single message to the database."""
    msg_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    conn = get_connection()
    
    # Save the message
    conn.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, intent, sentiment, now)
    )
    
    # Update conversation's last_active and message count
    conn.execute("""
        UPDATE conversations 
        SET last_active = ?, message_count = message_count + 1
        WHERE id = ?
    """, (now, conversation_id))
    
    conn.commit()
    conn.close()


def get_conversation_history(conversation_id: str, limit: int = 10) -> list:
    """
    Retrieves the last N messages for a conversation.
    Returns list of dicts with role + content — format LangChain expects.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT role, content FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (conversation_id, limit)).fetchall()
    conn.close()
    
    # Reverse so oldest message comes first
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_all_conversations(user_id: str) -> list:
    """Fetches all conversation sessions for a user (for the sidebar)."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, started_at, message_count, resolved
        FROM conversations WHERE user_id = ?
        ORDER BY last_active DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Ticket functions ──────────────────────────────────────────────────────────

def create_ticket(conversation_id: str, user_id: str, 
                  issue_summary: str, intent: str, priority: str = "medium") -> str:
    """Creates a support ticket when the bot needs to escalate."""
    ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"
    now = datetime.utcnow().isoformat()
    
    conn = get_connection()
    conn.execute(
        "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
        (ticket_id, conversation_id, user_id, issue_summary, intent, priority, now)
    )
    conn.commit()
    conn.close()
    return ticket_id


def get_user_tickets(user_id: str) -> list:
    """Gets all tickets for a user."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM tickets WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]