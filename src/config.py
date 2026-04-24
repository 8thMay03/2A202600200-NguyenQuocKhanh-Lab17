"""Configuration module for the Multi-Memory Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EPISODIC_LOG_DIR = Path(os.getenv("EPISODIC_LOG_DIR", str(DATA_DIR / "episodic_logs")))
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", str(DATA_DIR / "chroma_db")))

# Ensure directories exist
EPISODIC_LOG_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Redis ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# --- Context Window ---
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
TRIM_THRESHOLD = float(os.getenv("TRIM_THRESHOLD", "0.8"))  # Start trimming at 80%

# --- Memory Settings ---
BUFFER_MEMORY_MAX_MESSAGES = int(os.getenv("BUFFER_MEMORY_MAX_MESSAGES", "20"))

# --- Priority Levels for Context Eviction ---
PRIORITY_CRITICAL = 0  # System prompt, current query — never evict
PRIORITY_HIGH = 1      # Recent conversation turns (last 3)
PRIORITY_MEDIUM = 2    # Retrieved memories, semantic context
PRIORITY_LOW = 3       # Older conversation history, episodic logs
