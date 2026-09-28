"""
Central configuration for the whole application.

WHY THIS FILE EXISTS:
Every other module in this app that needs a setting (API key, model name,
chunk size, etc.) imports `settings` from HERE, instead of calling
os.getenv() directly. This gives us three things `os.getenv()` alone
doesn't:

1. VALIDATION — if a required setting is missing or the wrong type,
   the app fails loudly at STARTUP, not silently mid-request.
2. ONE SOURCE OF TRUTH — every setting is documented and typed in one place.
3. EASY TESTING — tests can override settings without touching real .env files.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Environment ---
    environment: str = "development"  # "development" | "production"

    # --- API server (Phase 15) ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- LLM provider ---
    openai_api_key: str  # required - app will refuse to start without it
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2  # low temperature = more deterministic, less "creative" -> good for a factual Q&A bot

    # --- Embeddings ---
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Vector DB ---
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "knowledge_base"

    # --- Chunking (tuned properly in Phase 6) ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 4

    # --- SQL database (chat history, metadata - Phase 17) ---
    database_url: str = "sqlite:///./data/chatbot.db"

    # --- Logging ---
    log_level: str = "INFO"

    # This tells pydantic-settings: "read values from a file named .env,
    # and match environment variable names case-insensitively"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


# A single, shared instance every other module will import.
# Created once, at import time - if OPENAI_API_KEY is missing, the app
# crashes immediately with a clear pydantic validation error, instead of
# failing confusingly later when we actually try to call the LLM.
settings = Settings()
