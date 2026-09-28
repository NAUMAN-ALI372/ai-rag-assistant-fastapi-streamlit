"""
Application entry point - this is Phase 15: builds the real FastAPI app,
registers routes and exception handlers, and starts the server.

Run with:
    python main.py
or, for auto-reload during development:
    uvicorn main:app --reload
"""

import logging

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.router import api_router
from app.api.errors import register_exception_handlers
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.vector_store import get_collection

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before the "yield") and once at shutdown (after).

    WHY WARM UP THE EMBEDDING MODEL AND VECTOR STORE HERE?
    Both get_embedding_model() and get_collection() are lazy singletons -
    without this, the FIRST real user request would pay the cost of
    downloading/loading the SentenceTransformer model and opening the
    ChromaDB client, making that one unlucky request very slow. Loading
    them during startup instead means every actual request is fast.
    """
    logger.info("Starting chatbot application...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM model configured: {settings.llm_model}")
    logger.info(f"Vector DB persist directory: {settings.chroma_persist_dir}")

    get_embedding_model()
    get_collection()
    logger.info("Startup complete: embedding model and vector store are ready.")

    yield

    logger.info("Shutting down chatbot application.")


app = FastAPI(
    title="RAG Chatbot API",
    description="A retrieval-augmented generation chatbot over internal documents.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: the Streamlit frontend (Phase 16) runs on a different port
# (typically :8501) than this API (:8000), so the browser treats it as a
# cross-origin request. Wide open here since this is a local/demo project;
# in a real production deployment, restrict allow_origins to the actual
# frontend's domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment == "development"),
    )
