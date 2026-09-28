"""
The query/chat pipeline: ties vector_store.search() + rag_prompt.py +
openai_client.py (Phases 4-6 + the new LLM layer) into one function that
takes a user question and returns a grounded answer with sources.

Mirrors the shape of app/services/ingestion_service.py deliberately: the
API layer should never need to know about retrieval, prompt formatting, or
the LLM client individually - it just calls one service function.
"""

import logging

from app.core.config import settings
from app.retrieval.vector_store import search

from app.prompts.rag_prompt import build_rag_messages
from app.llm.openai_client import generate_chat_completion

logger = logging.getLogger(__name__)


def answer_query(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> dict:
    """
    Run one full RAG turn: retrieve relevant chunks, build a grounded
    prompt, call the LLM, and return both the answer and the chunks it was
    based on (so the API/frontend can show citations).

    Returns a dict: {"answer": str, "sources": list[dict], "query": str}
    where each item in "sources" has the same shape vector_store.search()
    returns ({"text", "metadata", "distance"}).
    """
    where = {"source": source_filter} if source_filter else None

    chunks = search(query, top_k=top_k or settings.top_k, where=where)
    logger.info(f"Retrieved {len(chunks)} chunk(s) for query: {query!r}")

    messages = build_rag_messages(query, chunks)
    answer = generate_chat_completion(messages)

    return {
        "answer": answer,
        "sources": chunks,
        "query": query,
    }
