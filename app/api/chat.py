"""
Chat/query API route: the actual "ask the RAG chatbot a question" endpoint.
"""

import logging

from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.query_service import answer_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Run one RAG turn: retrieve relevant chunks for request.query, generate
    a grounded answer, and return both the answer and the source chunks
    it was based on.

    Any failure inside answer_query() (retrieval failure, LLM failure)
    raises our custom exceptions, which app/api/errors.py translates into
    the right HTTP status - this route stays a thin pass-through.
    """
    result = answer_query(
        query=request.query,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    sources = [
        SourceChunk(
            text=chunk["text"],
            source=chunk["metadata"].get("source", "unknown"),
            page=chunk["metadata"].get("page"),
            chunk_index=chunk["metadata"].get("chunk_index"),
            distance=chunk["distance"],
        )
        for chunk in result["sources"]
    ]

    return ChatResponse(answer=result["answer"], sources=sources, query=result["query"])
