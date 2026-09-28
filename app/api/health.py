"""
Health check route.

The Streamlit frontend calls this on startup to confirm the FastAPI
backend is reachable, and to display basic stats (how many chunks are
currently stored) without needing a dedicated "stats" endpoint.
"""

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse
from app.retrieval.vector_store import get_collection

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    collection = get_collection()
    return HealthResponse(
        status="ok",
        collection_name=settings.collection_name,
        chunk_count=collection.count(),
        embedding_model=settings.embedding_model_name,
        llm_model=settings.llm_model,
    )
