"""
Pydantic schemas: the request/response "shapes" for the API layer.

WHY SEPARATE FROM app/api/?
Keeping schemas here (rather than defined inline in the route files) means
the same models can be reused by the API layer, the Streamlit frontend, and
tests, without any of them needing to import from app/api/ directly.
"""

from pydantic import BaseModel, Field


# --- Ingestion ---------------------------------------------------------

class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str = "success"


class BulkIngestResponse(BaseModel):
    results: list[IngestResponse]
    total_files: int
    total_chunks: int


# --- Chat / Query ---------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="How many chunks to retrieve. Falls back to settings.top_k if omitted.",
    )
    source_filter: str | None = Field(
        default=None,
        description="Optional exact filename (e.g. 'employee_handbook.txt') to restrict retrieval to.",
    )


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    chunk_index: int | None = None
    distance: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    query: str


# --- Misc -----------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    collection_name: str
    chunk_count: int
    embedding_model: str
    llm_model: str
