"""
Ingestion API routes.

Two entry points:
  - POST /api/ingest/upload         upload one file, ingest it immediately
  - POST /api/ingest/sample-docs    (re-)ingest every file already sitting
                                     in data/sample_docs - handy for demos
                                     and for populating the vector store on
                                     first run without a manual upload step
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import IngestResponse, BulkIngestResponse
from app.services.ingestion_service import ingest_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
SAMPLE_DOCS_DIR = Path("data/sample_docs")
UPLOAD_DIR = Path("data/uploads")


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(file: UploadFile = File(...)) -> IngestResponse:
    """
    Accept a file upload, save it to disk, then run it through the existing
    ingest_document() pipeline (load -> chunk -> embed -> store).

    We save to disk first (rather than passing bytes directly into
    ingest_document) because document_loader.py's loaders (PdfReader,
    python-docx) all expect a file PATH, and keeping that interface
    unchanged means this route doesn't need to touch retrieval code at all.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = UPLOAD_DIR / file.filename

    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    logger.info(f"Saved upload to {dest_path}")

    # Any DocumentLoadError / EmptyDocumentError raised inside ingest_document
    # propagates up and is handled by the exception handlers in app/api/errors.py -
    # no need to catch it here.
    n_chunks = ingest_document(str(dest_path))

    return IngestResponse(filename=file.filename, chunks_created=n_chunks)


@router.post("/sample-docs", response_model=BulkIngestResponse)
async def ingest_sample_docs() -> BulkIngestResponse:
    """
    Ingest every supported file already present in data/sample_docs.
    Useful for a one-click "seed the knowledge base" action from the
    Streamlit frontend, instead of uploading the same 3 demo files by hand.
    """
    if not SAMPLE_DOCS_DIR.exists():
        raise HTTPException(status_code=404, detail=f"{SAMPLE_DOCS_DIR} does not exist.")

    files = sorted(
        p for p in SAMPLE_DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise HTTPException(status_code=404, detail=f"No supported files found in {SAMPLE_DOCS_DIR}.")

    results: list[IngestResponse] = []
    for path in files:
        try:
            n_chunks = ingest_document(str(path))
            results.append(IngestResponse(filename=path.name, chunks_created=n_chunks))
        except Exception as e:
            # Don't let one bad file (e.g. a corrupt PDF) abort the whole
            # batch - record it as zero chunks and keep going.
            logger.error(f"Failed to ingest {path.name}: {e}")
            results.append(IngestResponse(filename=path.name, chunks_created=0, status=f"failed: {e}"))

    return BulkIngestResponse(
        results=results,
        total_files=len(results),
        total_chunks=sum(r.chunks_created for r in results),
    )
