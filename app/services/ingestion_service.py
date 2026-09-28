"""
The ingestion pipeline: ties document_loader.py + chunking.py + vector_store.py
(Phases 4-6) into one function that takes a file path and ends with
searchable chunks in the vector database.
"""

import logging
from pathlib import Path
from app.retrieval.document_loader import load_document
from app.retrieval.chunking import recursive_character_split
from app.retrieval.vector_store import add_chunks
from app.core.config import settings

logger = logging.getLogger(__name__)


def ingest_document(file_path: str) -> int:
    """Load, chunk, embed, and store one document. Returns the number of
    chunks created, so callers (e.g. the upload API endpoint, Phase 15) can
    report back to the user.
    """
    filename = Path(file_path).name
    pages = load_document(file_path)  # [(page_num, text), ...] - Phase 6.1

    chunk_ids, chunk_texts, chunk_metadatas = [], [], []
    for page_num, page_text in pages:
        page_chunks = recursive_character_split(
            page_text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        for i, chunk_text in enumerate(page_chunks):
            if not chunk_text.strip():
                continue  # skip empty chunks (can happen at page boundaries)
            chunk_ids.append(f"{filename}_p{page_num}_c{i}")
            chunk_texts.append(chunk_text)
            # THIS metadata is exactly what Phase 0's "citation" requirement
            # depends on - without it, we could never tell the user WHERE an
            # answer came from.
            chunk_metadatas.append({"source": filename, "page": page_num, "chunk_index": i})

    if not chunk_texts:
        logger.warning(f"No chunks produced for {filename} - document may be empty after cleaning.")
        return 0

    add_chunks(chunk_ids, chunk_texts, chunk_metadatas)
    logger.info(f"Ingested {filename}: {len(chunk_texts)} chunks across {len(pages)} page(s).")
    return len(chunk_texts)
