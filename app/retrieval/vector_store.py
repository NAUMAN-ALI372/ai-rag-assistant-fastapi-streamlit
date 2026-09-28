"""
Vector store module - wraps ChromaDB so the rest of the app never touches
ChromaDB's API directly.

WHY WRAP IT INSTEAD OF CALLING chromadb DIRECTLY EVERYWHERE?
Same reasoning as embeddings.py in Phase 4: if we ever swap ChromaDB for
FAISS+our-own-metadata-layer, or for Pinecone, only THIS file changes.
Every other file just calls add_chunks() / search().
"""

import logging
import chromadb
from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.retrieval.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)

_client = None
_collection = None


def get_collection():
    """Lazily create a PERSISTENT ChromaDB client (data survives app restarts,
    stored on disk at settings.chroma_persist_dir) and cache it as a singleton."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(name=settings.collection_name)
        logger.info(f"Vector store ready: {_collection.count()} chunks currently stored.")
    return _collection


def add_chunks(chunk_ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    """Embed and store a batch of document chunks.

    We embed texts OURSELVES (via app.retrieval.embeddings, Phase 4) rather
    than letting ChromaDB compute embeddings internally - this keeps us in
    full control of which embedding model is used and guarantees queries
    (Phase 4's embed_query) use the EXACT same model as documents did here.
    """
    collection = get_collection()
    embeddings = embed_texts(texts)
    collection.add(ids=chunk_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info(f"Added {len(chunk_ids)} chunks to the vector store.")


def search(query: str, top_k: int | None = None, where: dict | None = None) -> list[dict]:
    """Embed the query and retrieve the top_k most similar chunks.

    Returns a list of dicts, each with 'text', 'metadata', and 'distance' -
    a clean, library-agnostic shape the rest of the app (Phase 7's RAG
    service) can rely on without knowing anything about ChromaDB's raw
    response format.
    """
    top_k = top_k or settings.top_k
    collection = get_collection()
    try:
        query_vec = embed_query(query)
        results = collection.query(query_embeddings=[query_vec], n_results=top_k, where=where)
    except Exception as e:
        # Wrap the raw ChromaDB/embedding exception in OUR domain exception
        # (Phase 1's exceptions.py) so calling code can react to
        # "retrieval failed" without needing to know ChromaDB's exact error types.
        raise RetrievalError(f"Vector search failed: {e}") from e

    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits
