"""
Embedding generation using a real Sentence-Transformer model (SBERT).

WHY all-MiniLM-L6-v2 SPECIFICALLY:
- It's a genuine sentence-transformer: a base Transformer FURTHER fine-tuned
  via Siamese-network contrastive training so that cosine similarity between
  its pooled outputs actually reflects semantic similarity (see Phase 4.8 -
  this is the step a raw pretrained Transformer is missing).
- Small (384 dimensions, ~80MB) and fast enough to run on CPU - important
  for a portfolio project without a GPU budget.
- Free and runs locally - no per-embedding API cost, unlike OpenAI's
  embedding API (a valid alternative - see the commented-out code below).

NOTE: this file requires network access to Hugging Face Hub the FIRST time
it runs (to download model weights, cached locally afterward). If you swap
to a fully offline environment later, download the model once and point
`SentenceTransformer(local_path)` at the cached folder instead.
"""

import logging
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

# Loaded ONCE at import time - loading a Transformer model is slow (disk I/O +
# weight initialization), so we never want to reload it per-request.
_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model as a singleton."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts (documents/chunks) into dense vectors.

    Batching (passing a LIST, not one string at a time) matters for real
    performance: the model processes the whole batch in one forward pass,
    which is far faster than calling this function in a per-item loop.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    # normalize_embeddings=True: scales each vector to unit length. Since we
    # compare with cosine similarity, and cosine similarity of two unit
    # vectors reduces to a simple dot product, this makes later similarity
    # search (Phase 5) both correct AND faster.
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single user query.

    A separate function (rather than always calling embed_texts([query]))
    exists purely for READABILITY at call sites - `embed_query(user_input)`
    makes the code's intent obvious, even though the implementation is
    the same batch call underneath.
    """
    return embed_texts([query])[0]


# --- Alternative: OpenAI's embedding API (commented out, for reference) ---
# from openai import OpenAI
# client = OpenAI(api_key=settings.openai_api_key)
#
# def embed_texts_openai(texts: list[str]) -> list[list[float]]:
#     response = client.embeddings.create(model="text-embedding-3-small", input=texts)
#     return [item.embedding for item in response.data]
#
# TRADEOFF: OpenAI's embeddings are generally higher quality and require no
# local model download/compute, but cost money per call and add network
# latency + an external dependency for every single embedding operation
# (including embedding every chunk during document ingestion in Phase 6).
