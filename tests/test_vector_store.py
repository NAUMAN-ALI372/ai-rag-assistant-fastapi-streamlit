"""
Tests for vector_store.py.

STRATEGY: mock the embedding functions (Phase 4 - genuinely needs network/a
model), but use a REAL, temporary ChromaDB instance (needs no network at
all, as we proved in Phase 5). This gives us a realistic integration test
for the vector DB layer without any external dependency.
"""

import shutil
from unittest.mock import patch
import pytest


@pytest.fixture
def temp_vector_store(tmp_path, monkeypatch):
    """Point the vector store at a fresh temporary directory for each test,
    so tests never interfere with each other or with real project data."""
    import app.retrieval.vector_store as vs_module
    monkeypatch.setattr("app.core.config.settings.chroma_persist_dir", str(tmp_path / "chroma_test"))
    vs_module._client = None
    vs_module._collection = None
    yield vs_module
    shutil.rmtree(tmp_path, ignore_errors=True)


@patch("app.retrieval.vector_store.embed_texts")
@patch("app.retrieval.vector_store.embed_query")
def test_add_and_search_returns_correct_shape(mock_embed_query, mock_embed_texts, temp_vector_store):
    # Fake embeddings: deliberately make "password chunk" closer to the query
    # than "cake chunk", so we can assert the RANKING is correct.
    mock_embed_texts.return_value = [
        [1.0, 0.0, 0.0],  # password chunk
        [0.0, 1.0, 0.0],  # cake chunk
    ]
    mock_embed_query.return_value = [0.9, 0.1, 0.0]  # query vector, closer to password chunk

    temp_vector_store.add_chunks(
        chunk_ids=["c1", "c2"],
        texts=["Reset your password here.", "Bake a cake at 180C."],
        metadatas=[{"source": "help.pdf"}, {"source": "recipes.pdf"}],
    )

    results = temp_vector_store.search("How do I change my password?", top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "Reset your password here."  # correctly ranked first
    assert results[0]["metadata"]["source"] == "help.pdf"
    assert "distance" in results[0]


@patch("app.retrieval.vector_store.embed_texts")
@patch("app.retrieval.vector_store.embed_query")
def test_metadata_filtering_excludes_nonmatching_chunks(mock_embed_query, mock_embed_texts, temp_vector_store):
    mock_embed_texts.return_value = [[1.0, 0.0], [0.9, 0.1]]
    mock_embed_query.return_value = [1.0, 0.0]

    temp_vector_store.add_chunks(
        chunk_ids=["c1", "c2"],
        texts=["chunk from help doc", "chunk from recipes doc"],
        metadatas=[{"source": "help.pdf"}, {"source": "recipes.pdf"}],
    )

    results = temp_vector_store.search("test query", top_k=5, where={"source": "help.pdf"})

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "help.pdf"
