"""
End-to-end ingestion test: real document loading, real chunking, mocked
embeddings, real ChromaDB storage and retrieval.
"""

import shutil
from unittest.mock import patch
import numpy as np
import pytest


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    import app.retrieval.vector_store as vs_module
    monkeypatch.setattr("app.core.config.settings.chroma_persist_dir", str(tmp_path / "chroma_test"))
    vs_module._client = None
    vs_module._collection = None
    yield vs_module
    shutil.rmtree(tmp_path, ignore_errors=True)


@patch("app.retrieval.vector_store.embed_texts")
@patch("app.retrieval.vector_store.embed_query")
def test_ingest_real_txt_document_end_to_end(mock_embed_query, mock_embed_texts, temp_store):
    from app.services.ingestion_service import ingest_document

    # Deterministic fake embeddings based on a hash of the text, so identical
    # calls produce identical vectors (needed since we don't control call order/count)
    def fake_embed(texts):
        return [[float((hash(t) % 1000)) / 1000, 0.5, 0.1] for t in texts]

    mock_embed_texts.side_effect = fake_embed
    mock_embed_query.return_value = [0.5, 0.5, 0.1]

    n_chunks = ingest_document("data/sample_docs/employee_handbook.txt")

    assert n_chunks > 0
    results = temp_store.search("remote work policy", top_k=3)
    assert len(results) > 0
    assert all(r["metadata"]["source"] == "employee_handbook.txt" for r in results)
    assert all("chunk_index" in r["metadata"] for r in results)
