"""
Tests for embeddings.py.

WHY MOCK THE MODEL INSTEAD OF LOADING THE REAL ONE?
- Unit tests should be fast and not depend on network access or a 400MB+
  model download - that belongs in a separate, occasional "integration test."
- We're testing OUR code's logic (batching, normalization flag, singleton
  caching), not sentence-transformers' internal correctness (that's SBERT's
  own test suite's job).
"""

from unittest.mock import patch, MagicMock
import numpy as np


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_calls_encode_with_normalization(mock_st_class):
    import app.retrieval.embeddings as emb_module
    emb_module._model = None  # reset singleton between tests

    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_st_class.return_value = mock_instance

    result = emb_module.embed_texts(["hello", "world"])

    mock_instance.encode.assert_called_once_with(["hello", "world"], normalize_embeddings=True)
    assert result == [[0.1, 0.2], [0.3, 0.4]]


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_query_returns_single_vector_not_a_list_of_vectors(mock_st_class):
    import app.retrieval.embeddings as emb_module
    emb_module._model = None

    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([[0.5, 0.6]])
    mock_st_class.return_value = mock_instance

    result = emb_module.embed_query("How do I reset my password?")

    assert result == [0.5, 0.6]  # unwrapped from the batch of size 1


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_model_is_loaded_only_once_singleton_behavior(mock_st_class):
    import app.retrieval.embeddings as emb_module
    emb_module._model = None

    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([[0.1, 0.2]])
    mock_st_class.return_value = mock_instance

    emb_module.embed_query("first call")
    emb_module.embed_query("second call")

    mock_st_class.assert_called_once()  # constructor called ONCE, not twice - proves caching works
