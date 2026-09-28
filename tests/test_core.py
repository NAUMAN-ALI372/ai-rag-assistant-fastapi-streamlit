"""
Our very first tests - proving Phase 1's foundation actually works.
Run with: pytest tests/test_core.py -v
"""

import pytest
from app.core.exceptions import (
    ChatbotBaseException,
    DocumentNotFoundError,
    RetrievalError,
)


def test_document_not_found_error_message():
    """A DocumentNotFoundError should embed the doc_id in a readable message."""
    err = DocumentNotFoundError(doc_id="doc_123")
    assert "doc_123" in str(err)
    assert err.doc_id == "doc_123"


def test_custom_exceptions_share_a_common_base():
    """This is WHY we bother with a base class: calling code can catch
    ChatbotBaseException to handle ANY app-specific error uniformly,
    e.g. in a FastAPI exception handler (Phase 15)."""
    with pytest.raises(ChatbotBaseException):
        raise DocumentNotFoundError(doc_id="abc")

    with pytest.raises(ChatbotBaseException):
        raise RetrievalError("vector db timed out")


def test_settings_load_successfully(monkeypatch):
    """Prove that Settings() reads from environment variables correctly,
    without needing a real .env file - this is exactly how we'll mock
    config in every future test."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests")
    from app.core.config import Settings

    test_settings = Settings()
    assert test_settings.openai_api_key == "sk-test-key-for-unit-tests"
    assert test_settings.llm_model == "gpt-4o-mini"  # confirms the default applies
