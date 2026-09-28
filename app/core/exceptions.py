"""
Custom exceptions for the chatbot application.

WHY CUSTOM EXCEPTIONS INSTEAD OF JUST `raise Exception("something broke")`?

1. SPECIFICITY - catching `RetrievalError` vs catching bare `Exception` lets
   calling code react differently to different failures (e.g., retry a
   timed-out LLM call, but don't retry a "document not found" error).
2. CLEAR API ERROR RESPONSES - in Phase 15 (FastAPI), we'll map each of
   these to a specific HTTP status code, so API consumers get meaningful
   errors instead of a generic 500.
3. READABILITY - `raise DocumentNotFoundError(doc_id)` documents intent
   far better than a raw string.
"""


class ChatbotBaseException(Exception):
    """Base class for every custom exception in this app.
    Lets calling code do `except ChatbotBaseException` to catch ANY
    app-specific error, while still letting truly unexpected Python
    errors (bugs) propagate normally instead of being silently swallowed.
    """
    pass


class DocumentNotFoundError(ChatbotBaseException):
    """Raised when a requested document ID doesn't exist in the vector DB."""
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        super().__init__(f"Document with id '{doc_id}' was not found.")


class RetrievalError(ChatbotBaseException):
    """Raised when the vector database query itself fails (not just 'no results')."""
    pass


class LLMGenerationError(ChatbotBaseException):
    """Raised when the LLM API call fails (timeout, rate limit, invalid response, etc.)."""
    pass


class EmptyDocumentError(ChatbotBaseException):
    """Raised when an uploaded document contains no extractable text."""
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"Document '{filename}' contains no extractable text.")
