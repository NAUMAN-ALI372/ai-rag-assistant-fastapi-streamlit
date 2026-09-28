"""
Exception handlers: map our custom exceptions (app/core/exceptions.py) to
proper HTTP responses.

WHY THIS MATTERS:
Without these handlers, any raised ChatbotBaseException subclass would
bubble up as a generic, unhelpful 500 Internal Server Error. Registering
handlers here means API consumers (the Streamlit frontend, curl, a future
mobile client) get a meaningful status code and a clean JSON error body
instead.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    ChatbotBaseException,
    DocumentNotFoundError,
    RetrievalError,
    LLMGenerationError,
    EmptyDocumentError,
)

logger = logging.getLogger(__name__)


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


def register_exception_handlers(app: FastAPI) -> None:
    """Call once from main.py, right after creating the FastAPI app."""

    @app.exception_handler(DocumentNotFoundError)
    async def handle_not_found(request: Request, exc: DocumentNotFoundError):
        return _error_response(404, str(exc))

    @app.exception_handler(EmptyDocumentError)
    async def handle_empty_document(request: Request, exc: EmptyDocumentError):
        # 422 Unprocessable Entity: the request was well-formed (a real file
        # was uploaded) but its content can't be processed as intended.
        return _error_response(422, str(exc))

    @app.exception_handler(RetrievalError)
    async def handle_retrieval_error(request: Request, exc: RetrievalError):
        logger.error(f"Retrieval error: {exc}")
        # 502 Bad Gateway: our own vector-store dependency failed, not the caller's fault.
        return _error_response(502, "The document search backend failed. Please try again.")

    @app.exception_handler(LLMGenerationError)
    async def handle_llm_error(request: Request, exc: LLMGenerationError):
        logger.error(f"LLM generation error: {exc}")
        return _error_response(502, "The language model failed to generate a response. Please try again.")

    @app.exception_handler(ChatbotBaseException)
    async def handle_generic_app_error(request: Request, exc: ChatbotBaseException):
        # Catch-all for any custom exception not specifically handled above -
        # still better than letting it fall through to FastAPI's raw 500 handler.
        logger.error(f"Unhandled app exception: {exc}")
        return _error_response(500, str(exc))
