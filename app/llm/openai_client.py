"""
OpenAI chat completion wrapper.

WHY WRAP IT INSTEAD OF CALLING THE OPENAI SDK DIRECTLY EVERYWHERE?
Same reasoning as embeddings.py / vector_store.py: every raw OpenAI
exception gets translated into our own LLMGenerationError here, ONCE, so
calling code (app/services/query_service.py, app/api/chat.py) only ever
needs to handle one exception type instead of importing and catching
openai.APIError, openai.RateLimitError, openai.APITimeoutError, etc.
individually throughout the codebase.
"""

import logging

from openai import OpenAI, OpenAIError

from app.core.config import settings
from app.core.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)

# Loaded once, like the embedding model in app/retrieval/embeddings.py -
# the OpenAI client itself is cheap to construct, but centralizing it here
# means the API key and any future client-level config (timeouts, retries)
# live in exactly one place.
_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.openai_api_key, base_url="https://api.groq.com/openai/v1"
        )
    return _client


def generate_chat_completion(messages: list[dict]) -> str:
    """
    Send a fully-formed `messages` list (see app.prompts.rag_prompt) to the
    configured chat model and return the assistant's reply text.

    Raises LLMGenerationError on any failure (auth, rate limit, timeout,
    malformed response) so callers don't need to know OpenAI's specific
    exception hierarchy.
    """
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
        )
    except OpenAIError as e:
        logger.error(f"OpenAI chat completion failed: {e}")
        raise LLMGenerationError(f"LLM call failed: {e}") from e

    choice = response.choices[0] if response.choices else None
    if choice is None or choice.message is None or choice.message.content is None:
        # Defensive check: an unexpected/empty response shape shouldn't
        # silently propagate as a confusing AttributeError further up the stack.
        raise LLMGenerationError("LLM returned an empty or malformed response.")

    return choice.message.content.strip()
