"""
The RAG prompt template.

WHY THIS LIVES IN ITS OWN FILE:
Prompt wording is the single most iterated-on part of a RAG system in
practice - keeping it isolated from app/llm/ (the API-calling mechanics)
and app/services/ (the orchestration logic) means you can tune wording,
add few-shot examples, or A/B test instructions without touching any other
layer.

DESIGN GOALS baked into this prompt:
1. Answer ONLY from the provided context - the #1 requirement for a
   trustworthy internal-docs chatbot is that it doesn't make things up.
2. Say "I don't know" honestly when the context doesn't contain the answer,
   rather than falling back on the model's general world knowledge.
3. Cite which source document(s) the answer came from, so a user can verify.
"""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context excerpts from internal company documents.

Rules you must follow:
1. Answer using only information found in the context below. Do not use any \
outside knowledge, even if you know the answer.
2. If the context does not contain enough information to answer the \
question, say clearly: "I don't have enough information in the provided \
documents to answer that." Do not guess or make anything up.
3. When you do answer, mention which source document(s) the information \
came from (the [Source: ...] tags in the context), so the user can verify it.
4. Be concise and directly answer the question - don't pad your answer with \
unnecessary preamble.
"""


def _format_context(chunks: list[dict]) -> str:
    """
    Turn retrieved chunks (the list-of-dicts shape returned by
    app.retrieval.vector_store.search: {"text", "metadata", "distance"})
    into a single readable context block, each chunk tagged with its
    source filename (and page, when available) so the LLM can cite it.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown source")
        page = meta.get("page")
        tag = f"[Source: {source}" + (f", page {page}]" if page else "]")
        blocks.append(f"Excerpt {i} {tag}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def build_rag_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the full OpenAI chat `messages` list for a RAG turn.

    Returns a list of {"role": ..., "content": ...} dicts ready to pass
    straight into `client.chat.completions.create(messages=...)`.
    """
    if not chunks:
        # No context retrieved at all: tell the LLM explicitly rather than
        # sending an empty context block, which models tend to respond to
        # by falling back on outside knowledge instead of admitting "I don't know".
        context_block = "(No relevant context was found in the document store.)"
    else:
        context_block = _format_context(chunks)

    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
