"""
Chunking strategies, implemented FROM SCRATCH so we understand exactly what
LangChain's text splitters (Phase 12) actually do internally before we let
a library hide it.
"""

import re
import numpy as np
from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------------------
# STRATEGY 1: Naive character splitting
# ---------------------------------------------------------------------
def character_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Cut the text into fixed-size windows of characters, sliding forward
    by (chunk_size - chunk_overlap) each time. Simplest possible approach -
    completely ignorant of word/sentence/paragraph boundaries.
    """
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


# ---------------------------------------------------------------------
# STRATEGY 2: Recursive character splitting (the real-world default)
# ---------------------------------------------------------------------
def recursive_character_split(text: str, chunk_size: int, chunk_overlap: int,
                                separators: list[str] = None) -> list[str]:
    """Try splitting on the FIRST (most 'natural') separator in the list.
    If a resulting piece is still too big, recursively split THAT piece
    using the next separator down the list. Only falls back to raw
    character splitting as an absolute last resort.

    This is exactly the strategy LangChain's RecursiveCharacterTextSplitter
    uses internally (Phase 12 will show the one-line library equivalent).

    IMPLEMENTATION NOTE: splitting and overlap are deliberately kept as two
    separate passes (_split_no_overlap does the recursive splitting; overlap
    is applied ONCE at the very end). An earlier version of this function
    applied overlap inside the recursive calls too, which caused duplicated
    text when a piece needed multiple levels of recursion - a real bug,
    caught by literally reading this function's own output (see Section 6.5).
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    chunks = _split_no_overlap(text, chunk_size, separators)

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped
    return chunks


def _split_no_overlap(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Pure recursive splitting, no overlap applied - see note above."""
    separator = separators[0]
    remaining_separators = separators[1:]

    pieces = list(text) if separator == "" else text.split(separator)

    chunks = []
    current_chunk = ""
    for piece in pieces:
        piece_with_sep = piece + (separator if separator else "")
        if len(current_chunk) + len(piece_with_sep) <= chunk_size:
            current_chunk += piece_with_sep
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(piece_with_sep) > chunk_size and remaining_separators:
                chunks.extend(_split_no_overlap(piece_with_sep, chunk_size, remaining_separators))
                current_chunk = ""
            else:
                current_chunk = piece_with_sep
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ---------------------------------------------------------------------
# STRATEGY 3: Sentence-based splitting
# ---------------------------------------------------------------------
def sentence_based_split(text: str, max_chunk_size: int) -> list[str]:
    """Never cut mid-sentence. Greedily pack whole sentences into a chunk
    until adding the next sentence would exceed max_chunk_size."""
    sentences = sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------
# STRATEGY 4: Token-based splitting
# ---------------------------------------------------------------------
def approximate_token_count(text: str) -> int:
    """A rough stand-in for a real tokenizer (see note below): ~4 characters
    per token is a commonly-cited approximation for English text with
    OpenAI's tokenizers. NOT precise - see the real tiktoken code below."""
    return max(1, len(text) // 4)


def token_based_split(text: str, chunk_size_tokens: int, chunk_overlap_tokens: int) -> list[str]:
    """Split by approximate TOKEN count rather than character count.
    WHY THIS MATTERS: the LLM's context window (Phase 10) is measured in
    tokens, not characters. A chunk_size defined in characters can silently
    produce wildly different token counts depending on the text (dense
    technical text tokenizes differently than plain English), risking
    chunks that blow past what fits in the prompt.
    """
    chunk_size_chars = chunk_size_tokens * 4       # reverse the approximation above
    chunk_overlap_chars = chunk_overlap_tokens * 4
    return character_split(text, chunk_size_chars, chunk_overlap_chars)


# --- REAL version for your actual machine (network access to fetch tiktoken's
# encoding file works normally outside this sandbox): ---
#
# import tiktoken
# encoding = tiktoken.encoding_for_model("gpt-4o-mini")
#
# def token_based_split_real(text, chunk_size_tokens, chunk_overlap_tokens):
#     tokens = encoding.encode(text)
#     chunks = []
#     step = chunk_size_tokens - chunk_overlap_tokens
#     for start in range(0, len(tokens), step):
#         chunk_tokens = tokens[start:start + chunk_size_tokens]
#         chunks.append(encoding.decode(chunk_tokens))
#     return chunks


# ---------------------------------------------------------------------
# STRATEGY 5: Semantic chunking (simplified)
# ---------------------------------------------------------------------
def semantic_split(text: str, embed_fn, similarity_threshold: float = 0.5) -> list[str]:
    """Group consecutive sentences together UNTIL the topic seems to shift,
    detected as a drop in embedding similarity between neighboring
    sentences. embed_fn: a function taking a string and returning a vector
    (we'll pass in our real transformer-based embedder for the real test below).

    This is a simplified illustration of semantic chunking's core idea, not
    a production-grade implementation (real systems often use more robust
    breakpoint-detection statistics, e.g. percentile-based thresholds).
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    vectors = [embed_fn(s) for s in sentences]

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine(vectors[i - 1], vectors[i])
        if sim < similarity_threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks
