"""
Document loading: turn a file on disk into plain text.

WHY SEPARATE LOADING FROM CHUNKING (Section 6.4)?
Loading is format-specific (PDF parsing != DOCX parsing != plain text).
Chunking is format-AGNOSTIC (once we have plain text, chunking logic is
identical regardless of source format). Keeping them separate means adding
a new format (e.g. CSV, HTML) later touches only this file.
"""

import logging
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument
from app.core.exceptions import EmptyDocumentError

logger = logging.getLogger(__name__)


def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str) -> str:
    """Extract text from every page and join with a clear page-boundary marker.

    WHY keep track of pages at all here? PDF text extraction is inherently
    per-page; we preserve that structure now so we CAN attach page-number
    metadata during chunking (Section 6.7) - losing this here would make
    accurate page citations impossible later.
    """
    reader = PdfReader(path)
    pages_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_text.append((page_num, text))
    return pages_text  # NOTE: returns list of (page_num, text) tuples, not one string - see loader dispatch below


def load_docx(path: str) -> str:
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_document(path: str) -> list[tuple[int, str]]:
    """Unified entry point. Always returns a list of (page_number, text) tuples
    for consistency - .txt/.docx are treated as a single 'page 1', so every
    downstream consumer (chunking) has one consistent shape to work with,
    regardless of source format.
    """
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        text = load_txt(path)
        result = [(1, text)]
    elif ext == ".pdf":
        result = load_pdf(path)
    elif ext == ".docx":
        text = load_docx(path)
        result = [(1, text)]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    full_text = " ".join(text for _, text in result).strip()
    if not full_text:
        raise EmptyDocumentError(filename=Path(path).name)

    logger.info(f"Loaded {path}: {len(result)} page(s), {len(full_text)} characters.")
    return result
