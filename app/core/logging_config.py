"""
Centralized logging setup.

WHY NOT JUST print()?
- print() output disappears once your terminal closes; logs need to persist to a file.
- print() gives no severity level - you can't easily filter "show me only errors."
- print() gives no timestamp, module name, or line number automatically.
- In production, print() output usually doesn't reach your monitoring system at all.

We configure Python's built-in `logging` module ONCE here, and every other
file just does `logger = logging.getLogger(__name__)` and starts logging.
"""

import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    """Call this once, at application startup (see main.py)."""
    logging.basicConfig(
        level=settings.log_level,  # e.g. "INFO" - hides DEBUG-level noise by default
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),        # prints to console
            logging.FileHandler("app.log", mode="a"),  # also appends to a file we can inspect later
        ],
    )
