"""VirtuOS: the draft corpus and a forced-choice benchmark runner."""

from __future__ import annotations

from .corpus import Corpus, Entry
from .providers import get_provider, parse_letter, retry_call, RetryPolicy
from .runner import evaluate, run, wilson
from .scenarios import Item, build_items
from .validate import Validation, validate

__version__ = "0.1.1"
__all__ = [
    "Corpus",
    "Entry",
    "Item",
    "RetryPolicy",
    "Validation",
    "build_items",
    "evaluate",
    "get_provider",
    "parse_letter",
    "retry_call",
    "run",
    "validate",
    "wilson",
    "__version__",
]
