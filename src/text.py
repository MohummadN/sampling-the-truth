"""Sentence splitting. One definition, used by data stats and by factuality."""
from __future__ import annotations

import nltk
from nltk.tokenize import sent_tokenize

MIN_CHARS = 15          # drops fragments like "." or "She." — fixed once, all arms


def ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


def split_sentences(text: str, min_chars: int = MIN_CHARS) -> list[str]:
    """Split into sentences, dropping fragments.

    Never split on "." — it shreds 1815., Ph.D., U.S. and every abbreviation.
    """
    ensure_punkt()
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) >= min_chars]