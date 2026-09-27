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

def is_claim(sentence: str, entity: str) -> bool:
    """Is this sentence a checkable biographical claim?

    The verifier scores every segment split_sentences returns, and three kinds
    of segment are not claims at all. Counting them inflates support rate, and
    unequally: non-claims are 14.0% of GPT-2 beam4's supported sentences
    against 6.2% of Llama-3b's, and reach 88.9% on an arm at the floor, so
    cross-model comparison inherits the bias.

      - the Wikipedia title echoed back ("Douglas Wood (engineer).") — a
        tautology once the entity prefix is prepended, so it is entailed by
        any page at all
      - questions ("How did he become involved?") — assert nothing
      - fragments the 256-token cap cut mid-sentence ("Jose Cardozo is"),
        identified by having no terminal punctuation

    Applied identically to every arm by the labelling frame, the genericity
    control and the dedup probe. NOT applied by score_generation: the reported
    support rate counts every segment, which the paper discloses in Section 3.3.
    """
    s = (sentence or "").strip()
    if not s:
        return False
    if s.endswith("?"):
        return False
    if not s.endswith((".", "!", '"', "'", "”")):
        return False
    core = s.rstrip(".").strip().lower()
    e = (entity or "").strip().lower()
    return core not in (e, e.split(" (")[0])
