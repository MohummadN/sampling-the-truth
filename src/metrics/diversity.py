"""Diversity and repetition metrics. Pure Python, no models.

Word tokens, never model tokens: GPT-2 and Llama tokenize differently, so a
model-token metric would not be comparable across models — the same trap as
perplexity (06 §2).
"""
from __future__ import annotations

import re
from collections import Counter

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

PREFIX_TOKENS = 128     # fixed-prefix length for the length-controlled variant
# Unicode-aware: [a-z0-9']+ would shatter "Autónoma" into "aut"+"noma",
# manufacturing n-grams that were never in the text. Seven entities have
# accented names and their pages are dense with accented place names.
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def tokenize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation, split on whitespace. Fixed once, used
    everywhere, stated in the paper."""
    return _WORD.findall(text.lower())


def _toks(x: str | list[str]) -> list[str]:
    if isinstance(x, str):
        return tokenize_words(x)
    x = list(x)
    if any(" " in t for t in x):
        raise TypeError(
            "expected one generation (str) or its word tokens (list[str]); "
            "got a list of multi-word strings — use distinct_n_corpus for "
            "several generations")
    return x


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(x: str | list[str], n: int = 2) -> float:
    """Unique n-grams / total n-grams, within one generation.

    Falls mechanically as text gets longer, which is why every result is also
    reported on a fixed prefix (06 §2.4).
    """
    tokens = _toks(x)
    grams = _ngrams(tokens, n)
    if not grams:
        return float("nan")
    return len(set(grams)) / len(grams)


def distinct_n_corpus(generations: list[str | list[str]], n: int = 2) -> float:
    """Same ratio, pooled across generations — the inter-generation sense.

    Distinct from the within-generation version: an arm can write fluent,
    internally varied text and still produce near-identical text every time.
    """
    grams: list[tuple] = []
    for g in generations:
        grams.extend(_ngrams(_toks(g), n))
    if not grams:
        return float("nan")
    return len(set(grams)) / len(grams)


def rep_n(x: str | list[str], n: int = 4) -> float:
    """1 - distinct_n. Report it as the complement, never alongside distinct-n
    as if the two were independent evidence."""
    d = distinct_n(x, n)
    return float("nan") if d != d else 1.0 - d


def loop_severity(text: str, min_len: int = 10) -> float:
    """Longest repeated substring, as a fraction of the text.

    The genuinely independent signal in this file. rep-n averages over the
    whole text, so one catastrophic 200-character loop inside otherwise varied
    output is under-reported; this catches it.
    """
    s = text.strip()
    if len(s) < min_len:
        return 0.0

    def repeats(length: int) -> bool:
        seen = set()
        for i in range(len(s) - length + 1):
            sub = s[i:i + length]
            if sub in seen:
                return True
            seen.add(sub)
        return False

    # Upper bound is len(s) - 1, not len(s) // 2: a repeated substring may
    # overlap itself ("ababab" occurs twice in "abababab"), so a //2 cap
    # would pin every fully degenerate generation at exactly 0.5 — the one
    # region where this metric has to discriminate.
    lo, hi, best = min_len, len(s) - 1, 0
    while lo <= hi:                       # binary search on the repeat length
        mid = (lo + hi) // 2
        if repeats(mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best / len(s)


def self_bleu(generations: list[str], n: int = 4) -> float:
    """Mean BLEU of each generation against all the others.

    HIGH = generations resemble each other = LOW diversity. Deterministic arms
    score 1.0 across seeds by construction, which is the correct answer and
    worth stating in the caption rather than looking like a bug.

    Smoothing is mandatory: without it a single missing 4-gram order zeroes the
    whole score (06 §3.6).
    """
    if len(generations) < 2:
        return float("nan")
    toks = [tokenize_words(g) for g in generations]
    smooth = SmoothingFunction().method1
    weights = tuple([1.0 / n] * n)
    scores = []
    for i, hyp in enumerate(toks):
        refs = [t for j, t in enumerate(toks) if j != i]
        if not hyp or not any(refs):
            continue
        scores.append(sentence_bleu(refs, hyp, weights=weights,
                                    smoothing_function=smooth))
    return sum(scores) / len(scores) if scores else float("nan")


def first_tokens(x: str | list[str], length: int = PREFIX_TOKENS) -> list[str]:
    """The length control. Greedy and beam run to max_new_tokens while sampled
    arms terminate early, so length differences alone can produce the entire
    diversity ordering (06 §2.4). Every length-sensitive metric is reported
    full-length AND on this prefix."""
    return _toks(x)[:length]