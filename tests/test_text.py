"""Tests for src/text.py — sentence segmentation and the claim filter.

Both decide what counts as a unit of the support-rate metric, so a change here
silently moves every factuality number in the paper.
"""
import pytest

from src.text import MIN_CHARS, is_claim, split_sentences


# ------------------------------------------------------------ segmentation

def test_splits_plain_prose():
    assert len(split_sentences(
        "Ada Lovelace was an English mathematician. "
        "She worked with Charles Babbage.")) == 2


@pytest.mark.parametrize("text", [
    "She earned a Ph.D. in 1952. She then taught for many years.",
    "He moved to the U.S. in 1961. He became a citizen later on.",
    "She was born in 1815. She died much later in 1852.",
])
def test_abbreviations_and_years_do_not_split(text):
    """Never split on '.' — it shreds Ph.D., U.S. and every year. This is why
    split_sentences uses punkt rather than text.split('.')."""
    assert len(split_sentences(text)) == 2


def test_fragments_below_min_chars_are_dropped():
    """MIN_CHARS keeps 'Yes.' and 'No.' out of the denominator. Lowering it
    would admit them as scoreable claims, which is worse than losing them."""
    out = split_sentences("Yes. No. Ok. She was a mathematician of some note.")
    assert all(len(s) >= MIN_CHARS for s in out)
    assert not any(s.strip() in ("Yes.", "No.") for s in out)


def test_empty_text_yields_nothing():
    assert split_sentences("") == []


def test_newline_separated_text_does_not_split():
    """A known limitation, documented rather than fixed: punkt needs terminal
    punctuation, so newline-separated repetition collapses to one segment and
    the record is excluded as degenerate (docs/metric_validity.md Problem 2)."""
    assert len(split_sentences("Amr Shabana\nAmr Shabana\nAmr Shabana")) == 1


# -------------------------------------------------------------- is_claim

@pytest.mark.parametrize("sentence, entity", [
    ("Douglas Wood (engineer).", "Douglas Wood (engineer)"),
    ("Felipe (footballer, born 1977).", "Felipe (footballer, born 1977)"),
    ("Radja Nainggolan", "Radja Nainggolan"),
])
def test_title_echo_is_not_a_claim(sentence, entity):
    """The article title echoed back becomes '{entity}: {entity}' once the
    verifier prepends the entity — a tautology entailed by any page at all.
    210 of GPT-2 beam4's supported sentences were these."""
    assert is_claim(sentence, entity) is False


@pytest.mark.parametrize("sentence", [
    "How did he become involved?",
    "What is her qualification?",
])
def test_questions_are_not_claims(sentence):
    assert is_claim(sentence, "Adnan Sami") is False


@pytest.mark.parametrize("sentence, entity", [
    ("Jose Cardozo is", "Jose Cardozo"),
    ("Fahadh Faasil is", "Fahadh Faasil"),
])
def test_truncation_fragments_are_not_claims(sentence, entity):
    """The 256-token cap cuts mid-sentence; the tell is no terminal
    punctuation. 492 supported sentences were fragments like these."""
    assert is_claim(sentence, entity) is False


@pytest.mark.parametrize("sentence, entity", [
    ("He was British.", "Tom Coburn"),
    ("I'm a musician.", "Terrence Romeo"),
    ("She worked with Charles Babbage on the Analytical Engine.", "Ada Lovelace"),
    ('He said "I never lost."', "Ricardo Silva"),
    ("Douglas Wood was an engineer.", "Douglas Wood (engineer)"),
])
def test_real_claims_survive(sentence, entity):
    """Short is not the same as empty: 'He was British.' is three words and a
    perfectly good claim, which is why the filter has no word-count rule. And
    the name plus a predicate is a claim even though the bare name is not."""
    assert is_claim(sentence, entity) is True


def test_empty_is_not_a_claim():
    assert is_claim("", "Ada Lovelace") is False
    assert is_claim(None, "Ada Lovelace") is False


def test_filter_is_applied_to_both_sides_of_the_ratio():
    """Guard against the tempting bug: dropping non-claims from the numerator
    only would inflate every rate instead of correcting it."""
    ent = "Douglas Wood (engineer)"
    sents = ["Douglas Wood (engineer).", "He was a British engineer.",
             "How did he begin?", "He worked in Leeds."]
    kept = [s for s in sents if is_claim(s, ent)]
    assert kept == ["He was a British engineer.", "He worked in Leeds."]
