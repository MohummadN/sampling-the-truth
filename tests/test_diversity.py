"""Tests for src/metrics/diversity.py.

The worked-example tests come straight from 06 §2.5, so they double as a check
that the file was read correctly rather than merely that the code runs.
"""
import math

import pytest

from src.metrics.diversity import (PREFIX_TOKENS, distinct_n,
                                   distinct_n_corpus, first_tokens,
                                   loop_severity, rep_n, self_bleu,
                                   tokenize_words)


# ------------------------------------------------------------- tokenization

def test_tokenizer_is_word_level_and_normalised():
    """Word tokens, never model tokens: GPT-2 and Llama tokenize differently,
    so a model-token metric would not be comparable across models — the same
    trap as perplexity."""
    assert tokenize_words("Ada Lovelace was born in 1815!") == \
        ["ada", "lovelace", "was", "born", "in", "1815"]


def test_tokenizer_keeps_contractions_whole():
    assert tokenize_words("she didn't publish it") == \
        ["she", "didn't", "publish", "it"]


# ------------------------------------------------------------- distinct / rep

def test_worked_examples_from_06():
    assert distinct_n("the cat sat on the mat", 1) == 5 / 6
    assert distinct_n("the cat sat on the mat", 2) == 1.0
    assert distinct_n("she was a mathematician " * 3, 2) == 4 / 11


def test_rep_is_the_complement():
    """rep-n is 1 - distinct-n. Report it as the complement; never present the
    two side by side as if they were independent evidence."""
    s = "she was a mathematician " * 3
    assert math.isclose(rep_n(s, 2) + distinct_n(s, 2), 1.0)


def test_too_short_is_nan():
    """Fewer tokens than n means no n-grams at all — nan, not 0.0, so it can
    be filtered and counted downstream rather than dragging a mean to zero."""
    assert math.isnan(distinct_n("word", 2))
    assert math.isnan(rep_n("word", 4))


def test_distinct_falls_as_text_repeats():
    """The whole premise of the diversity axis: degenerate text scores low."""
    varied = "ada lovelace was an english mathematician and writer"
    looped = "i am a buddhist " * 12
    assert distinct_n(looped, 2) < distinct_n(varied, 2)


def test_list_of_generations_is_rejected():
    """distinct_n takes ONE generation. Passing several used to be read as a
    single giant token and silently returned nan."""
    with pytest.raises(TypeError, match="distinct_n_corpus"):
        distinct_n(["first generation here", "second one"], 2)


def test_token_list_is_accepted():
    """The other half of that guard: pre-tokenized input must still work."""
    assert distinct_n(["the", "cat", "sat", "on", "the", "mat"], 1) == 5 / 6


# ------------------------------------------------------------------- corpus

def test_corpus_distinct_sees_across_generations():
    """Three identical generations are each internally varied but collectively
    identical — exactly what within-generation distinct-n cannot see."""
    g = "she was an english mathematician and writer"
    assert distinct_n(g, 2) == 1.0
    assert distinct_n_corpus([g, g, g], 2) < 0.4


def test_corpus_distinct_is_high_for_varied_generations():
    gens = ["she was an english mathematician and writer",
            "the treaty was signed in vienna during autumn",
            "photosynthesis converts light into chemical energy"]
    assert distinct_n_corpus(gens, 2) > 0.9


# ------------------------------------------------------------ loop severity

def test_loop_severity_catches_a_local_loop():
    """rep-n averages over the whole text, so one catastrophic loop inside
    otherwise varied output is under-reported. This is the independent signal."""
    varied = "ada lovelace was an english mathematician who worked with babbage"
    looped = varied + " i am a buddhist" * 12
    assert loop_severity(looped) > loop_severity(varied)


def test_loop_severity_is_bounded():
    assert 0.0 <= loop_severity("ada lovelace was an english mathematician") <= 1.0
    assert 0.0 <= loop_severity("i am a buddhist " * 20) <= 1.0


def test_loop_severity_of_short_text_is_zero():
    assert loop_severity("short") == 0.0


def test_loop_severity_finds_a_long_repeat():
    """A generation that is one phrase repeated is almost entirely a repeat."""
    assert loop_severity("i am a buddhist. " * 20) > 0.8


# ---------------------------------------------------------------- self-BLEU

def test_self_bleu_direction():
    """HIGH self-BLEU = generations resemble each other = LOW diversity.
    Getting this backwards would invert the headline result."""
    same = ["ada lovelace was an english mathematician."] * 3
    diff = ["ada lovelace was an english mathematician.",
            "the treaty was signed in vienna during autumn.",
            "photosynthesis converts light into chemical energy."]
    assert self_bleu(same) > 0.9
    assert self_bleu(diff) < 0.1


def test_self_bleu_is_one_for_deterministic_arms():
    """Greedy returns the same string for every seed, so self-BLEU across
    seeds is 1.0 by construction — correct, and worth stating in the caption
    rather than looking like a bug.

    The string must be longer than n: a 2-token string contains no 4-grams at
    all, so smoothing drags even an exact match to ~0.32. Real generations are
    100+ tokens, so that is a property of the toy input, not of the metric.
    """
    identical = "ada lovelace was an english mathematician and writer"
    assert self_bleu([identical] * 3) > 0.99


def test_self_bleu_needs_two_generations():
    assert math.isnan(self_bleu(["only one generation here"]))
    assert math.isnan(self_bleu([]))


def test_self_bleu_survives_a_missing_ngram_order():
    """Smoothing is mandatory: without it a single missing 4-gram order zeroes
    the whole score, and short generations would silently read as maximally
    diverse."""
    short = ["ada lovelace mathematician", "ada lovelace writer"]
    v = self_bleu(short)
    assert v == v and v > 0.0        # not nan, not a hard zero


# ------------------------------------------------------------- length control

def test_fixed_prefix_truncates():
    """Greedy and beam run to max_new_tokens while sampled arms terminate
    early, so length differences alone can produce the entire diversity
    ordering. Every length-sensitive metric is also reported on this prefix."""
    long = " ".join(["token"] * 500)
    assert len(first_tokens(long, 128)) == 128
    assert len(first_tokens("short text", 128)) == 2


def test_default_prefix_is_128():
    assert PREFIX_TOKENS == 128
    assert len(first_tokens(" ".join(["t"] * 300))) == 128


def test_prefix_changes_the_measured_value():
    """The confound made concrete: the same generation scores differently at
    full length than on a fixed prefix, which is why both are reported."""
    text = "ada lovelace was an english mathematician " * 20
    full = distinct_n(text, 2)
    clipped = distinct_n(first_tokens(text, 24), 2)
    assert clipped > full

def test_tokenizer_keeps_accented_letters():
    """[a-z0-9']+ shattered "Autónoma" into "aut" + "noma", inflating token
    counts and manufacturing n-grams. Seven of the 100 entities have accented
    names — the subset where the models are weakest."""
    assert tokenize_words("Universidad Nacional Autónoma de México") == \
        ["universidad", "nacional", "autónoma", "de", "méxico"]
    assert tokenize_words("Émile Zola") == ["émile", "zola"]