import math

import pytest

from src.metrics.fluency import load_scorer, perplexity


@pytest.fixture(scope="module")
def scorer():
    """Module-scoped: loading the scorer is the expensive part, and the model
    is stateless under torch.no_grad(), so one instance serves every test."""
    return load_scorer()


def test_repetition_lowers_perplexity(scorer):
    """06 §4 — the NLG lecture's warning as a unit test.

    Degenerate repetition makes text *more* predictable, so a low perplexity
    does not mean good text. Content is held constant: the same sentence, once
    versus ten times. Every copy after the first is nearly free to predict, so
    the looped version scores lower.

    Note this must be tested with the content controlled and the loop long
    enough for the model to lock on. A short nonsense loop compared against a
    different, near-memorized sentence does not show the effect (measured:
    2.69 vs 5.90 under Qwen2.5-0.5B) — the surprising first cycles dominate
    the mean NLL before the pattern is established.
    """
    model, tok = scorer
    s = "Ada Lovelace was an English mathematician and writer."
    once = perplexity(s, model, tok)
    looped = perplexity(" ".join([s] * 10), model, tok)
    assert looped < once, f"once={once:.2f} looped={looped:.2f}"


def test_fluent_scores_lower_than_word_salad(scorer):
    """Validity check: perplexity must actually track fluency, otherwise the
    metric measures nothing and every downstream fluency claim is empty."""
    model, tok = scorer
    fluent = perplexity("The capital of France is Paris.", model, tok)
    salad = perplexity("capital seventeen the beneath France running of", model, tok)
    assert fluent < salad, f"fluent={fluent:.2f} salad={salad:.2f}"


def test_too_short_is_nan(scorer):
    """Fewer than 2 tokens: there is no token with a context to condition on,
    so there is nothing to score. Must be nan, not 0.0 and not a crash —
    downstream code filters nan and reports the count (06 §6.2)."""
    model, tok = scorer
    assert math.isnan(perplexity("", model, tok))


def test_finite_and_positive(scorer):
    """Perplexity is exp of a mean NLL, so it is always > 0. A nan or inf here
    on ordinary text means the forward pass is broken, and one inf poisons the
    mean of a whole arm."""
    model, tok = scorer
    p = perplexity("The capital of France is Paris.", model, tok)
    assert math.isfinite(p) and p > 0, f"p={p}"