import os

import pytest

from src.metrics.factuality import (NLI_NAME, THETA, TOP_M, WINDOW,
                                    build_index, load_nli, load_retriever,
                                    retrieve, score_generation, entail_probs)

PAGE = ("Ada Lovelace was born on 10 December 1815 in London. "
        "Her father was the poet Lord Byron. "
        "She worked with Charles Babbage on the Analytical Engine. "
        "She is often regarded as the first computer programmer. "
        "She died in 1852.")

slow = pytest.mark.skipif(not os.environ.get("SLOW"),
                          reason="set SLOW=1 to load the NLI model")


@pytest.fixture(scope="module")
def retriever():
    return load_retriever()


@pytest.fixture(scope="module")
def nli():
    return load_nli()


def test_chunking_overlaps(retriever):
    """Windows overlap by one sentence so a fact split across a window
    boundary still appears intact in some chunk."""
    idx = build_index(PAGE, retriever, window=WINDOW)
    assert len(idx["chunks"]) >= 2
    assert all(c.strip() for c in idx["chunks"])


def test_empty_page_is_handled(retriever):
    idx = build_index("", retriever)
    assert idx["chunks"] == []
    assert retrieve("anything", idx, retriever) == []


def test_retrieve_dedups_and_caps(retriever):
    """Hybrid takes the union of BM25 and dense top-k, so the same chunk can
    arrive twice."""
    idx = build_index(PAGE, retriever)
    prem = retrieve("She was born in London.", idx, retriever, m=TOP_M)
    assert len(prem) == len(set(prem))
    assert len(prem) <= TOP_M


@slow
def test_entailment_index_is_read_not_hardcoded(nli):
    """Across five MNLI checkpoints entailment sits at index 0, 0, 2, 1, 2.
    Hard-coding it returns a different label's probability, silently."""
    _, model, eid = nli
    assert model.config.id2label[eid].lower() == "entailment"


@slow
def test_verifier_discriminates(nli, retriever):
    """The three cases the metric must separate: supported, contradicted, and
    absent-from-the-page. If 'London' scores low, either the premise and
    hypothesis are swapped or the label index is wrong."""
    tok, model, eid = nli
    idx = build_index(PAGE, retriever)
    def p(claim):
        prem = retrieve(claim, idx, retriever)
        return max(entail_probs(prem, [f"Ada Lovelace: {claim}"] * len(prem),
                                tok, model, eid))
    assert p("She was born in 1815 in London.") > 0.9
    assert p("She was born in 1815 in Paris.") < 0.2
    assert p("She won an Olympic gold medal.") < 0.2


@slow
def test_degenerate_generation_is_excluded(nli, retriever):
    """Fewer than 2 sentences gives an unstable denominator — beam-4 produced
    256 tokens with no sentence-ending punctuation at GATE 1. Counted per arm,
    not scored."""
    tok, model, eid = nli
    idx = build_index(PAGE, retriever)
    r = score_generation("Ada Lovelace", "Ada Lovelace", idx, retriever,
                         tok, model, eid)
    assert r["degenerate"] is True
    assert r["support_rate"] is None


@slow
def test_support_rate_is_a_fraction(nli, retriever):
    tok, model, eid = nli
    idx = build_index(PAGE, retriever)
    text = ("Ada Lovelace was born in London in 1815. "
            "She worked with Charles Babbage. "
            "She won three Olympic gold medals in swimming.")
    r = score_generation(text, "Ada Lovelace", idx, retriever, tok, model, eid)
    assert r["n_sentences"] == 3
    assert 0.0 <= r["support_rate"] <= 1.0
    assert r["support_rate"] < 1.0          # the Olympic claim must fail