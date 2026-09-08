import pytest

from src.data import (N_DEV, N_ENTITIES, N_STRATA, MIN_PAGE_SENTENCES,
                      build_entities, build_prompt, load_entities, load_pages)


def test_prompt_is_factscore_verbatim():
    """A richer prompt would lower output entropy and shrink the very
    differences between decoders that the study measures (04 Topic 1)."""
    assert build_prompt("Ada Lovelace") == "Tell me a bio of Ada Lovelace."


def test_frozen_file_has_the_expected_shape():
    e = load_entities()
    assert len(e) == N_ENTITIES
    assert sum(1 for c in e if c["split"] == "dev") == N_DEV
    assert len({c["entity"] for c in e}) == N_ENTITIES
    assert len({c["prompt_id"] for c in e}) == N_ENTITIES


def test_all_strata_populated():
    e = load_entities()
    counts = [sum(1 for c in e if c["stratum"] == s) for s in range(N_STRATA)]
    assert min(counts) > 0 and sum(counts) == N_ENTITIES


def test_dev_is_spread_across_strata():
    """A dev set drawn from one stratum would tune theta on easy entities only."""
    e = load_entities()
    dev_strata = {c["stratum"] for c in e if c["split"] == "dev"}
    assert dev_strata == set(range(N_STRATA))


def test_every_entity_has_a_usable_page():
    e = load_entities()
    pages = load_pages()
    for c in e:
        assert c["entity"] in pages
        assert c["page_sentences"] >= MIN_PAGE_SENTENCES


def test_selection_is_deterministic():
    """Same seed, same list — otherwise the frozen file is not reproducible."""
    a = [c["entity"] for c in build_entities(seed=0)]
    b = [c["entity"] for c in build_entities(seed=0)]
    assert a == b


def test_different_seed_gives_a_different_sample():
    a = {c["entity"] for c in build_entities(seed=0)}
    b = {c["entity"] for c in build_entities(seed=1)}
    assert a != b
    