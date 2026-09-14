import pytest

from src.metrics.consistency import (load_embedder, pairwise_similarity,
                                     select_medoid)


@pytest.fixture(scope="module")
def emb():
    return load_embedder()


def test_medoid_picks_the_consensus_sample(emb):
    """Three samples agree, one is unrelated. The medoid must come from the
    agreeing group — that is the whole mechanism."""
    samples = [
        "Ada Lovelace was an English mathematician who worked with Babbage.",
        "Ada Lovelace, an English mathematician, collaborated with Babbage.",
        "Ada Lovelace was a mathematician from England who knew Babbage.",
        "The Treaty of Westphalia ended the Thirty Years War in 1648.",
    ]
    r = select_medoid(samples, emb)
    assert r["selected_index"] in (0, 1, 2)
    assert r["text"] == samples[r["selected_index"]]


def test_outlier_is_never_selected(emb):
    samples = ["The cat sat on the mat."] * 3 + ["Quantum chromodynamics."]
    assert select_medoid(samples, emb)["selected_index"] != 3


def test_identical_samples_have_high_agreement(emb):
    """mean_sim is the pool's internal agreement. Identical samples mean
    selection cannot matter — worth reporting, because it says whether
    self-consistency has anything to work with."""
    r = select_medoid(["Ada Lovelace was a mathematician."] * 5, emb)
    assert r["mean_sim"] > 0.99


def test_disagreeing_samples_have_low_agreement(emb):
    samples = ["Ada Lovelace was a mathematician.",
               "The treaty was signed in Vienna.",
               "Photosynthesis converts light to energy.",
               "The bridge collapsed in heavy rain.",
               "He scored twice in the final."]
    assert select_medoid(samples, emb)["mean_sim"] < 0.3


def test_similarity_matrix_is_symmetric_and_bounded(emb):
    sim = pairwise_similarity(["one text", "another text", "a third"], emb)
    assert sim.shape == (3, 3)
    assert abs(sim - sim.T).max() < 1e-5
    assert sim.max() <= 1.001 and sim.min() >= -1.001


def test_empty_and_single(emb):
    assert select_medoid([], emb)["selected_index"] is None
    r = select_medoid(["only one"], emb)
    assert r["selected_index"] == 0 and r["mean_sim"] == 1.0