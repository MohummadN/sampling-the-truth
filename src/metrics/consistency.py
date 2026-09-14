"""Self-consistency selection for the sc_k5 arm.

Wang et al. (2023) aggregate by MAJORITY VOTE over answers. That is undefined
for open-ended text: five biographies are never string-identical, so there is
nothing to count. The proposal therefore replaces the vote with embedding
centroid selection — pick the sample most similar to the other four.

This is a real redefinition of the method, not an implementation detail, and
it must be declared: what we evaluate is "sample five, keep the medoid", not
self-consistency as Wang et al. defined it (09 Part C).

The medoid is a sensible stand-in for the same intuition — the vote keeps the
answer the model agrees with itself about most often; the medoid keeps the
sample nearest the consensus of the pool. It cannot invent a better sample
than the pool contains, which is why the analysis reports the oracle
best-of-five bound alongside it (03 §10).
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.device import pick_device

# Same encoder as the evidence retriever: one embedding model in the project,
# already cached, and its checkpoint is already named in Experimental Setup.
EMBEDDER_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_embedder(name: str = EMBEDDER_NAME) -> SentenceTransformer:
    return SentenceTransformer(name, device=pick_device())


def pairwise_similarity(samples: list[str], embedder) -> np.ndarray:
    """Cosine similarity matrix over the k samples.

    Embeddings are L2-normalised, so the inner product IS cosine similarity —
    the equivalence the Retrieval lecture notes for normalised vectors.
    """
    emb = embedder.encode(samples, normalize_embeddings=True,
                          show_progress_bar=False)
    return np.asarray(emb) @ np.asarray(emb).T


def select_medoid(samples: list[str], embedder) -> dict:
    """Pick the sample with the highest mean similarity to the others.

    Returns the index, the text, and diagnostics. mean_sim is the pool's
    internal agreement: near 1 means the five samples say the same thing (so
    selection cannot matter much), near 0 means the model is guessing
    differently each time. That number is itself a result — it says whether
    self-consistency has anything to work with.
    """
    if not samples:
        return {"selected_index": None, "text": None,
                "mean_sim": None, "selected_sim": None}
    if len(samples) == 1:
        return {"selected_index": 0, "text": samples[0],
                "mean_sim": 1.0, "selected_sim": 1.0}

    sim = pairwise_similarity(samples, embedder)
    k = len(samples)
    np.fill_diagonal(sim, 0.0)                  # exclude self-similarity (=1)
    mean_to_others = sim.sum(axis=1) / (k - 1)
    idx = int(np.argmax(mean_to_others))

    # Off-diagonal mean over the whole matrix: the pool's internal agreement.
    pool_mean = float(sim.sum() / (k * (k - 1)))

    return {"selected_index": idx,
            "text": samples[idx],
            "mean_sim": round(pool_mean, 4),
            "selected_sim": round(float(mean_to_others[idx]), 4)}