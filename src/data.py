"""Entity selection, reference pages, prompts, and the dataset statistics table.

data/entities.json is frozen: built once with a fixed seed and never
regenerated. Re-selecting entities after seeing results is selection bias.
"""
from __future__ import annotations

import json
import os
import random
import statistics as st

from src.text import split_sentences

POOL_TXT = "data/factscore_entities.txt"
PAGES_JSON = "data/reference_pages.json"
ENTITIES_JSON = "data/entities.json"

PROMPT_TEMPLATE = "Tell me a bio of {entity}."
SNAPSHOT = "20231101.en"

N_ENTITIES = 100
N_STRATA = 3
N_DEV = 20
MIN_PAGE_SENTENCES = 20      # the verifier needs something to retrieve from


def build_prompt(entity: str) -> str:
    """FActScore's prompt, verbatim. Base models take raw text, no chat template."""
    return PROMPT_TEMPLATE.format(entity=entity)


def load_pool() -> list[str]:
    """The 500 FActScore candidate names, as published."""
    return open(POOL_TXT).read().splitlines()


def load_pages() -> dict[str, str]:
    if not os.path.exists(PAGES_JSON):
        raise FileNotFoundError(
            f"{PAGES_JSON} missing — run scripts/build_reference_pages.py once")
    with open(PAGES_JSON) as f:
        return json.load(f)


def _candidates() -> list[dict]:
    """Pool entries that resolved to a page long enough to verify against."""
    pages = load_pages()
    out = []
    for title in load_pool():
        text = pages.get(title)
        if text is None:
            continue                       # absent from the pinned snapshot
        n_sent = len(split_sentences(text))
        if n_sent >= MIN_PAGE_SENTENCES:
            out.append({"entity": title,
                        "page_chars": len(text),
                        "page_sentences": n_sent})
    return out


def build_entities(n: int = N_ENTITIES, seed: int = 0) -> list[dict]:
    """Select n entities, stratified by reference-page length.

    Stratification matters: an all-famous sample puts every arm at the ceiling
    and an all-obscure one puts every arm at the floor. Either way the arms
    compress and the study loses its signal.
    """
    cand = _candidates()
    if len(cand) < n:
        raise ValueError(f"only {len(cand)} candidates pass the filter, need {n}")

    # Tertiles of page length — a proxy for how much the model knows about the
    # entity. Contiguous blocks after sorting; cand[i::3] would interleave the
    # strata and destroy the stratification.
    cand.sort(key=lambda c: c["page_chars"])
    strata = [cand[i * len(cand) // N_STRATA:(i + 1) * len(cand) // N_STRATA]
              for i in range(N_STRATA)]

    rng = random.Random(seed)
    per = [n // N_STRATA] * N_STRATA
    for i in range(n - sum(per)):          # spread the remainder
        per[i] += 1

    chosen: list[dict] = []
    for s_idx, (stratum, k) in enumerate(zip(strata, per)):
        picks = rng.sample(stratum, k)
        rng.shuffle(picks)
        n_dev = round(N_DEV * k / n)       # dev drawn within each stratum, so
        for j, c in enumerate(picks):      # the split is stratified too
            chosen.append({**c, "stratum": s_idx,
                           "split": "dev" if j < n_dev else "test"})

    # Correct any drift from the per-stratum rounding.
    dev = [c for c in chosen if c["split"] == "dev"]
    while len(dev) > N_DEV:
        dev.pop()["split"] = "test"
    for c in chosen:
        if len(dev) >= N_DEV:
            break
        if c["split"] == "test":
            c["split"] = "dev"
            dev.append(c)

    chosen.sort(key=lambda c: c["entity"])
    for i, c in enumerate(chosen):
        c["prompt_id"] = f"p{i:03d}"
        c["prompt"] = build_prompt(c["entity"])
    return chosen


def save_entities(entities: list[dict], path: str = ENTITIES_JSON) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists — it is frozen; do not regenerate")
    with open(path, "w") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)


def load_entities(path: str = ENTITIES_JSON) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def dataset_stats(entities: list[dict] | None = None) -> str:
    """The graded dataset-statistics table (guidelines: Methodology).

    Counts are recomputed rather than hard-coded, so the table can never drift
    away from what the code actually did.
    """
    e = entities or load_entities()
    pages = load_pages()
    n_pool = len(load_pool())
    n_resolved = sum(1 for t in load_pool() if t in pages)
    n_usable = len(_candidates())
    sents = [c["page_sentences"] for c in e]

    rows = [
        ("Number of entities", len(e)),
        ("Source", "FActScore biography entities (Min et al., 2023)"),
        ("Candidate pool", n_pool),
        ("Resolved against the pinned snapshot", f"{n_resolved} of {n_pool}"),
        (f"Usable (>= {MIN_PAGE_SENTENCES} reference sentences)", n_usable),
        ("Selection", f"stratified by reference-page length, {N_STRATA} strata"),
        ("Entities per stratum",
         "/".join(str(sum(1 for c in e if c["stratum"] == s))
                  for s in range(N_STRATA))),
        ("Dev / test split",
         f"{sum(1 for c in e if c['split'] == 'dev')} / "
         f"{sum(1 for c in e if c['split'] == 'test')}"),
        ("Reference length (sentences): mean", round(st.mean(sents), 1)),
        ("  median / min / max",
         f"{int(st.median(sents))} / {min(sents)} / {max(sents)}"),
        ("Data source and snapshot", f"wikimedia/wikipedia, {SNAPSHOT}"),
        ("Prompt", PROMPT_TEMPLATE),
    ]
    w = max(len(str(k)) for k, _ in rows)
    return "\n".join(f"{k:<{w}}  {v}" for k, v in rows)