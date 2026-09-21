#!/usr/bin/env python3
"""Diversity and fluency for every generation in one (model, seed) shard.

    python -m scripts.compute_metrics --model gpt2 --seed 1234

Reads outputs/gen_{model}_seed{seed}.jsonl for the eight single-text arms and
outputs/scsel_{model}_seed{seed}.jsonl for the five sc_k5 samples, and writes
outputs/metrics_{model}_seed{seed}.jsonl. Append-only, fsynced, resumable and
lock-protected - the same contract as generation and verification.

Every length-sensitive metric is reported twice: over the whole generation and
over the first 128 word tokens (DECISIONS 2026-09-07). Greedy and beam run to
the 256-token cap while sampled arms stop early, so an uncontrolled diversity
gap is partly a length gap - and it would flatter our own hypothesis.

Self-BLEU and pooled distinct-n are NOT here: they compare generations to each
other across seeds, so they belong to the analysis step, which sees all three
seeds at once. This script is per record.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time

from src.data import load_entities
from src.device import pick_device
from src.generate import GIT_COMMIT, acquire_lock, append, load_done, release_lock
from src.metrics.diversity import (PREFIX_TOKENS, distinct_n, first_tokens,
                                   loop_severity, rep_n, tokenize_words)
from src.metrics.fluency import SCORER, load_scorer, perplexity

OUT_DIR = "outputs"


def clean(x):
    """nan is not representable in JSON; store null so consumers can filter."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def measure(text: str, tok, model) -> dict:
    words = tokenize_words(text)
    prefix = first_tokens(text, PREFIX_TOKENS)
    prefix_text = " ".join(prefix)
    return {
        "n_words": len(words),
        "n_words_prefix": len(prefix),
        "distinct_1": clean(distinct_n(words, 1)),
        "distinct_2": clean(distinct_n(words, 2)),
        "rep_4": clean(rep_n(words, 4)),
        "loop_severity": clean(loop_severity(text)),
        "distinct_1_prefix": clean(distinct_n(prefix, 1)),
        "distinct_2_prefix": clean(distinct_n(prefix, 2)),
        "rep_4_prefix": clean(rep_n(prefix, 4)),
        "loop_severity_prefix": clean(loop_severity(prefix_text)),
        "perplexity": clean(perplexity(text, tok, model)),
        "perplexity_prefix": clean(perplexity(prefix_text, tok, model)),
    }


def records_for(model: str, seed: int) -> list[dict]:
    """Every generation with text: the eight single-text arms, plus the five
    sc_k5 samples from the selector's pseudo-records."""
    out = []
    for path, in_sc in ((f"{OUT_DIR}/gen_{model}_seed{seed}.jsonl", False),
                        (f"{OUT_DIR}/scsel_{model}_seed{seed}.jsonl", True)):
        if not os.path.exists(path):
            if not in_sc:
                raise SystemExit(f"{path} not found")
            continue
        for line in open(path):
            r = json.loads(line)
            if not isinstance(r.get("text"), str) or not r["text"].strip():
                continue          # sc_k5 parent rows carry samples, not text
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--scorer", default=SCORER, help="fluency scorer; override for tests")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="debugging only; never for grid shards")
    args = ap.parse_args()

    out = f"{OUT_DIR}/metrics_{args.model}_seed{args.seed}.jsonl"
    recs = records_for(args.model, args.seed)
    if args.limit:
        recs = recs[:args.limit]
    splits = {e["entity"]: e["split"] for e in load_entities()}

    lock = acquire_lock(out)
    signal.signal(signal.SIGTERM, lambda *_: (release_lock(lock), exit(143)))
    try:
        done = load_done(out)
        todo = [r for r in recs
                if (r["model"], r["decoding"], r["seed"], r["prompt_id"]) not in done]
        print(f"{len(recs)} generations | {len(done)} done | {len(todo)} to run -> {out}",
              flush=True)
        if not todo:
            return

        if pick_device() != "cuda" and not args.allow_cpu:
            raise SystemExit(
                "refusing to run on cpu: perplexity would take hours and the "
                "numbers must match the rest of the grid. Resubmit this index.")

        tok, scorer = load_scorer(args.scorer)
        print(f"scorer {args.scorer} on {scorer.device}", flush=True)

        t0 = time.perf_counter()
        for i, r in enumerate(todo, 1):
            rec = {
                "model": r["model"], "decoding": r["decoding"], "seed": r["seed"],
                "prompt_id": r["prompt_id"], "entity": r["entity"],
                "split": splits.get(r["entity"]), "stratum": r.get("stratum"),
                **measure(r["text"], tok, scorer),
                "sc_sample_index": r.get("sc_sample_index"),
                "sc_selected": r.get("sc_selected"),
                "scorer": args.scorer,
                "device": str(scorer.device),
                "git_commit": GIT_COMMIT,
            }
            append(out, rec)
            if i % 200 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f} gen/s  "
                      f"eta {(len(todo)-i)/rate/60:.0f} min", flush=True)
        print(f"done in {(time.perf_counter()-t0)/60:.1f} min")
    finally:
        release_lock(lock)


if __name__ == "__main__":
    main()
