"""Verify one generation shard: support rate per record.

    python -m scripts.verify --model gpt2 --seed 1234 --split dev
    python -m scripts.verify --model llama-3b --seed 1234

Reads outputs/gen_{model}_seed{seed}.jsonl, writes
outputs/verdicts_{model}_seed{seed}.jsonl. Append-only, fsynced, resumable and
lock-protected — the same contract as generation, for the same reason: these
jobs get pre-empted.

sc_k5 records are skipped: their `text` is null until the self-consistency
selector runs. Verify them in a second pass afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time

from src.data import load_entities, load_pages
from src.generate import (GIT_COMMIT, acquire_lock, append, load_done,
                          release_lock)
from src.metrics.factuality import (NLI_NAME, RETRIEVER_NAME, THETA, TOP_M,
                                    WINDOW, build_index, load_nli,
                                    load_retriever, score_generation)

OUT_DIR = "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--split", default=None, choices=["dev", "test"])
    ap.add_argument("--arms", default=None, help="comma-separated; default all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--details", action="store_true", default=True,
                    help="store per-sentence evidence (needed for the manual check)")
    args = ap.parse_args()

    src = f"{OUT_DIR}/gen_{args.model}_seed{args.seed}.jsonl"
    out = f"{OUT_DIR}/verdicts_{args.model}_seed{args.seed}.jsonl"
    if not os.path.exists(src):
        raise SystemExit(f"{src} not found")

    arms = set(args.arms.split(",")) if args.arms else None
    splits = {e["entity"]: e["split"] for e in load_entities()}

    records = []
    for line in open(src):
        r = json.loads(line)
        if r.get("text") is None:          # sc_k5, pending the selector
            continue
        if arms and r["decoding"] not in arms:
            continue
        if args.split and splits.get(r["entity"]) != args.split:
            continue
        records.append(r)
    if args.limit:
        records = records[:args.limit]

    lock = acquire_lock(out)
    signal.signal(signal.SIGTERM,
                  lambda *_: (release_lock(lock), exit(143)))
    try:
        done = load_done(out)
        todo = [r for r in records
                if (r["model"], r["decoding"], r["seed"], r["prompt_id"]) not in done]
        print(f"{len(records)} scorable | {len(done)} done | {len(todo)} to run "
              f"-> {out}", flush=True)
        if not todo:
            return

        pages = load_pages()
        tok, model, entail_id = load_nli()
        retriever = load_retriever()
        print(f"entail_id={entail_id} ({model.config.id2label[entail_id]})", flush=True)

        index_cache: dict[str, dict] = {}     # per entity, built once
        nli_cache: dict[tuple, float] = {}    # per (premise, hypothesis)

        t0 = time.perf_counter()
        for i, r in enumerate(todo, 1):
            ent = r["entity"]
            if ent not in index_cache:
                index_cache[ent] = build_index(pages[ent], retriever)
            res = score_generation(r["text"], ent, index_cache[ent], retriever,
                                   tok, model, entail_id,
                                   theta=args.theta, cache=nli_cache)

            rec = {
                "model": r["model"], "decoding": r["decoding"],
                "seed": r["seed"], "prompt_id": r["prompt_id"],
                "entity": ent, "split": r["split"], "stratum": r["stratum"],
                "n_sentences": res["n_sentences"],
                "n_supported": res["n_supported"],
                "support_rate": res["support_rate"],
                "degenerate": res["degenerate"],
                "sentences": res["sentences"] if args.details else None,
                "nli_model": NLI_NAME, "retriever": RETRIEVER_NAME,
                "theta": args.theta, "top_m": TOP_M, "window": WINDOW,
                "git_commit": GIT_COMMIT,
            }
            append(out, rec)

            if i % 25 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(todo)}  {rate:.2f} gen/s  "
                      f"eta {(len(todo)-i)/rate/60:.0f} min  "
                      f"cache {len(nli_cache)}", flush=True)

        print(f"done in {(time.perf_counter()-t0)/60:.1f} min")
    finally:
        release_lock(lock)


if __name__ == "__main__":
    main()