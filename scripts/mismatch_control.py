#!/usr/bin/env python3
"""Genericity control: are 'supported' sentences supported by the WRONG page?

Every sentence the verifier marked supported is re-scored against a different
entity's Wikipedia page, drawn from the SAME stratum so the evidence pool size
is comparable. A real fact about one person should not be entailed by another
person's page. The rate at which supported sentences survive the swap is a
direct measure of how generic each decoding arm's claims are.

    python -m scripts.mismatch_control --model gpt2 --seed 1234

Reads outputs/verdicts_{model}_seed{seed}.jsonl, writes
outputs/mismatch_{model}_seed{seed}.jsonl. Append-only, fsynced, resumable,
lock-protected — same contract as verification.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import signal
import time

from src.data import load_entities, load_pages
from src.generate import GIT_COMMIT, acquire_lock, append, release_lock
from src.metrics.factuality import (NLI_NAME, RETRIEVER_NAME, THETA, TOP_M,
                                    WINDOW, build_index, entail_probs,
                                    load_nli, load_retriever, retrieve)

OUT_DIR = "outputs"

_P = list(inspect.signature(entail_probs).parameters)
assert _P[:2] == ["premises", "hypotheses"], f"unexpected entail_probs signature: {_P}"
_PASS_CACHE = "cache" in _P


def wrong_entity_map(entities: list[dict]) -> dict[str, str]:
    """entity -> a different entity in the same stratum (deterministic rotation)."""
    by: dict = {}
    for e in entities:
        by.setdefault(e["stratum"], []).append(e["entity"])
    out = {}
    for names in by.values():
        names = sorted(names)
        for i, n in enumerate(names):
            out[n] = names[(i + 1) % len(names)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    stem = f"{args.model}_seed{args.seed}"
    src = f"{OUT_DIR}/verdicts_{stem}.jsonl"
    out = f"{OUT_DIR}/mismatch_{stem}.jsonl"
    if not os.path.exists(src):
        raise SystemExit(f"{src} not found")

    ents = load_entities()
    wrong = wrong_entity_map(ents)
    stratum = {e["entity"]: e["stratum"] for e in ents}

    work = []
    for line in open(src):
        r = json.loads(line)
        for i, s in enumerate(r.get("sentences") or []):
            if s.get("supported"):
                work.append({
                    "decoding": r["decoding"], "entity": r["entity"],
                    "prompt_id": r["prompt_id"], "sent_index": i,
                    "sentence": s["sentence"], "p_entail_true": s.get("p_entail"),
                })
    if args.limit:
        work = work[:args.limit]

    lock = acquire_lock(out)
    signal.signal(signal.SIGTERM, lambda *_: (release_lock(lock), exit(143)))
    try:
        done = set()
        if os.path.exists(out):
            for line in open(out):
                d = json.loads(line)
                done.add((d["decoding"], d["entity"], d["sent_index"]))
        todo = [w for w in work
                if (w["decoding"], w["entity"], w["sent_index"]) not in done]
        print(f"{len(work)} supported sentences | {len(done)} done | "
              f"{len(todo)} to run -> {out}", flush=True)
        if not todo:
            return

        # group by the wrong page so each index is built once
        todo.sort(key=lambda w: (wrong[w["entity"]], w["entity"], w["sent_index"]))

        pages = load_pages()
        tok, model, entail_id = load_nli()
        retriever = load_retriever()
        print(f"entail_id={entail_id} ({model.config.id2label[entail_id]})", flush=True)

        cache: dict = {}
        index = None
        cur = None
        t0 = time.perf_counter()

        for i, w in enumerate(todo, 1):
            w_ent = wrong[w["entity"]]
            if w_ent != cur:
                index = build_index(pages[w_ent], retriever)
                cur = w_ent

            windows = retrieve(w["sentence"], index, retriever, TOP_M)
            kw = {"cache": cache} if _PASS_CACHE else {}
            probs = entail_probs(windows, [w["sentence"]] * len(windows),
                                 tok, model, entail_id, **kw)
            p = max(probs) if probs else 0.0

            append(out, {
                "model": args.model, "seed": args.seed,
                "decoding": w["decoding"], "entity": w["entity"],
                "prompt_id": w["prompt_id"], "sent_index": w["sent_index"],
                "stratum": stratum[w["entity"]],
                "wrong_entity": w_ent,
                "sentence": w["sentence"],
                "p_entail_true": w["p_entail_true"],
                "p_entail_wrong": p,
                "supported_wrong": bool(p >= args.theta),
                "theta": args.theta, "top_m": TOP_M, "window": WINDOW,
                "nli_model": NLI_NAME, "retriever": RETRIEVER_NAME,
                "git_commit": GIT_COMMIT,
            })

            if i % 100 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f} sent/s  "
                      f"eta {(len(todo) - i) / rate / 60:.0f} min", flush=True)

        print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
    finally:
        release_lock(lock)


if __name__ == "__main__":
    main()
