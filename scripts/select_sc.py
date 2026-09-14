#!/usr/bin/env python3
"""Pick the self-consistency answer for every sc_k5 record.

For each entity the sc_k5 arm produced k=5 independent samples. This picks the
medoid (the sample most similar to the other four) and writes two files:

  outputs/scsel_{model}_seed{seed}.jsonl    5 pseudo-records per entity,
      decoding = "sc_k5#{i}", ready for scripts/verify.py. Verifying all five
      rather than only the winner costs the same order of GPU time and buys the
      oracle best-of-5 bound and the pool-vs-selector decomposition.

  outputs/scchoice_{model}_seed{seed}.jsonl 1 row per entity: which sample the
      selector chose and how much the pool agreed.

gen_tokens / compute_tokens are null on the pseudo-records: the arm's cost is
already accounted once on the original sc_k5 record, and copying it onto five
rows would quintuple it in any downstream sum.
"""
from __future__ import annotations

import argparse
import json
import os

from src.device import pick_device
from src.metrics.consistency import load_embedder, select_medoid

ARM = "sc_k5"
SAMPLE_KEYS = ("samples", "texts", "generations", "sequences")


def samples_of(rec: dict) -> list[str]:
    for key in SAMPLE_KEYS:
        val = rec.get(key)
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val
    val = rec.get("text")
    if isinstance(val, list) and val and isinstance(val[0], str):
        return val
    raise KeyError(
        f"no list-of-strings sample field for {rec.get('entity')!r}; "
        f"keys present: {sorted(rec)}"
    )


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N entities")
    args = ap.parse_args()

    stem = f"{args.model}_seed{args.seed}"
    src = os.path.join(args.outdir, f"gen_{stem}.jsonl")
    sel_path = os.path.join(args.outdir, f"scsel_{stem}.jsonl")
    choice_path = os.path.join(args.outdir, f"scchoice_{stem}.jsonl")

    records = [r for r in read_jsonl(src) if r.get("decoding") == ARM]
    if not records:
        raise SystemExit(f"no {ARM} records in {src}")
    if args.limit:
        records = records[: args.limit]

    done = {r.get("entity") for r in read_jsonl(choice_path)}
    seen = {(r.get("entity"), r.get("sc_sample_index")) for r in read_jsonl(sel_path)}
    todo = [r for r in records if r.get("entity") not in done]
    print(
        f"{src}: {len(records)} {ARM} records, {len(done)} already selected, "
        f"{len(todo)} to do",
        flush=True,
    )
    if not todo:
        return

    print(f"device: {pick_device()}", flush=True)
    embedder = load_embedder()

    for i, rec in enumerate(todo, 1):
        samples = samples_of(rec)
        sel = select_medoid(samples, embedder)
        entity = rec.get("entity")

        for j, text in enumerate(samples):
            if (entity, j) in seen:
                continue
            row = dict(rec)
            for key in SAMPLE_KEYS:
                row.pop(key, None)
            row["decoding"] = f"{ARM}#{j}"
            row["text"] = text
            row["sc_parent"] = ARM
            row["sc_sample_index"] = j
            row["sc_selected"] = j == sel["selected_index"]
            row["sc_mean_sim"] = sel["mean_sim"]
            row["sc_selected_sim"] = sel["selected_sim"]
            row["gen_tokens"] = None
            row["compute_tokens"] = None
            append(sel_path, row)

        append(
            choice_path,
            {
                "model": rec.get("model"),
                "seed": rec.get("seed"),
                "task": rec.get("task"),
                "entity": entity,
                "decoding": ARM,
                "k": len(samples),
                "selected_index": sel["selected_index"],
                "mean_sim": sel["mean_sim"],
                "selected_sim": sel["selected_sim"],
                "gen_tokens": rec.get("gen_tokens"),
                "compute_tokens": rec.get("compute_tokens"),
            },
        )

        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)

    print(f"wrote {sel_path} and {choice_path}", flush=True)


if __name__ == "__main__":
    main()