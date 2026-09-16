#!/usr/bin/env python3
"""Draw the 100-sentence manual validation sample (Stage 7).

Stratified by entity stratum x p_entail band, with the 0.1-0.9 middle
oversampled because that is where the theta decision lives. Each cell's
population is recorded so estimates reweight back to the full frame.

Writes:
  outputs/labeling_sheet.csv  BLIND: id, entity, sentence, evidence, label, note
                              shuffled, no verifier output visible
  outputs/labeling_key.csv    id -> model, arm, seed, stratum, band, p_entail,
                              verifier verdict, sampling weight
  outputs/labeling_frame.json cell populations, for reweighting

Label values: supported | unsupported | absent
  supported   - the evidence shown establishes the claim
  unsupported - the claim contradicts the evidence, or is not establishable
  absent      - the claim is TRUE of this person but is not in the page
The third value is what bounds the reference-coverage mechanism in
docs/metric_validity.md; it cannot be recovered later, so use it carefully.
"""
import collections
import csv
import glob
import json
import random
import re

from src.data import load_entities
from src.text import is_claim

SEED = 20260914
BANDS = [(0.0, 0.1, "lo"), (0.1, 0.5, "mid_lo"),
         (0.5, 0.9, "mid_hi"), (0.9, 1.01, "hi")]
TARGET = {"lo": 4, "mid_lo": 9, "mid_hi": 11, "hi": 9}   # per stratum -> 33 x 3
WS = re.compile(r"\s+")


def band_of(p):
    for lo, hi, name in BANDS:
        if lo <= p < hi:
            return name
    return "hi"


def main():
    ents = load_entities()
    strat = {e["entity"]: e["stratum"] for e in ents}
    # DEV ONLY. theta is chosen to maximise agreement with these labels
    # (DECISIONS.md, 2026-09-07: all tuning happens on the 20 dev entities,
    # only the 80 test are reported). Drawing them from test would tune theta
    # on test and the reported numbers would no longer be held out.
    dev = {e["entity"] for e in ents if e["split"] == "dev"}
    frame = collections.defaultdict(list)

    for path in sorted(glob.glob("outputs/verdicts_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["entity"] not in dev:
                continue
            seen = set()
            for i, s in enumerate(r.get("sentences") or []):
                # Non-claims (title echoes, questions, truncation fragments)
                # would burn the 100-label budget on segments that assert
                # nothing. Same filter the metric uses.
                if not is_claim(s["sentence"], r["entity"]):
                    continue
                norm = WS.sub(" ", s["sentence"].strip().lower())
                if norm in seen:          # a repetition loop contributes once
                    continue
                seen.add(norm)
                p = float(s["p_entail"])
                frame[(strat[r["entity"]], band_of(p))].append({
                    "model": r["model"], "arm": r["decoding"], "seed": r["seed"],
                    "entity": r["entity"], "stratum": strat[r["entity"]],
                    "band": band_of(p), "p_entail": p,
                    "verifier": "supported" if s["supported"] else "unsupported",
                    "sentence": s["sentence"], "evidence": s["evidence"],
                })

    rng = random.Random(SEED)

    # allocate first, then sample once, so every row carries its cell weight
    want = {k: min(TARGET[k[1]], len(v)) for k, v in frame.items()}
    while sum(want.values()) < 100:
        cell = max(frame, key=lambda k: len(frame[k]) - want[k])
        if len(frame[cell]) - want[cell] <= 0:
            break
        want[cell] += 1

    picked, pops = [], {}
    for key, rows in sorted(frame.items()):
        n = want[key]
        if not n:
            continue
        pops[f"stratum{key[0]}_{key[1]}"] = {"population": len(rows), "sampled": n}
        for row in rng.sample(rows, n):
            row["weight"] = len(rows) / n
            picked.append(row)

    rng.shuffle(picked)
    for i, row in enumerate(picked, 1):
        row["id"] = f"L{i:03d}"

    with open("outputs/labeling_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "entity", "sentence", "evidence", "label", "note"])
        for r in picked:
            w.writerow([r["id"], r["entity"], r["sentence"], r["evidence"], "", ""])

    with open("outputs/labeling_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "model", "arm", "seed", "entity", "stratum", "band",
                    "p_entail", "verifier", "weight"])
        for r in picked:
            w.writerow([r["id"], r["model"], r["arm"], r["seed"], r["entity"],
                        r["stratum"], r["band"], f"{r['p_entail']:.4f}",
                        r["verifier"], f"{r['weight']:.1f}"])

    json.dump({"seed": SEED, "cells": pops,
               "total_frame": sum(len(v) for v in frame.values()),
               "sampled": len(picked)},
              open("outputs/labeling_frame.json", "w"), indent=2)

    print(f"frame: {sum(len(v) for v in frame.values())} unique sentences")
    print(f"sampled: {len(picked)}")
    by_band = collections.Counter(r["band"] for r in picked)
    by_strat = collections.Counter(r["stratum"] for r in picked)
    by_verif = collections.Counter(r["verifier"] for r in picked)
    print("by band:   ", dict(by_band))
    print("by stratum:", dict(by_strat))
    print("by verifier (hidden from the sheet):", dict(by_verif))
    print("\nwrote outputs/labeling_sheet.csv, labeling_key.csv, labeling_frame.json")


if __name__ == "__main__":
    main()
