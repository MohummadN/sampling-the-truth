#!/usr/bin/env python3
"""Calibrate reported support rates against the 100 manual labels.

The verifier is high-recall and low-precision: it almost never misses a
supported sentence, but confidently entails claims the evidence does not
address (34 of 36 false positives are human-labelled not_addressed, mean
p_entail 0.754). Reported support rates are therefore inflated by its
false-positive rate.

Rogan-Gladen correction, which is prevalence-independent and so transfers
from the label frame to each arm:

    T = (R - FPR) / (TPR - FPR)

Uncertainty combines both sources — the 100 labels and the 100 prompts —
resampled jointly, with the prompt resample shared across arms so the
comparison stays paired.
"""
import argparse
import collections
import csv
import glob
import json
import random

LAB, KEY = "outputs/labels.csv", "outputs/labeling_key.csv"
THETA, B = 0.5, 2000
MODELS = ("gpt2", "llama-1b", "llama-3b")


def load_labels():
    lab = {r["id"]: r for r in csv.DictReader(open(LAB, encoding="utf-8"))}
    key = {r["id"]: r for r in csv.DictReader(open(KEY, encoding="utf-8"))}
    rows = []
    for i, l in lab.items():
        if l["label"] == "unsure":
            continue
        k = key[i]
        rows.append({"h": l["label"] == "supported",
                     "v": float(k["p_entail"]) >= THETA,
                     "w": float(k["weight"]),
                     "s": str(k["stratum"])})
    return rows


def rates(rows):
    pos = sum(r["w"] for r in rows if r["h"])
    neg = sum(r["w"] for r in rows if not r["h"])
    tp = sum(r["w"] for r in rows if r["h"] and r["v"])
    fp = sum(r["w"] for r in rows if not r["h"] and r["v"])
    return (tp / pos if pos else None, fp / neg if neg else None)


def correct(R, tpr, fpr):
    if tpr is None or fpr is None:
        return None
    d = tpr - fpr
    if d <= 0:
        return None
    return min(1.0, max(0.0, (R - fpr) / d))


def load_observed():
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    strat = {}
    for p in glob.glob("outputs/verdicts_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            v = r["support_rate"]
            acc[(r["model"], r["decoding"])][r["entity"]].append(
                0.0 if v is None else v)
            strat[r["entity"]] = str(r["stratum"])
    return acc, strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="beam4", help="arm for the stratum table")
    args = ap.parse_args()

    labels = load_labels()
    acc, strat = load_observed()
    ents = sorted(strat)
    idx_of = {e: i for i, e in enumerate(ents)}

    tpr, fpr = rates(labels)
    print(f"verifier at theta={THETA}: TPR {tpr:.3f}  FPR {fpr:.3f}  "
          f"(n={len(labels)} labels)")
    print(f"calibration: T = (R - {fpr:.3f}) / ({tpr:.3f} - {fpr:.3f})"
          f"  ->  T ~ {1/(tpr-fpr):.2f} R - {fpr/(tpr-fpr):.3f}")
    print(f"false-positive floor: any arm at or below R = {fpr:.3f} is "
          f"indistinguishable from zero true support\n")

    # per-arm vectors aligned to `ents`
    vec = {}
    for k, per_ent in acc.items():
        vec[k] = [sum(per_ent[e]) / len(per_ent[e]) if per_ent.get(e) else 0.0
                  for e in ents]

    rng = random.Random(0)
    boots = collections.defaultdict(list)
    n = len(ents)
    for _ in range(B):
        ls = rng.choices(labels, k=len(labels))
        t_b, f_b = rates(ls)
        idx = [rng.randrange(n) for _ in range(n)]
        for k, v in vec.items():
            R_b = sum(v[i] for i in idx) / n
            c = correct(R_b, t_b, f_b)
            if c is not None:
                boots[k].append(c)

    for model in MODELS:
        rows = []
        for (m, arm), v in vec.items():
            if m != model:
                continue
            R = sum(v) / n
            T = correct(R, tpr, fpr)
            bs = sorted(boots[(m, arm)])
            lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
            rows.append((arm, R, T, lo, hi))
        rows.sort(key=lambda r: -r[1])
        print(f"\n{model}")
        print(f"  {'arm':14s} {'raw':>7s} {'calibrated':>11s} {'95% CI':>18s}")
        print("  " + "-" * 54)
        for arm, R, T, lo, hi in rows:
            flag = "   <- at the FP floor" if lo <= 0.0005 else ""
            print(f"  {arm:14s} {R:7.3f} {T:11.3f} "
                  f"[{lo:.3f}, {hi:.3f}]{flag}")

    print(f"\n\nper-stratum calibration, arm = {args.arm}")
    print(f"  {'model':10s} {'str':>3s} {'TPR':>6s} {'FPR':>6s} "
          f"{'raw':>7s} {'calibrated':>11s}")
    print("  " + "-" * 50)
    for model in MODELS:
        v = acc.get((model, args.arm))
        if not v:
            continue
        for s in ("0", "1", "2"):
            sub = [r for r in labels if r["s"] == s]
            t_s, f_s = rates(sub)
            es = [e for e in ents if strat[e] == s]
            R = sum(sum(v[e]) / len(v[e]) for e in es if v.get(e)) / len(es)
            T = correct(R, t_s, f_s)
            ts = f"{t_s:.3f}" if t_s is not None else "  n/a"
            fs = f"{f_s:.3f}" if f_s is not None else "  n/a"
            Ts = f"{T:.3f}" if T is not None else "  n/a"
            print(f"  {model:10s} {s:>3s} {ts:>6s} {fs:>6s} {R:7.3f} {Ts:>11s}")


if __name__ == "__main__":
    main()
