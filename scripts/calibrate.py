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
import os
import random

from src.data import load_entities, reported_entities
from scripts.f1_teaser import load_support_per_entity

from src.metrics.calibration import correct, load_labels, rates

B = 10000        # the count §3.6 states; analysis.py uses the same
THETA = 0.5
# Reported numbers are the 80 test entities (DECISIONS 2026-09-07). The
# labels themselves come from dev, so TPR/FPR are estimated on one split and
# applied to the other - which is the point of a prevalence-independent
# correction, and keeps the reported rates held out.
SPLIT = os.environ.get("SPLIT", "test")
KEEP = reported_entities()
MODELS = ("gpt2", "llama-1b", "llama-3b")


def load_observed():
    """Per-entity support per arm, plus each reported entity's stratum.

    Support comes from scripts.f1_teaser so the calibrated table covers the
    same nine arms the figures and Table 2 do. Globbing verdicts_*.jsonl here
    quietly dropped sc_k5, whose medoid verdicts are in scverdicts_*.jsonl and
    need the scchoice_*.jsonl join.

    Strata come from the frozen entity list rather than from whichever records
    happened to load, so the bootstrap's prompt vector is the reported split
    itself even if an arm is missing an entity.
    """
    acc = load_support_per_entity()
    strat = {e["entity"]: str(e["stratum"]) for e in load_entities()
             if e["entity"] in KEEP}
    return acc, strat


def calibrated_intervals():
    """{(model, arm): (raw, calibrated, lo, hi)}.

    Table 2 imports this instead of correcting a point estimate of its own, so
    the dagger in the table and the count in the text apply one test to one set
    of numbers.
    """
    labels = load_labels()
    acc, strat = load_observed()
    ents = sorted(strat)
    n = len(ents)
    tpr, fpr = rates(labels)

    vec = {k: [sum(pe[e]) / len(pe[e]) if pe.get(e) else 0.0 for e in ents]
           for k, pe in acc.items()}

    rng = random.Random(0)
    boots = collections.defaultdict(list)
    for _ in range(B):
        ls = rng.choices(labels, k=len(labels))
        t_b, f_b = rates(ls)
        idx = [rng.randrange(n) for _ in range(n)]
        for k, v in vec.items():
            R_b = sum(v[i] for i in idx) / n
            c = correct(R_b, t_b, f_b)
            if c is not None:
                boots[k].append(c)

    out = {}
    for k, v in vec.items():
        R = sum(v) / n
        bs = sorted(boots[k])
        lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
        out[k] = (R, correct(R, tpr, fpr), lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="beam4", help="arm for the stratum table")
    args = ap.parse_args()

    labels = load_labels()
    acc, strat = load_observed()
    ents = sorted(strat)

    tpr, fpr = rates(labels)
    print(f"reporting split: {SPLIT} ({len(KEEP)} entities); labels are from dev")
    print(f"verifier at theta={THETA}: TPR {tpr:.3f}  FPR {fpr:.3f}  "
          f"(n={len(labels)} labels)")
    print(f"calibration: T = (R - {fpr:.3f}) / ({tpr:.3f} - {fpr:.3f})"
          f"  ->  T ~ {1/(tpr-fpr):.2f} R - {fpr/(tpr-fpr):.3f}")
    print(f"false-positive floor: any arm at or below R = {fpr:.3f} is "
          f"indistinguishable from zero true support\n")

    ci = calibrated_intervals()

    n_floor = 0
    for model in MODELS:
        rows = [(arm, R, T, lo, hi)
                for (m, arm), (R, T, lo, hi) in ci.items() if m == model]
        rows.sort(key=lambda r: -r[1])
        print(f"\n{model}")
        print(f"  {'arm':14s} {'raw':>7s} {'calibrated':>11s} {'95% CI':>18s}")
        print("  " + "-" * 54)
        n_model = 0
        for arm, R, T, lo, hi in rows:
            zero = lo <= 0.0005          # the interval includes zero
            n_model += zero
            flag = "   <- includes zero" if zero else ""
            print(f"  {arm:14s} {R:7.3f} {T:11.3f} "
                  f"[{lo:.3f}, {hi:.3f}]{flag}")
        print(f"  {n_model} of {len(rows)} arms indistinguishable from zero")
        n_floor += n_model
    print(f"\ntotal arms whose calibrated interval includes zero: {n_floor}")

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
