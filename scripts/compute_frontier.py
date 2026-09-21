#!/usr/bin/env python3
"""Factuality against inference compute, per model.

Every arm scored under the zeroed convention (degenerate = 0.0) so the
denominators match. Compute is mean compute_tokens = generated tokens x beams
(x samples for sc_k5), relative to greedy.
"""
import collections
import glob
import json
import os
import statistics as st

from src.data import reported_entities
from src.metrics.calibration import correct, load_labels, rates

SPLIT = os.environ.get("SPLIT", "test")
KEEP = reported_entities()

sup = collections.defaultdict(list)
for p in glob.glob("outputs/verdicts_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        if r["entity"] not in KEEP:
            continue
        v = r["support_rate"]
        sup[(r["model"], r["decoding"])].append(0.0 if v is None else v)

comp = collections.defaultdict(list)
for p in glob.glob("outputs/gen_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        if r["entity"] not in KEEP:
            continue
        c = r.get("compute_tokens")
        if c:
            comp[(r["model"], r["decoding"])].append(c)

sel = {}
for p in glob.glob("outputs/scchoice_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        if r["entity"] not in KEEP:
            continue
        sel[(r["model"], r["seed"], r["entity"])] = r["selected_index"]

rates = collections.defaultdict(dict)
for p in glob.glob("outputs/scverdicts_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        if r["entity"] not in KEEP:
            continue
        v = r["support_rate"]
        rates[(r["model"], r["seed"], r["entity"])][int(r["decoding"].split("#")[1])] = \
            0.0 if v is None else v

sc = collections.defaultdict(lambda: collections.defaultdict(list))
for k, d in rates.items():
    v = [d[j] for j in sorted(d)]
    sc[k[0]]["sc_k5 (selected)"].append(d.get(sel.get(k), 0.0))
    sc[k[0]]["sc_k5 (one sample)"].append(st.mean(v))
    sc[k[0]]["sc_k5 (oracle best-5)"].append(max(v))

TPR, FPR = rates(load_labels())

print(f"reporting split: {SPLIT} ({len(KEEP)} entities)")
print(f"calibrated with TPR {TPR:.3f}, FPR {FPR:.3f}; per-1x uses the calibrated rate")

for model in ("gpt2", "llama-1b", "llama-3b"):
    base = st.mean(comp[(model, "greedy")])
    rows = []
    for (m, arm), v in sup.items():
        if m != model or arm == "sc_k5":
            continue
        rows.append((arm, len(v), st.mean(comp[(m, arm)]) / base, st.mean(v)))
    c5 = st.mean(comp[(model, "sc_k5")]) / base
    for label, v in sc[model].items():
        rows.append((label, len(v), c5 / 5 if "one sample" in label else c5, st.mean(v)))
    rows.sort(key=lambda r: r[2])

    print(f"\n{model}")
    print(f"{'arm':22s} {'n':>4s} {'compute':>8s} {'raw':>7s} {'calib':>7s} {'per 1x':>8s}")
    print("-" * 62)
    for arm, n, c, s in rows:
        T = correct(s, TPR, FPR)
        # Below the false-positive floor the calibrated rate is not different
        # from zero, so support-per-unit-compute would divide a quantity that
        # is not there. Suppressed rather than printed as a small number.
        per = "       -" if s <= FPR else f"{T / c:8.3f}"
        flag = "  <- calibrates to zero" if s <= FPR else ""
        print(f"{arm:22s} {n:4d} {c:7.2f}x {s:7.3f} {T:7.3f} {per}{flag}")

print("\nper-1x is suppressed where the calibrated rate is 0.000 - dividing a")
print("quantity that is not different from zero. scripts/calibrate.py applies a")
print("stricter test, flagging arms whose 95% bootstrap CI includes zero, so a")
print("few arms priced here are still not distinguishable from zero there.")
