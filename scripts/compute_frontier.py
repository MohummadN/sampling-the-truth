#!/usr/bin/env python3
"""Factuality against inference compute, per model.

Every arm scored under the zeroed convention (degenerate = 0.0) so the
denominators match. Compute is mean compute_tokens = generated tokens x beams
(x samples for sc_k5), relative to greedy.
"""
import collections
import glob
import json
import statistics as st

sup = collections.defaultdict(list)
for p in glob.glob("outputs/verdicts_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        v = r["support_rate"]
        sup[(r["model"], r["decoding"])].append(0.0 if v is None else v)

comp = collections.defaultdict(list)
for p in glob.glob("outputs/gen_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        c = r.get("compute_tokens")
        if c:
            comp[(r["model"], r["decoding"])].append(c)

sel = {}
for p in glob.glob("outputs/scchoice_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        sel[(r["model"], r["seed"], r["entity"])] = r["selected_index"]

rates = collections.defaultdict(dict)
for p in glob.glob("outputs/scverdicts_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        v = r["support_rate"]
        rates[(r["model"], r["seed"], r["entity"])][int(r["decoding"].split("#")[1])] = \
            0.0 if v is None else v

sc = collections.defaultdict(lambda: collections.defaultdict(list))
for k, d in rates.items():
    v = [d[j] for j in sorted(d)]
    sc[k[0]]["sc_k5 (selected)"].append(d.get(sel.get(k), 0.0))
    sc[k[0]]["sc_k5 (one sample)"].append(st.mean(v))
    sc[k[0]]["sc_k5 (oracle best-5)"].append(max(v))

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
    print(f"{'arm':22s} {'n':>4s} {'compute':>8s} {'support':>8s} {'per 1x':>8s}")
    print("-" * 54)
    for arm, n, c, s in rows:
        print(f"{arm:22s} {n:4d} {c:7.2f}x {s:8.3f} {s / c:8.3f}")
