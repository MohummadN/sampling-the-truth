#!/usr/bin/env python3
"""Specificity-adjusted support rate.

A sentence counts as supported only if the true page entails it AND the wrong
page (same stratum) does not. The gap between raw and adjusted support rate is
the share of each arm's 'facts' that are generic enough to fit anyone.
"""
import collections
import glob
import json
import statistics as st

wrong = {}
for p in glob.glob("outputs/mismatch_*.jsonl"):
    for line in open(p):
        d = json.loads(line)
        wrong[(d["model"], d["decoding"], d["seed"], d["entity"],
               d["sent_index"])] = bool(d["supported_wrong"])

raw = collections.defaultdict(list)
adj = collections.defaultdict(list)
gen_n = collections.Counter()
gen_d = collections.Counter()

for p in glob.glob("outputs/verdicts_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        k = (r["model"], r["decoding"])
        sents = r.get("sentences")
        if not sents:
            raw[k].append(0.0)
            adj[k].append(0.0)
            continue
        n_sup = n_adj = 0
        for i, s in enumerate(sents):
            if not s.get("supported"):
                continue
            n_sup += 1
            gen_d[k] += 1
            if wrong.get((r["model"], r["decoding"], r["seed"], r["entity"], i)):
                gen_n[k] += 1
            else:
                n_adj += 1
        raw[k].append(n_sup / len(sents))
        adj[k].append(n_adj / len(sents))

print(f"{'model':10s} {'arm':14s} {'n':>4s} {'raw':>7s} {'adjusted':>9s} "
      f"{'drop':>7s} {'generic %':>10s}")
print("-" * 66)
for model in ("gpt2", "llama-1b", "llama-3b"):
    rows = [(a, v) for (m, a), v in raw.items() if m == model]
    for arm, v in sorted(rows, key=lambda r: -st.mean(r[1])):
        k = (model, arm)
        a_mean = st.mean(adj[k])
        g = gen_n[k] / gen_d[k] if gen_d[k] else 0.0
        print(f"{model:10s} {arm:14s} {len(v):4d} {st.mean(v):7.3f} {a_mean:9.3f} "
              f"{a_mean - st.mean(v):+7.3f} {g * 100:9.1f}%")
    print()
