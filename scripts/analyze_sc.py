#!/usr/bin/env python3
"""Self-consistency: does it help because the pool is good or the selector is?

pool   = mean support rate over the 5 samples  (= expected score of one sample)
sc     = support rate of the medoid the selector chose
oracle = best of the 5  (upper bound any selector could reach)
worst  = worst of the 5 (lower bound)

Degenerate samples score 0.0, per the zeroed convention in DECISIONS.md.
"""
import collections
import glob
import json
import statistics as st

sel = {}
for p in glob.glob("outputs/scchoice_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        sel[(r["model"], r["seed"], r["entity"])] = r["selected_index"]

rates = collections.defaultdict(dict)
for p in glob.glob("outputs/scverdicts_*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        j = int(r["decoding"].split("#")[1])
        v = r["support_rate"]
        rates[(r["model"], r["seed"], r["entity"])][j] = 0.0 if v is None else v

out = collections.defaultdict(lambda: collections.defaultdict(list))
incomplete = 0
no_selection = 0
for k, d in rates.items():
    if len(d) < 5:
        incomplete += 1
    j = sel.get(k)
    if j is None or j not in d:
        # A missing selection used to score 0.0, which silently depressed the
        # sc column and made the selector look worse than the pool. Drop the
        # entity instead: pool/sc/oracle/worst must describe the same entities
        # or the comparison between them is meaningless.
        no_selection += 1
        continue
    v = [d[i] for i in sorted(d)]
    o = out[k[0]]
    o["pool"].append(st.mean(v))
    o["sc"].append(d[j])
    o["oracle"].append(max(v))
    o["worst"].append(min(v))

print(f"entities with fewer than 5 scored samples: {incomplete}")
print(f"entities dropped for a missing selection: {no_selection}\n")
hdr = f"{'model':10s} {'n':>4s} {'pool':>7s} {'sc':>7s} {'oracle':>7s} {'worst':>7s} {'sc-pool':>8s} {'headroom':>9s}"
print(hdr)
print("-" * len(hdr))
for m in sorted(out):
    o = out[m]
    pool, sc = st.mean(o["pool"]), st.mean(o["sc"])
    orc, wst = st.mean(o["oracle"]), st.mean(o["worst"])
    print(f"{m:10s} {len(o['pool']):4d} {pool:7.3f} {sc:7.3f} {orc:7.3f} {wst:7.3f} "
          f"{sc - pool:+8.3f} {orc - sc:9.3f}")
