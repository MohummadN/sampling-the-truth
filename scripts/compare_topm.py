#!/usr/bin/env python3
"""Does retrieval depth change the results? top-m 5 vs 20 on the dev entities.

Paired over the 20 dev entities, seeds averaged within an entity, bootstrap CI
over entities. Degenerate outputs score 0.0.

If support rates rise only slightly and rise equally across arms, the top-5
protocol costs level but not ranking, and the limitation is bounded.
"""
import collections
import glob
import json
import random
import statistics as st

from src.data import load_entities

DEV = {e["entity"] for e in load_entities() if e["split"] == "dev"}


def read(pattern):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob(pattern):
        for line in open(p):
            r = json.loads(line)
            if r["entity"] not in DEV:
                continue
            v = r["support_rate"]
            out[(r["model"], r["decoding"])][r["entity"]].append(
                0.0 if v is None else v)
    return out


base = read("outputs/verdicts_*.jsonl")
wide = read("outputs/topm20_*.jsonl")
if not wide:
    raise SystemExit("no topm20 verdicts yet — is the array still running?")

rng = random.Random(0)
print(f"{'model':10s} {'arm':14s} {'n':>3s} {'top5':>7s} {'top20':>7s} "
      f"{'delta':>7s} {'95% CI':>18s}")
print("-" * 74)
deltas = []
for key in sorted(wide):
    b, w = base.get(key), wide[key]
    if not b:
        continue
    ents = sorted(set(b) & set(w) & DEV)
    if not ents:
        continue
    pairs = [(st.mean(b[e]), st.mean(w[e])) for e in ents]
    m5 = st.mean(p[0] for p in pairs)
    m20 = st.mean(p[1] for p in pairs)
    boot = sorted(st.mean([x[1] - x[0] for x in rng.choices(pairs, k=len(pairs))])
                  for _ in range(10000))
    lo, hi = boot[250], boot[9750]
    flag = "" if lo <= 0 <= hi else "  *"
    deltas.append(m20 - m5)
    print(f"{key[0]:10s} {key[1]:14s} {len(ents):3d} {m5:7.3f} {m20:7.3f} "
          f"{m20 - m5:+7.3f} [{lo:+.3f}, {hi:+.3f}]{flag}")

print(f"\nmean shift across arms: {st.mean(deltas):+.3f}   "
      f"spread (sd): {st.pstdev(deltas):.3f}")
print("* = 95% CI excludes zero. A uniform positive shift means top-5 costs "
      "level but not ranking.")
