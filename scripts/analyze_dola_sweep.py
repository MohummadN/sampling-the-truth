#!/usr/bin/env python3
"""dola_layers sensitivity: 'low' (the frozen grid) vs 'high' (dev-only sweep).

Paired over the 20 dev entities, seeds averaged within an entity, bootstrap CI
over entities. Degenerate outputs score 0.0, per the zeroed convention.
"""
import collections
import glob
import json
import random
import statistics as st

from src.data import load_entities

PAIRS = [("dola", "dola_high"), ("dola_nucleus", "dola_nucleus_high")]
DEV = {e["entity"] for e in load_entities() if e["split"] == "dev"}


def read(pattern, arms):
    out = collections.defaultdict(list)
    for p in glob.glob(pattern):
        for line in open(p):
            r = json.loads(line)
            if r["decoding"] not in arms or r["entity"] not in DEV:
                continue
            v = r["support_rate"]
            out[(r["model"], r["decoding"], r["entity"])].append(
                0.0 if v is None else v)
    return out


low = read("outputs/verdicts_*.jsonl", {a for a, _ in PAIRS})
high = read("outputs/dolasweep_verdicts_*.jsonl", {b for _, b in PAIRS})
if not high:
    raise SystemExit("no dolasweep verdicts found — is the array still running?")

print(f"dev entities: {len(DEV)}\n")
print(f"{'model':10s} {'comparison':30s} {'n':>3s} {'low':>7s} {'high':>7s} "
      f"{'delta':>7s} {'95% CI':>18s}")
print("-" * 90)

random.seed(0)
for model in ("gpt2", "llama-1b", "llama-3b"):
    for a, b in PAIRS:
        rows = []
        for e in sorted(DEV):
            la, hb = low.get((model, a, e)), high.get((model, b, e))
            if not la or not hb:
                continue
            rows.append((st.mean(la), st.mean(hb)))
        if not rows:
            print(f"{model:10s} {a + ' vs ' + b:30s}   0   (no paired records)")
            continue
        lm = st.mean(r[0] for r in rows)
        hm = st.mean(r[1] for r in rows)
        boot = sorted(
            st.mean([x[1] - x[0] for x in random.choices(rows, k=len(rows))])
            for _ in range(10000))
        lo, hi = boot[250], boot[9750]
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"{model:10s} {a + ' vs ' + b:30s} {len(rows):3d} {lm:7.3f} {hm:7.3f} "
              f"{hm - lm:+7.3f} [{lo:+.3f}, {hi:+.3f}]{flag}")
    print()
print("* = 95% CI excludes zero. n should be 20; fewer means missing pairs.")
