#!/usr/bin/env python3
"""How much does sentence repetition inflate support rate?

Recomputes support rate over *unique* sentences and reports the shift per arm.
No GPU: reads the per-sentence verdicts already on disk.
"""
import collections
import glob
import json
import os
import re
import statistics as st

from src.data import reported_entities
from src.text import is_claim

SPLIT = os.environ.get("SPLIT", "test")
KEEP = reported_entities()

WS = re.compile(r"\s+")
TEXT_KEYS = ("sentence", "text", "hypothesis")
SUP_KEYS = ("supported", "is_supported", "support")


def pick_text(d):
    for k in TEXT_KEYS:
        if isinstance(d.get(k), str):
            return k
    raise KeyError(f"no sentence-text key; keys: {sorted(d)}")


def pick_sup(d):
    for k in SUP_KEYS:
        if isinstance(d.get(k), bool):
            return k
    for k in SUP_KEYS:
        if k in d:
            return k
    raise KeyError(f"no supported-flag key; keys: {sorted(d)}")


def norm(s):
    return WS.sub(" ", s.strip().lower())


def main():
    rows = collections.defaultdict(lambda: {"raw": [], "ded": [], "dup": []})
    tkey = skey = None
    drift = 0

    for path in sorted(glob.glob("outputs/verdicts_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["entity"] not in KEEP:
                continue
            sents = r.get("sentences")
            if not sents:
                continue
            if tkey is None:
                tkey, skey = pick_text(sents[0]), pick_sup(sents[0])
                print(f"reporting split: {SPLIT} ({len(KEEP)} entities)")
                print(f"sentence key {tkey!r}, supported key {skey!r}\n")

            # drift check on the UNFILTERED list: the stored support_rate was
            # computed over all sentences, so comparing a claim-filtered rate
            # to it would differ by construction and validate nothing
            all_rate = sum(bool(x[skey]) for x in sents) / len(sents)
            if r.get("support_rate") is not None and abs(all_rate - r["support_rate"]) > 1e-6:
                drift += 1

            sents = [x for x in sents if is_claim(x[tkey], r["entity"])]
            if not sents:
                continue

            seen = {}
            for s in sents:
                seen.setdefault(norm(s[tkey]), s)
            uniq = list(seen.values())

            raw = sum(bool(s[skey]) for s in sents) / len(sents)
            ded = sum(bool(s[skey]) for s in uniq) / len(uniq)

            a = rows[r["decoding"]]
            a["raw"].append(raw)
            a["ded"].append(ded)
            a["dup"].append(1 - len(uniq) / len(sents))

    if drift:
        print(f"WARNING: recomputed raw rate differs from stored support_rate "
              f"on {drift} records — the flag key may be wrong\n")

    print(f"{'arm':14s} {'n':>5s} {'raw':>7s} {'dedup':>7s} {'delta':>7s} {'dup frac':>9s}")
    order = sorted(rows.items(),
                   key=lambda kv: st.mean(kv[1]["raw"]) - st.mean(kv[1]["ded"]),
                   reverse=True)
    for arm, a in order:
        raw, ded = st.mean(a["raw"]), st.mean(a["ded"])
        print(f"{arm:14s} {len(a['raw']):5d} {raw:7.3f} {ded:7.3f} "
              f"{raw - ded:+7.3f} {st.mean(a['dup']):9.3f}")


if __name__ == "__main__":
    main()
