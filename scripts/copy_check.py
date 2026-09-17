#!/usr/bin/env python3
"""Is high support just verbatim copying of the reference page?

For every scored sentence, the fraction of its 8-grams that appear verbatim in
that entity's Wikipedia page. Reported separately for supported and unsupported
sentences: if supported sentences are near-verbatim copies, support rate is
measuring memorisation, not factual generation.
"""
import collections
import glob
import json
import os
import statistics as st

from src.data import load_pages, reported_entities
# One tokenizer in the project, same rule as the diversity metrics. A second
# copy here would drift the moment either is changed.
from src.metrics.diversity import tokenize_words

N = 8
SPLIT = os.environ.get("SPLIT", "test")
KEEP = reported_entities()


def grams(t, n=N):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


pages = load_pages()
page_grams = {e: grams(tokenize_words(" ".join(p) if isinstance(p, list) else p))
              for e, p in pages.items()}

acc = collections.defaultdict(list)
short = 0
for path in glob.glob("outputs/verdicts_*.jsonl"):
    for line in open(path):
        r = json.loads(line)
        pg = page_grams.get(r["entity"])
        if not pg or not r.get("sentences") or r["entity"] not in KEEP:
            continue
        for s in r["sentences"]:
            g = grams(tokenize_words(s["sentence"]))
            if not g:
                short += 1
                continue
            acc[(r["model"], r["decoding"], bool(s["supported"]))].append(
                len(g & pg) / len(g))

print(f"reporting split: {SPLIT} ({len(KEEP)} entities)")
print(f"sentences under {N} words, skipped: {short}\n")
print(f"{'model':10s} {'arm':14s} {'supported':>10s} {'unsupported':>12s} {'gap':>7s}")
print("-" * 58)
for model in ("gpt2", "llama-1b", "llama-3b"):
    arms = sorted({a for (m, a, _) in acc if m == model})
    for arm in arms:
        sup = acc.get((model, arm, True), [])
        uns = acc.get((model, arm, False), [])
        if not sup or not uns:
            continue
        a, b = st.mean(sup), st.mean(uns)
        print(f"{model:10s} {arm:14s} {a:10.3f} {b:12.3f} {a - b:+7.3f}")
