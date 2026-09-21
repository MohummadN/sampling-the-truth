#!/usr/bin/env python3
"""Every headline table: factuality, diversity, fluency and compute together.

    PYTHONPATH=. python scripts/analysis.py

Reads verdicts_*, metrics_*, gen_* and scchoice_*/scverdicts_*. Reports the 80
test entities (DECISIONS 2026-09-07).

Conventions this script enforces, all of them decided earlier:
  - the unit is the PROMPT: seeds are averaged within an entity first, so the
    three seeds are a precision statement and not three extra samples
  - degenerate generations score 0.0, the zeroed convention, so every arm has
    the same 80 pairs and the bootstrap stays paired
  - length-sensitive diversity is read off the 128-token prefix
  - support is reported raw and Rogan-Gladen calibrated; arms calibrating to
    zero are marked, because dividing or ranking them is meaningless
  - the reference arm is nucleus0.9: the proposal asks whether anything beats
    a well-tuned nucleus baseline, so every comparison answers that question
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import statistics as st

from src.data import reported_entities
from src.metrics.calibration import correct, load_labels
from src.metrics.calibration import rates as label_rates
from src.metrics.diversity import self_bleu

SPLIT = os.environ.get("SPLIT", "test")
KEEP = reported_entities()
REF = "nucleus0.9"
B = 2000
MODELS = ("gpt2", "llama-1b", "llama-3b")


def read(pattern, keep_entity=True):
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            r = json.loads(line)
            if keep_entity and r.get("entity") not in KEEP:
                continue
            yield r


def per_prompt():
    """(model, arm) -> entity -> {metric: value}, seeds averaged within entity."""
    acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))

    for r in read("outputs/verdicts_*.jsonl"):
        v = r["support_rate"]
        acc[(r["model"], r["decoding"])][r["entity"]]["support"].append(
            0.0 if v is None else v)

    for r in read("outputs/metrics_*.jsonl"):
        d = acc[(r["model"], r["decoding"])][r["entity"]]
        for src, dst in (("rep_4_prefix", "rep4"), ("distinct_2_prefix", "dist2"),
                         ("loop_severity_prefix", "loop"), ("perplexity", "ppl"),
                         ("n_words", "words")):
            if r.get(src) is not None:
                d[dst].append(r[src])

    for r in read("outputs/gen_*.jsonl"):
        c = r.get("compute_tokens")
        if c:
            acc[(r["model"], r["decoding"])][r["entity"]]["compute"].append(c)

    # sc_k5 needs assembling. Its parent record carries samples rather than
    # text, so metrics_* has no row for it and verdicts_* excludes it; left
    # alone every field would default to 0.0 and invent an arm out of nothing.
    # The selected sample IS the arm; the mean over five is what one sample of
    # the same sampler buys, and the cost of both comes from the parent.
    sel = {}
    for r in read("outputs/scchoice_*.jsonl"):
        sel[(r["model"], r["seed"], r["entity"])] = r.get("selected_index")
    sup = collections.defaultdict(dict)
    for r in read("outputs/scverdicts_*.jsonl"):
        j = int(r["decoding"].split("#")[1])
        v = r["support_rate"]
        sup[(r["model"], r["seed"], r["entity"])][j] = 0.0 if v is None else v
    met = collections.defaultdict(dict)
    for r in read("outputs/metrics_*.jsonl"):
        if not str(r["decoding"]).startswith("sc_k5#"):
            continue
        met[(r["model"], r["seed"], r["entity"])][int(r["decoding"].split("#")[1])] = r

    for (m, seed, ent), per_j in sup.items():
        chosen = sel.get((m, seed, ent))
        rows_j = met.get((m, seed, ent), {})
        cost = acc[(m, "sc_k5")][ent]["compute"]
        for label, js in (("sc_k5 (selected)", [chosen] if chosen in per_j else []),
                          ("sc_k5 (one sample)", sorted(per_j))):
            if not js:
                continue
            d = acc[(m, label)][ent]
            d["support"].append(st.mean([per_j[j] for j in js]))
            for src, dst in (("rep_4_prefix", "rep4"), ("distinct_2_prefix", "dist2"),
                             ("loop_severity_prefix", "loop"), ("perplexity", "ppl"),
                             ("n_words", "words")):
                vals = [rows_j[j][src] for j in js
                        if j in rows_j and rows_j[j].get(src) is not None]
                if vals:
                    d[dst].append(st.mean(vals))
            if cost:
                c = st.mean(cost)
                d["compute"].append(c / 5 if "one sample" in label else c)

    # the raw pieces are now represented by the two assembled rows
    for k in [k for k in acc if k[1] == "sc_k5" or str(k[1]).startswith("sc_k5#")]:
        del acc[k]

    return {k: {e: {m: st.mean(v) for m, v in d.items()}
                for e, d in ents.items()} for k, ents in acc.items()}


def selfbleu_by_prompt():
    """Self-BLEU across the three seeds of one prompt: how much does re-running
    change the output? Deterministic arms score ~1.0 by construction."""
    texts = collections.defaultdict(list)
    for r in read("outputs/gen_*.jsonl"):
        if isinstance(r.get("text"), str) and r["text"].strip():
            texts[(r["model"], r["decoding"], r["entity"])].append(r["text"])
    out = collections.defaultdict(dict)
    for (m, arm, ent), ts in texts.items():
        if len(ts) >= 2:
            out[(m, arm)][ent] = self_bleu(ts)
    return out


def paired_ci(a: list[float], b: list[float], rng, lower_is_better=False):
    """Bootstrap the mean difference a - b over prompts, resampling PROMPTS.
    The same resample indexes both arms, which is what makes it paired."""
    n = len(a)
    diffs = sorted(
        st.mean([a[i] - b[i] for i in (rng.randrange(n) for _ in range(n))])
        for _ in range(B))
    lo, hi = diffs[int(.025 * B)], diffs[int(.975 * B)]
    d = st.mean(a) - st.mean(b)
    sig = "*" if (lo > 0 or hi < 0) else " "
    return d, lo, hi, sig


def main():
    tpr, fpr = label_rates(load_labels())
    pp = per_prompt()
    sb = selfbleu_by_prompt()
    rng = random.Random(0)

    print(f"reporting split: {SPLIT} ({len(KEEP)} entities); seeds averaged "
          f"within prompt; reference arm {REF}")
    print(f"calibration TPR {tpr:.3f} FPR {fpr:.3f}; arms at or below "
          f"R={fpr:.3f} calibrate to zero\n")

    for model in MODELS:
        arms = sorted({a for (m, a) in pp if m == model},
                      key=lambda a: -st.mean([d.get("support", 0.0)
                                              for d in pp[(model, a)].values()]))
        print(f"=== {model} " + "=" * 62)
        print(f"{'arm':14s}{'n':>4s}{'support':>9s}{'calib':>8s}{'rep4':>7s}"
              f"{'dist2':>7s}{'selfBLEU':>10s}{'ppl':>9s}{'words':>7s}{'cost':>7s}")
        # greedy = 1x, established before the loop: arms are printed in support
        # order, so computing it inline would leave it unset for the top rows
        gd = pp.get((model, "greedy"), {})
        base = st.mean([d.get("compute", 0.0) for d in gd.values()]) if gd else 0.0
        rows = {}
        for arm in arms:
            ents = sorted(set(pp[(model, arm)]) & KEEP)
            g = lambda m, default=0.0: [pp[(model, arm)][e].get(m, default) for e in ents]
            for metric in ("support", "rep4", "dist2", "ppl"):
                have = sum(1 for e in ents if metric in pp[(model, arm)][e])
                if have < len(ents):
                    print(f"  ! {arm}: {metric} missing for {len(ents)-have} of "
                          f"{len(ents)} prompts, counted as 0.0")
            sup, comp = g("support"), g("compute")
            rows[arm] = {"ents": ents, "support": sup, "rep4": g("rep4"),
                         "dist2": g("dist2"), "ppl": g("ppl"),
                         "sb": [sb[(model, arm)].get(e, float("nan")) for e in ents]}
            R = st.mean(sup)
            T = correct(R, tpr, fpr)
            sbv = [x for x in rows[arm]["sb"] if x == x]
            flag = " <-0" if R <= fpr else ""
            print(f"{arm:14s}{len(ents):4d}{R:9.3f}{T:8.3f}{st.mean(g('rep4')):7.3f}"
                  f"{st.mean(g('dist2')):7.3f}{(st.mean(sbv) if sbv else float('nan')):10.3f}"
                  f"{st.mean(g('ppl')):9.1f}{st.mean(g('words')):7.0f}"
                  f"{(st.mean(comp)/base if base else 1):6.2f}x{flag}")

        if REF not in rows:
            print("  (no reference arm)\n")
            continue
        print(f"\n  paired bootstrap vs {REF}, over {len(rows[REF]['ents'])} prompts "
              f"(* = 95% CI excludes zero)")
        print(f"  {'arm':14s}{'d support':>22s}{'d rep4':>22s}{'d dist2':>22s}")
        for arm in arms:
            if arm == REF:
                continue
            cells = []
            for met in ("support", "rep4", "dist2"):
                a = rows[arm][met]
                b = rows[REF][met]
                if len(a) != len(b):
                    cells.append(f"{'n/a':>22s}")
                    continue
                d, lo, hi, sig = paired_ci(a, b, rng)
                cells.append(f"{d:+7.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
            print(f"  {arm:14s}" + "".join(cells))
        print()


if __name__ == "__main__":
    main()
