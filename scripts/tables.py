#!/usr/bin/env python3
"""Generate T1 (dataset statistics) and T2 (main results) as LaTeX.

Writes paper/tables.tex for \\input into the paper. Numbers come from the same
loaders the figures use, so a table can never disagree with a plot.
"""
import collections
import glob
import json
import statistics as st

from src.data import load_entities, reported_entities
from scripts.f1_teaser import (FAMILY, LABEL, MODELS, TITLES,
                               load_compute, load_diversity, load_support)
from scripts.calibrate import correct, load_labels, rates

KEEP = reported_entities()
OUT = "paper/tables.tex"


def load_repetition():
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob("outputs/metrics_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if r["entity"] not in KEEP:
                continue
            arm = r["decoding"]
            if arm.startswith("sc_k5"):
                if not r.get("sc_selected"):
                    continue
                arm = "sc_k5"
            v = r.get("rep_4_prefix")
            if v is not None:
                acc[(r["model"], arm)][r["entity"]].append(v)
    return {k: st.mean(st.mean(x) for x in v.values()) for k, v in acc.items()}


def table1():
    ents = load_entities()
    by = collections.defaultdict(list)
    for e in ents:
        by[e["stratum"]].append(e)
    rows = []
    for s in sorted(by):
        g = by[s]
        sents = [e["page_sentences"] for e in g]
        n_dev = sum(1 for e in g if e["split"] == "dev")
        rows.append((s, len(g), n_dev, len(g) - n_dev,
                     min(sents), st.median(sents), max(sents)))

    n_gen = sum(1 for p in glob.glob("outputs/gen_*.jsonl") for _ in open(p))

    L = [r"\begin{table}[t]", r"  \centering", r"  \small",
         r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
         r"    \textbf{Stratum} & \textbf{$n$} & \textbf{dev} & \textbf{test} "
         r"& \textbf{min} & \textbf{med} & \textbf{max} \\",
         r"    \midrule"]
    for s, n, d, t, mn, md, mx in rows:
        L.append(f"    {s} & {n} & {d} & {t} & {mn} & {md:.0f} & {mx} \\\\")
    tot = sum(r[1] for r in rows)
    L += [r"    \midrule",
          f"    all & {tot} & {sum(r[2] for r in rows)} & "
          f"{sum(r[3] for r in rows)} & & & \\\\",
          r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{Entity pool. Strata are formed by Wikipedia page length;"
          r" \textbf{min}, \textbf{med} and \textbf{max} are page sentences."
          f" The grid produces {n_gen:,} generations"
          r" (3 models $\times$ 9 configurations $\times$ 3 seeds $\times$ 100"
          r" prompts); all reported results use the test split.}",
          r"  \label{tab:data}", r"\end{table}", ""]
    return "\n".join(L)


def table2():
    sup, div, comp = load_support(), load_diversity(), load_compute()
    rep = load_repetition()
    tpr, fpr = rates(load_labels())

    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \begin{tabular}{ll rrrr r}", r"    \toprule",
         r"    \textbf{Model} & \textbf{Configuration} & \textbf{Support} "
         r"& \textbf{Calibrated} & \textbf{Distinct-2} & \textbf{Rep-4} "
         r"& \textbf{Compute} \\", r"    \midrule"]
    for model in MODELS:
        first = True
        arms = sorted((a for a in FAMILY if (model, a) in sup),
                      key=lambda a: -sup[(model, a)])
        for a in arms:
            cal = correct(sup[(model, a)], tpr, fpr)
            cal_s = "0.000$^{\\dagger}$" if cal is not None and cal <= 0.0005 \
                else (f"{cal:.3f}" if cal is not None else "--")
            name = TITLES[model] if first else ""
            first = False
            L.append(f"    {name} & {LABEL[a]} & {sup[(model, a)]:.3f} & {cal_s} "
                     f"& {div.get((model, a), float('nan')):.3f} "
                     f"& {rep.get((model, a), float('nan')):.3f} "
                     f"& {comp.get((model, a), float('nan')):.2f}$\\times$ \\\\")
        if model != MODELS[-1]:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \caption{Main results on the test split, ordered by support rate."
          r" \textbf{Calibrated} applies the Rogan--Gladen correction using the"
          f" verifier's measured sensitivity ({tpr:.3f}) and false-positive rate"
          f" ({fpr:.3f}) from 100 human labels; $\\dagger$ marks arms at or below"
          r" the false-positive floor, indistinguishable from zero true support."
          r" Diversity and repetition are computed over the first 128 tokens to"
          r" remove the length confound. Compute is generated tokens $\times$"
          r" beams ($\times$ samples), relative to greedy.}",
          r"  \label{tab:main}", r"\end{table*}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    body = table1() + "\n" + table2()
    open(OUT, "w").write(body)
    print(f"wrote {OUT}\n")
    print(body)
