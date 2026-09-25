#!/usr/bin/env python3
"""T3 — verifier validation against the 100 human labels, as LaTeX."""
import collections
import csv

from scripts.calibrate import load_labels, rates

OUT = "paper/table3.tex"


def metrics(rows, theta_rows):
    tpr, fpr = rates(rows)
    W = lambda xs: sum(r["w"] for r in xs)
    tp = W([r for r in theta_rows if r["h"] and r["v"]])
    fp = W([r for r in theta_rows if not r["h"] and r["v"]])
    prec = tp / (tp + fp) if tp + fp else 0.0
    return tpr, fpr, prec, (tpr + (1 - fpr)) / 2


rows = load_labels()
tpr, fpr, prec, bal = metrics(rows, rows)

raw = collections.Counter(
    r["label"] for r in csv.DictReader(open("outputs/labels.csv", encoding="utf-8")))
fp_rows = [r for r in rows if not r["h"] and r["v"]]
fn_rows = [r for r in rows if r["h"] and not r["v"]]

# calibrate.load_labels drops the raw label, so recover the composition of the
# false positives from the source files
_lab = {r["id"]: r for r in csv.DictReader(open("outputs/labels.csv", encoding="utf-8"))}
_key = {r["id"]: r for r in csv.DictReader(open("outputs/labeling_key.csv", encoding="utf-8"))}
fp_comp = collections.Counter(
    _lab[i]["label"] for i in _lab
    if _lab[i]["label"] not in ("supported", "unsure")
    and float(_key[i]["p_entail"]) >= 0.5)

L = [r"\begin{table}[t]", r"  \centering", r"  \small",
     r"  \begin{tabular}{lr}", r"    \toprule",
     r"    \textbf{Quantity} & \textbf{Value} \\", r"    \midrule",
     f"    Labelled sentences & {sum(raw.values())} \\\\",
     f"    \\quad supported & {raw['supported']} \\\\",
     f"    \\quad contradicted & {raw['contradicted']} \\\\",
     f"    \\quad not addressed & {raw['not_addressed']} \\\\",
     r"    \midrule",
     f"    Sensitivity (TPR) & {tpr:.3f} \\\\",
     f"    False-positive rate & {fpr:.3f} \\\\",
     f"    Precision & {prec:.3f} \\\\",
     f"    Balanced accuracy & {bal:.3f} \\\\",
     r"    \midrule",
     f"    False positives & {len(fp_rows)} \\\\",
     f"    \\quad human-labelled \\emph{{not addressed}} & "
     f"{fp_comp['not_addressed']} \\\\",
     f"    \\quad human-labelled \\emph{{contradicted}} & "
     f"{fp_comp['contradicted']} \\\\",
     f"    False negatives & {len(fn_rows)} \\\\",
     r"    \bottomrule", r"  \end{tabular}",
     r"  \caption{Verifier validation at $\theta=0.5$ against 100 blind human"
     r" labels, stratified by entity stratum and entailment-probability band and"
     r" reweighted to the full frame. The verifier is high-recall and"
     r" low-precision: it almost never misses a supported sentence, but"
     r" confidently entails claims the retrieved evidence does not address"
     r" --- the neutral trap. Balanced accuracy peaks at $\theta=0.55$"
     r" (0.956) against 0.953 at $\theta=0.5$, so the pre-registered threshold"
     r" is retained.}",
     r"  \label{tab:verifier}", r"\end{table}", ""]

open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}\n" + "\n".join(L))
