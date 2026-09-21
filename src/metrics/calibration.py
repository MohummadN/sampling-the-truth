"""Rogan-Gladen calibration of observed support rates.

The verifier is high recall, low precision: measured against the 100 manual
labels it has TPR 0.947 and FPR 0.041, and 34 of its 36 false positives are
sentences a human judged `not_addressed`. An observed rate R is therefore
inflated, and the correction is prevalence-independent, so rates estimated on
the dev labels transfer to the test arms:

    T = (R - FPR) / (TPR - FPR)

One definition, imported by scripts/calibrate.py and scripts/compute_frontier.py,
so the two can never disagree about where the false-positive floor sits.
"""
from __future__ import annotations

import csv

LAB = "outputs/labels.csv"
KEY = "outputs/labeling_key.csv"
THETA = 0.5


def load_labels(lab_path: str = LAB, key_path: str = KEY) -> list[dict]:
    """The manual labels joined to the verifier's verdict and sampling weight.

    `unsure` rows are dropped; there were none in the 100.
    """
    lab = {r["id"]: r for r in csv.DictReader(open(lab_path, encoding="utf-8"))}
    key = {r["id"]: r for r in csv.DictReader(open(key_path, encoding="utf-8"))}
    rows = []
    for i, l in lab.items():
        if l["label"] == "unsure":
            continue
        k = key[i]
        rows.append({"h": l["label"] == "supported",
                     "v": float(k["p_entail"]) >= THETA,
                     "w": float(k["weight"]),
                     "s": str(k["stratum"])})
    return rows


def rates(rows: list[dict]) -> tuple[float | None, float | None]:
    """Weighted (TPR, FPR). Weighting lifts the stratified sample to the frame."""
    pos = sum(r["w"] for r in rows if r["h"])
    neg = sum(r["w"] for r in rows if not r["h"])
    tp = sum(r["w"] for r in rows if r["h"] and r["v"])
    fp = sum(r["w"] for r in rows if not r["h"] and r["v"])
    return (tp / pos if pos else None, fp / neg if neg else None)


def correct(R: float, tpr: float | None, fpr: float | None) -> float | None:
    """Rogan-Gladen, clamped to [0, 1]. None when the rates cannot separate."""
    if tpr is None or fpr is None:
        return None
    d = tpr - fpr
    if d <= 0:
        return None
    return min(1.0, max(0.0, (R - fpr) / d))


def floor(rows: list[dict] | None = None) -> float:
    """The false-positive floor: an arm at or below this observed rate is
    indistinguishable from zero true support, because the verifier produces
    that much support from sentences a human says are not supported."""
    _, fpr = rates(rows if rows is not None else load_labels())
    return fpr
