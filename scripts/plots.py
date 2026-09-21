#!/usr/bin/env python3
"""The factuality-diversity Pareto plot, one panel per model.

    PYTHONPATH=. python scripts/plots.py

Proposal success criterion 2: "complete factuality-diversity Pareto plots per
model size with seed error bars". Writes figures/pareto.pdf.

Axes: distinct-2 on the 128-token prefix against calibrated support rate.
Calibrated, not raw, because levels are what the axis shows and raw is inflated
by the verifier's false positives (DECISIONS 2026-09-21).

Error bars are 95% bootstrap CIs over PROMPTS, the unit of analysis. Seeds are
averaged within a prompt first, so seed variance is absorbed rather than
ignored; the seed-level spread is reported in the analysis table.

Arms whose support calibrates to zero are drawn hollow. They are not ranked
against each other - the verifier produces that much support from sentences a
human says are unsupported - so a filled marker would overstate what is known.
"""
from __future__ import annotations

import os
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis import KEEP, MODELS, per_prompt
from src.metrics.calibration import correct, load_labels
from src.metrics.calibration import rates as label_rates

FIG = "figures/pareto.pdf"
B = 2000
INK, MUTED, ACCENT = "#1a1a1a", "#6b6b6b", "#2a6fb0"


def ci(vals, rng, f=lambda xs: st.mean(xs)):
    n = len(vals)
    boot = sorted(f([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return boot[int(.025 * B)], boot[int(.975 * B)]


def main() -> None:
    tpr, fpr = label_rates(load_labels())
    pp = per_prompt()
    rng = random.Random(0)

    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.8), sharey=False)

    for ax, model in zip(axes, MODELS):
        arms = sorted({a for (m, a) in pp if m == model})
        pts = []
        for arm in arms:
            ents = sorted(set(pp[(model, arm)]) & KEEP)
            sup = [pp[(model, arm)][e].get("support", 0.0) for e in ents]
            div = [pp[(model, arm)][e]["dist2"] for e in ents
                   if "dist2" in pp[(model, arm)][e]]
            if not div:
                continue
            R = st.mean(sup)
            y = correct(R, tpr, fpr)
            x = st.mean(div)
            ylo, yhi = ci(sup, rng)
            xlo, xhi = ci(div, rng)
            pts.append({"arm": arm, "x": x, "y": y, "zero": R <= fpr,
                        "xerr": [[x - xlo], [xhi - x]],
                        "yerr": [[max(0.0, y - correct(ylo, tpr, fpr) or 0.0)],
                                 [max(0.0, (correct(yhi, tpr, fpr) or 0.0) - y)]]})

        for p in pts:
            ax.errorbar(p["x"], p["y"], xerr=p["xerr"], yerr=p["yerr"],
                        fmt="o", ms=9, capsize=3, lw=1.2,
                        color=ACCENT if not p["zero"] else MUTED,
                        mfc="white" if p["zero"] else ACCENT,
                        mec=ACCENT if not p["zero"] else MUTED, zorder=3)
            ax.annotate(p["arm"], (p["x"], p["y"]), textcoords="offset points",
                        xytext=(6, 6), fontsize=10, color=INK)

        # Pareto front over the arms that are actually distinguishable from zero
        real = sorted([p for p in pts if not p["zero"]], key=lambda p: p["x"])
        front, best = [], -1.0
        for p in reversed(real):            # rightmost first: more diverse
            if p["y"] > best:
                front.append(p)
                best = p["y"]
        if len(front) > 1:
            ax.plot([p["x"] for p in front], [p["y"] for p in front],
                    "--", color=ACCENT, lw=1.2, alpha=.6, zorder=2)

        ax.set_title(model, color=INK)
        ax.set_xlabel("distinct-2 (first 128 tokens)")
        ax.set_ylabel("calibrated support rate")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(color="#e2e2e2", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)

    fig.suptitle("Factuality against diversity, by model scale "
                 "(hollow = calibrates to zero)", y=1.04, color=INK)
    os.makedirs("figures", exist_ok=True)
    fig.savefig(FIG)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
