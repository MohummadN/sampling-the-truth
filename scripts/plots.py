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
from matplotlib.lines import Line2D

from scripts.analysis import KEEP, MODELS, per_prompt
from src.metrics.calibration import correct, load_labels
from src.metrics.calibration import rates as label_rates

FIG = "figures/pareto.pdf"
B = 2000
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e2e2"

# Ten arms is past the point where categorical colour works, so colour carries
# the FAMILY - what the decoder is doing - and marker shape separates arms
# within it. Slots 1-4 of the reference palette, in fixed order.
FAMILY = {
    "likelihood maximisation": ("#2a78d6", ["greedy", "beam4", "greedy_reppen"]),
    "temperature / nucleus":   ("#eb6834", ["temp0.7", "temp1.3", "nucleus0.9"]),
    "DoLa":                    ("#1baf7a", ["dola", "dola_nucleus"]),
    "self-consistency":        ("#eda100", ["sc_k5 (selected)", "sc_k5 (one sample)"]),
}
MARKERS = ["o", "s", "^"]
STYLE = {}
for fam, (colour, arms_) in FAMILY.items():
    for i, a in enumerate(arms_):
        STYLE[a] = (colour, MARKERS[i], fam)


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
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 5.0), sharey=True)

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

        # Pareto front first: only front arms get a direct label, which is what
        # keeps ten arms per panel readable. The rest are identified by the
        # shared legend, never by a label collision.
        real = sorted([p for p in pts if not p["zero"]], key=lambda p: p["x"])
        front, best = [], -1.0
        for p in reversed(real):            # rightmost first: more diverse
            if p["y"] > best:
                front.append(p)
                best = p["y"]
        # At most four direct labels per panel, the highest-support arms on the
        # front. A low-support front can run to seven arms, and labelling them
        # all is what produced unreadable overlap; the legend carries the rest.
        on_front = {p["arm"] for p in
                    sorted(front, key=lambda q: -q["y"])[:4]}

        if len(front) > 1:
            ax.plot([p["x"] for p in front], [p["y"] for p in front],
                    "--", color=MUTED, lw=1.1, alpha=.55, zorder=2)

        # labelled points in x order, so alternating the vertical offset below
        # separates any two whose labels would otherwise land on one line
        label_rank = {a: i for i, a in enumerate(
            [q["arm"] for q in sorted(pts, key=lambda q: q["x"])
             if q["arm"] in on_front])}
        for p in pts:
            colour, marker, _ = STYLE.get(p["arm"], (MUTED, "o", ""))
            ax.errorbar(p["x"], p["y"], xerr=p["xerr"], yerr=p["yerr"],
                        fmt=marker, ms=9, capsize=2.5, lw=1.0, ecolor=MUTED,
                        mfc="white" if p["zero"] else colour,
                        mec=colour, mew=1.6, zorder=3)
            if p["arm"] in on_front:
                # near the right edge the label would run off the axis
                right = p["x"] > 0.72
                dy = 9 if label_rank[p["arm"]] % 2 == 0 else -16
                ax.annotate(p["arm"], (p["x"], p["y"]),
                            textcoords="offset points",
                            xytext=(-8 if right else 8, dy),
                            ha="right" if right else "left",
                            fontsize=10, color=INK, zorder=4)

        ax.set_title(model, color=INK)
        ax.set_xlabel("distinct-2 (first 128 tokens)")
        if model == MODELS[0]:
            ax.set_ylabel("calibrated support rate")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)

    handles = [Line2D([], [], color=STYLE[a][0], marker=STYLE[a][1], ls="",
                      ms=8, mec=STYLE[a][0], mew=1.6, label=a)
               for fam, (_, arms_) in FAMILY.items() for a in arms_]
    handles.append(Line2D([], [], color=MUTED, marker="o", ls="", ms=8,
                          mfc="white", mec=MUTED, mew=1.6,
                          label="hollow: calibrates to zero"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("Factuality against diversity, by model scale", y=1.02, color=INK)
    os.makedirs("figures", exist_ok=True)
    fig.savefig(FIG)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
