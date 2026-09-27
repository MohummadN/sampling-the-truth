#!/usr/bin/env python3
"""F3 — factuality against inference compute, with the Pareto frontier.

Reuses the loaders in f1_teaser so both figures are guaranteed to plot the
same numbers, and reuses its arm numbering so a reader learns the key once.

Compute is on a log axis: seven arms sit between 0.74x and 1.13x greedy while
beam-4 and self-consistency sit at 3.5x and 4.5x, and a linear axis would
crush the first group into a single column. Marker size is constant here —
compute is the axis, so encoding it twice would be redundant.

The dashed staircase is the Pareto frontier: arms that no other arm beats on
both axes at once.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from scripts.f1_teaser import (FAMILY, INK, MODELS, MUTED, TITLES, HUE,
                               load_compute, load_support)


def frontier(points):
    """Non-dominated points: nothing cheaper is also at least as factual."""
    out, best = [], -1.0
    for c, s, arm in sorted(points):
        if s > best:
            out.append((c, s, arm))
            best = s
    return out


def main():
    sup, comp = load_support(), load_compute()
    IDX = {arm: n for n, arm in enumerate(FAMILY, 1)}
    # seven arms compress into 0.74-1.13x on a log axis; fixed directions
    OFF = {"greedy": (5, 4), "beam4": (5, 4), "greedy_reppen": (-10, -9),
           "temp0.7": (5, -9), "temp1.3": (5, -9), "nucleus0.9": (-11, 4),
           "dola": (-11, -9), "dola_nucleus": (5, 5), "sc_k5": (5, 4)}

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.8), sharex=True, sharey=True)

    ys = [sup[k] for k in sup if k[1] in FAMILY and sup[k] is not None]
    ypad = (max(ys) - min(ys)) * 0.14

    for ax, model in zip(axes, MODELS):
        ax.grid(True, lw=0.4, color="#e5e5e2", zorder=0)
        ax.set_axisbelow(True)

        pts = []
        for arm, (_, marker) in FAMILY.items():
            c, s = comp.get((model, arm)), sup.get((model, arm))
            if c is None or s is None:
                continue
            pts.append((c, s, arm))
            ax.scatter(c, s, s=26, marker=marker, color=HUE,
                       edgecolor="white", linewidth=0.7, zorder=3)
            ax.annotate(str(IDX[arm]), (c, s), textcoords="offset points",
                        xytext=OFF.get(arm, (5, 4)), fontsize=6, color=INK,
                        zorder=4)

        f = frontier(pts)
        if len(f) > 1:
            ax.step([p[0] for p in f], [p[1] for p in f], where="post",
                    color=MUTED, lw=0.8, ls="--", zorder=2)

        ax.set_xscale("log")
        ax.set_xticks([0.75, 1, 2, 4])
        ax.set_xticklabels(["0.75", "1", "2", "4"])
        # a log axis labels its minor ticks too, which collides with ours
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(TITLES[model], color=INK)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
        # room for the left-pointing digit labels, which otherwise clip
        ax.set_xlim(0.60, 6.0)

    axes[0].set_ylabel("support rate")
    axes[1].set_xlabel("inference compute  (relative to greedy, log scale)")

    k1 = ("\u25cb likelihood:  1 greedy,  2 beam-4,  3 greedy+rp          "
          "\u25b3 sampling:  4 T=0.7,  5 T=1.3,  6 nucleus")
    k2 = ("\u25a1 contrastive:  7 DoLa,  8 DoLa+nuc          "
          "\u25c7 aggregation:  9 SC k=5          "
          "dashed line: Pareto frontier")
    fig.text(0.5, -0.14, k1, ha="center", fontsize=6, color=MUTED)
    fig.text(0.5, -0.21, k2, ha="center", fontsize=6, color=MUTED)

    os.makedirs("paper/figures", exist_ok=True)
    fig.savefig("paper/figures/f3_frontier.pdf")
    print("wrote paper/figures/f3_frontier.pdf\n")
    for model in MODELS:
        pts = [(comp[(model, a)], sup[(model, a)], a) for a in FAMILY
               if (model, a) in sup and (model, a) in comp]
        names = ", ".join(a for _, _, a in frontier(pts))
        print(f"{model:10s} Pareto frontier: {names}")


if __name__ == "__main__":
    main()
