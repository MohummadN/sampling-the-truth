#!/usr/bin/env python3
"""F1 — the page-1 teaser: factuality against diversity, all arms, three models.

One hue, shape by decoding family, marker area proportional to inference
compute. Nine categorical hues would fail colourblind separation in a scatter
and would vanish in greyscale print, so identity is carried by shape plus a
direct label on every point.

Test split. Support rate is the raw (uncalibrated) headline metric, matching
the first column of the main results table. Degenerate outputs score 0.0.
"""
import collections
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import reported_entities

KEEP = reported_entities()
MODELS = ("gpt2", "llama-1b", "llama-3b")
TITLES = {"gpt2": "GPT-2 (124M)", "llama-1b": "Llama-3.2-1B",
          "llama-3b": "Llama-3.2-3B"}

FAMILY = {
    "greedy": ("likelihood", "o"), "beam4": ("likelihood", "o"),
    "greedy_reppen": ("likelihood", "o"),
    "temp0.7": ("sampling", "^"), "temp1.3": ("sampling", "^"),
    "nucleus0.9": ("sampling", "^"),
    "dola": ("contrastive", "s"), "dola_nucleus": ("contrastive", "s"),
    "sc_k5": ("aggregation", "D"),
}
LABEL = {"greedy": "greedy", "beam4": "beam-4", "greedy_reppen": "greedy+rp",
         "temp0.7": "T=0.7", "temp1.3": "T=1.3", "nucleus0.9": "nucleus",
         "dola": "DoLa", "dola_nucleus": "DoLa+nuc", "sc_k5": "SC k=5"}

HUE = "#2a78d6"
INK, MUTED = "#0b0b0b", "#52514e"


def macro(per_entity):
    """Average seeds within a prompt, then average prompts."""
    return st.mean(st.mean(v) for v in per_entity.values()) if per_entity else None


def load_support_per_entity():
    """Per-entity support values, before any averaging.

    The figures want the macro mean; calibrate.py wants the per-prompt vectors
    to bootstrap over. Both come from here, so an arm can no longer be present
    in one and silently absent from the other - sc_k5 was.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob("outputs/verdicts_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if r["entity"] not in KEEP:
                continue
            v = r["support_rate"]
            acc[(r["model"], r["decoding"])][r["entity"]].append(
                0.0 if v is None else v)

    # sc_k5: the medoid-selected sample
    sel = {}
    for p in glob.glob("outputs/scchoice_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            sel[(r["model"], r["seed"], r["entity"])] = r["selected_index"]
    for p in glob.glob("outputs/scverdicts_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if r["entity"] not in KEEP:
                continue
            j = int(r["decoding"].split("#")[1])
            if sel.get((r["model"], r["seed"], r["entity"])) != j:
                continue
            v = r["support_rate"]
            acc[(r["model"], "sc_k5")][r["entity"]].append(
                0.0 if v is None else v)
    return acc


def load_support():
    return {k: macro(v) for k, v in load_support_per_entity().items()}


def load_diversity():
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob("outputs/metrics_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if r["entity"] not in KEEP:
                continue
            arm = r["decoding"]
            if arm.startswith("sc_k5"):
                # metrics stores one row per sample as sc_k5#0..#4; the arm is
                # represented by the medoid the selector chose
                if not r.get("sc_selected"):
                    continue
                arm = "sc_k5"
            d = r.get("distinct_2_prefix")
            if d is None:
                continue
            acc[(r["model"], arm)][r["entity"]].append(d)
    return {k: macro(v) for k, v in acc.items()}


def load_compute():
    acc = collections.defaultdict(list)
    for p in glob.glob("outputs/gen_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if r["entity"] not in KEEP:
                continue
            c = r.get("compute_tokens")
            if c:
                acc[(r["model"], r["decoding"])].append(c)
    out = {}
    for model in MODELS:
        base = st.mean(acc[(model, "greedy")])
        for (m, arm), v in acc.items():
            if m == model:
                out[(m, arm)] = st.mean(v) / base
    return out


def main():
    sup, div, comp = load_support(), load_diversity(), load_compute()

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

    xs = [div[k] for k in div if k[1] in FAMILY and div[k] is not None]
    ys = [sup[k] for k in sup if k[1] in FAMILY and sup[k] is not None]
    xpad = (max(xs) - min(xs)) * 0.10
    ypad = (max(ys) - min(ys)) * 0.14

    # five arms share almost identical diversity, so words collide no matter
    # where they are placed; digits plus a key below the figure do not
    IDX = {arm: n for n, arm in enumerate(FAMILY, 1)}
    # five arms land within a few hundredths of each other on x; fixed
    # per-arm directions keep their digits apart in every panel
    OFF = {"greedy": (5, 4), "beam4": (5, 4), "greedy_reppen": (5, -10),
           "temp0.7": (5, 4), "temp1.3": (-10, -10), "nucleus0.9": (5, 5),
           "dola": (-11, 2), "dola_nucleus": (5, 4), "sc_k5": (-11, -10)}

    for ax, model in zip(axes, MODELS):
        ax.grid(True, lw=0.4, color="#e5e5e2", zorder=0)
        ax.set_axisbelow(True)
        for arm, (fam, marker) in FAMILY.items():
            x, y = div.get((model, arm)), sup.get((model, arm))
            if x is None or y is None:
                continue
            c = comp.get((model, arm), 1.0)
            ax.scatter(x, y, s=16 + 30 * c, marker=marker, color=HUE,
                       edgecolor="white", linewidth=0.7, zorder=3)
            ax.annotate(str(IDX[arm]), (x, y), textcoords="offset points",
                        xytext=OFF.get(arm, (5, 4)), fontsize=6, color=INK,
                        zorder=4)
        ax.set_title(TITLES[model], color=INK)
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    axes[0].set_ylabel("support rate")
    axes[1].set_xlabel("distinct-2, first 128 tokens  (diversity)")

    k1 = ("○ likelihood:  1 greedy,  2 beam-4,  3 greedy+rp          "
          "△ sampling:  4 T=0.7,  5 T=1.3,  6 nucleus")
    k2 = ("□ contrastive:  7 DoLa,  8 DoLa+nuc          "
          "◇ aggregation:  9 SC k=5          "
          "marker area ∝ inference compute")
    fig.text(0.5, -0.14, k1, ha="center", fontsize=6, color=MUTED)
    fig.text(0.5, -0.21, k2, ha="center", fontsize=6, color=MUTED)

    os.makedirs("paper/figures", exist_ok=True)
    fig.savefig("paper/figures/f1_teaser.pdf")
    print("wrote paper/figures/f1_teaser.pdf")
    for model in MODELS:
        print(f"\n{model}")
        for arm in FAMILY:
            x, y = div.get((model, arm)), sup.get((model, arm))
            if x is not None and y is not None:
                print(f"  {arm:14s} distinct2={x:.3f} support={y:.3f} "
                      f"compute={comp.get((model, arm), float('nan')):.2f}x")


if __name__ == "__main__":
    main()
