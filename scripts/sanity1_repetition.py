"""GATE 2 / task A6 — sanity experiment 1 (graded).

Reproduces Holtzman et al. 2020: deterministic decoders degenerate into loops
while nucleus sampling does not. If this fails, the bug is in the decoding
kwargs, the prompt stripping, or the tokenization — not in the metric.

    python -m src.generate --model gpt2 --seed 1234 --limit 30 \
        --arms greedy,beam4,nucleus0.9 --out outputs/sanity1.jsonl
    python scripts/sanity1_repetition.py
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics.diversity import first_tokens, loop_severity, rep_n

SHARD = "outputs/sanity1.jsonl"
FIG = "figures/sanity1_repetition.pdf"
ARMS = ["greedy", "beam4", "nucleus0.9"]
LABELS = {"greedy": "greedy", "beam4": "beam-4", "nucleus0.9": "nucleus $p$=0.9"}

ACCENT = "#2a6fb0"      # one hue: the arms are x categories, not a second series
INK = "#1a1a1a"
MUTED = "#6b6b6b"


def ci95(xs: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% CI across PROMPTS.

    Not across seeds: greedy and beam-4 are deterministic, so a seed-wise
    error bar would be exactly zero. Say this in the caption or it reads as an
    omission.
    """
    xs = [x for x in xs if x == x]                  # drop nan
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def main() -> None:
    if not os.path.exists(SHARD):
        raise SystemExit(f"{SHARD} missing — run the generation command first")

    rep: dict[str, list[float]] = {a: [] for a in ARMS}
    loop: dict[str, list[float]] = {a: [] for a in ARMS}
    caps: dict[str, list[bool]] = {a: [] for a in ARMS}

    for line in open(SHARD):
        r = json.loads(line)
        arm = r["decoding"]
        if arm not in rep or not r.get("text"):
            continue
        # Fixed 128-token prefix: greedy and beam run to max_new_tokens while
        # nucleus terminates early, so length alone could produce the ordering.
        prefix = first_tokens(r["text"], 128)
        rep[arm].append(rep_n(prefix, 4))
        loop[arm].append(loop_severity(r["text"]))
        caps[arm].append(bool(r["hit_cap"]))

    print(f"{'arm':<16}{'n':>4}{'rep-4':>18}{'loop sev.':>18}{'cap rate':>10}")
    stats = {}
    for a in ARMS:
        rm, rc = ci95(rep[a])
        lm, lc = ci95(loop[a])
        cap = sum(caps[a]) / len(caps[a]) if caps[a] else float("nan")
        stats[a] = (rm, rc, lm, lc)
        print(f"{LABELS[a]:<16}{len(rep[a]):>4}"
              f"{rm:>11.3f} ±{rc:.3f}{lm:>11.3f} ±{lc:.3f}{cap:>10.0%}")

    # ---- figure: two panels, never a dual axis (different scales) ----
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = range(len(ARMS))
    names = [LABELS[a] for a in ARMS]

    for ax, idx, title, ylab in (
            (axes[0], 0, "Repetition", "rep-4 (first 128 tokens)"),
            (axes[1], 2, "Loop severity", "longest repeat / length")):
        vals = [stats[a][idx] for a in ARMS]
        errs = [stats[a][idx + 1] for a in ARMS]
        ax.bar(x, vals, yerr=errs, capsize=5, color=ACCENT, width=0.62,
               edgecolor="white", linewidth=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_ylabel(ylab)
        ax.set_title(title, color=INK)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e2e2e2", linewidth=0.8)   # recessive grid
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        # Direct labels: three bars, so every one is labelled without clutter.
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + 0.03, f"{v:.2f}", ha="center", color=INK)

    os.makedirs("figures", exist_ok=True)
    fig.savefig(FIG)
    print(f"\nwrote {FIG}")

    # ---- the gate ----
    ok = (stats["greedy"][0] - stats["greedy"][1]
          > stats["nucleus0.9"][0] + stats["nucleus0.9"][1]
          and stats["beam4"][0] - stats["beam4"][1]
          > stats["nucleus0.9"][0] + stats["nucleus0.9"][1])
    print("\nGATE 2:", "PASS — separated error bars" if ok else "FAIL")
    if not ok:
        raise SystemExit("check decoding kwargs, prompt stripping, tokenization "
                         "— in that order")


if __name__ == "__main__":
    main()