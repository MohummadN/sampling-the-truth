#!/usr/bin/env python3
"""T4 — paired bootstrap against nucleus-0.9, as LaTeX.

Parses the analysis.py blocks already in paper/numbers.txt rather than
recomputing them, so the table cannot drift from the text output. Regenerate
numbers.txt first if the data changed.
"""
import re

SRC, OUT = "paper/numbers.txt", "paper/table4.tex"
MODELS = [("gpt2", "GPT-2"), ("llama-1b", "Llama-1B"), ("llama-3b", "Llama-3B")]
NUM = re.compile(r"([+-]\d\.\d+) \[([+-]\d\.\d+),([+-]\d\.\d+)\](\*?)")
NAME = {"beam4": "beam-4", "greedy": "greedy", "greedy_reppen": "greedy+rp",
        "temp0.7": "T=0.7", "temp1.3": "T=1.3", "dola": "DoLa",
        "dola_nucleus": "DoLa+nuc", "sc_k5 (selected)": "SC k=5",
        "sc_k5 (one sample)": "SC, one sample"}

text = open(SRC).read()
block = text[text.index("### analysis"):]
data, model = {}, None
for line in block.splitlines():
    if line.startswith("=== "):
        model = line.split()[1]
        continue
    m = list(NUM.finditer(line))
    if len(m) == 3 and model:
        arm = line[:m[0].start()].strip()
        d, lo, hi, star = m[0].groups()
        data[(model, arm)] = (float(d), lo, hi, star)

order = [a for a in NAME if any((mk, a) in data for mk, _ in MODELS)]
L = [r"\begin{table}[t]", r"  \centering", r"  \small",
     r"  \begin{tabular}{l rrr}", r"    \toprule",
     r"    & \multicolumn{3}{c}{$\Delta$ support rate vs.\ nucleus-0.9} \\",
     r"    \cmidrule(lr){2-4}",
     r"    \textbf{Configuration} & " +
     " & ".join(f"\\textbf{{{lab}}}" for _, lab in MODELS) + r" \\",
     r"    \midrule"]
for arm in order:
    cells = []
    for mk, _ in MODELS:
        v = data.get((mk, arm))
        cells.append("--" if v is None
                     else f"${v[0]:+.3f}" + (r"^{*}$" if v[3] else "$"))
    L.append(f"    {NAME[arm]} & " + " & ".join(cells) + r" \\")
L += [r"    \bottomrule", r"  \end{tabular}",
      r"  \caption{Paired bootstrap over the 80 test prompts, 10{,}000"
      r" resamples, seeds averaged within a prompt. $^{*}$ marks a 95\%"
      r" confidence interval excluding zero at the three decimals reported."
      r" The controlled DoLa comparison is DoLa+nucleus, which differs from"
      r" nucleus only in the layer contrast and is significant at all three"
      r" scales; the plain DoLa row is a greedy-family decoder measured"
      r" against a sampling baseline, which is why it reverses sign."
      r" Self-consistency reaches significance only"
      r" at 3B, where DoLa+nucleus"
      r" delivers twice the gain at a fifth of the compute.}",
      r"  \label{tab:paired}", r"\end{table}", ""]

open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}\n" + "\n".join(L))
