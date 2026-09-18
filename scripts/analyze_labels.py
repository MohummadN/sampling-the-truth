#!/usr/bin/env python3
"""Verifier validation and theta tuning against the 100 manual labels.

Joins outputs/labels.csv to outputs/labeling_key.csv. Every estimate is
weighted by the sampling weight, so numbers refer to the full 51,973-sentence
frame rather than to the stratified sample.

Human labels collapse to binary the way the verifier's output is binary:
  supported                      -> supported
  contradicted, not_addressed    -> not supported
`unsure` rows are excluded and counted separately.
"""
import collections
import csv
import statistics as st

LAB = "outputs/labels.csv"
KEY = "outputs/labeling_key.csv"


def load():
    labels = {r["id"]: r for r in csv.DictReader(open(LAB, encoding="utf-8"))}
    key = {r["id"]: r for r in csv.DictReader(open(KEY, encoding="utf-8"))}
    rows, unsure = [], 0
    for i, lab in labels.items():
        if i not in key:
            raise SystemExit(f"{i} missing from key")
        if lab["label"] == "unsure":
            unsure += 1
            continue
        k = key[i]
        rows.append({
            "id": i,
            "human": lab["label"],
            "human_sup": lab["label"] == "supported",
            "note": lab["note"],
            "p": float(k["p_entail"]),
            "w": float(k["weight"]),
            "stratum": k["stratum"],
            "band": k["band"],
            "arm": k["arm"],
            "model": k["model"],
        })
    return rows, unsure, len(labels)


def wmean(rows, pred):
    num = sum(r["w"] for r in rows if pred(r))
    den = sum(r["w"] for r in rows)
    return num / den if den else 0.0


def metrics(rows, theta):
    tp = [r for r in rows if r["p"] >= theta and r["human_sup"]]
    fp = [r for r in rows if r["p"] >= theta and not r["human_sup"]]
    fn = [r for r in rows if r["p"] < theta and r["human_sup"]]
    tn = [r for r in rows if r["p"] < theta and not r["human_sup"]]
    W = lambda xs: sum(r["w"] for r in xs)
    tpw, fpw, fnw, tnw = W(tp), W(fp), W(fn), W(tn)
    acc = (tpw + tnw) / (tpw + fpw + fnw + tnw)
    prec = tpw / (tpw + fpw) if tpw + fpw else 0.0
    rec = tpw / (tpw + fnw) if tpw + fnw else 0.0
    spec = tnw / (tnw + fpw) if tnw + fpw else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "spec": spec, "f1": f1,
            "bal": (rec + spec) / 2,
            "n": (len(tp), len(fp), len(fn), len(tn))}


def main():
    rows, unsure, total = load()
    print(f"labels: {total}   usable: {len(rows)}   unsure (excluded): {unsure}\n")

    print("human label distribution")
    c = collections.Counter(r["human"] for r in rows)
    for k in sorted(c):
        w = wmean(rows, lambda r, k=k: r["human"] == k)
        print(f"  {k:15s} {c[k]:3d} raw   {w:6.1%} reweighted to frame")

    notes = collections.Counter(r["note"] for r in rows if r["note"])
    if notes:
        print("\nnotes")
        for k, v in notes.most_common():
            print(f"  {k:15s} {v:3d}")

    print("\ntheta sweep (weighted)")
    print(f"  {'theta':>6s} {'accuracy':>9s} {'balanced':>9s} {'precision':>10s} {'recall':>8s}")
    best = None
    for i in range(1, 20):
        th = i / 20
        m = metrics(rows, th)
        star = ""
        if best is None or m["bal"] > best[1]["bal"]:
            best, star = (th, m), ""
        print(f"  {th:6.2f} {m['acc']:9.3f} {m['bal']:9.3f} {m['prec']:10.3f} {m['rec']:8.3f}")
    print(f"\n  best balanced accuracy at theta = {best[0]:.2f}")

    for th in sorted({0.5, best[0]}):
        m = metrics(rows, th)
        tp, fp, fn, tn = m["n"]
        print(f"\n--- theta = {th:.2f} " + "-" * 40)
        print(f"  confusion (raw counts)     human sup   human not")
        print(f"    verifier supported       {tp:9d}   {fp:9d}")
        print(f"    verifier not supported   {fn:9d}   {tn:9d}")
        print(f"  weighted: accuracy {m['acc']:.3f}  balanced {m['bal']:.3f}  "
              f"precision {m['prec']:.3f}  recall {m['rec']:.3f}  "
              f"specificity {m['spec']:.3f}  F1 {m['f1']:.3f}")

    print("\nagreement by stratum (theta = 0.50, weighted)")
    for s in sorted({r["stratum"] for r in rows}):
        sub = [r for r in rows if r["stratum"] == s]
        m = metrics(sub, 0.5)
        print(f"  stratum {s}  n={len(sub):3d}  accuracy {m['acc']:.3f}  "
              f"balanced {m['bal']:.3f}")

    print("\nwhat the verifier gets wrong at theta = 0.50")
    fp = [r for r in rows if r["p"] >= 0.5 and not r["human_sup"]]
    fn = [r for r in rows if r["p"] < 0.5 and r["human_sup"]]
    print(f"  false positives: {len(fp)}  "
          f"({collections.Counter(r['human'] for r in fp)})")
    print(f"  false negatives: {len(fn)}")
    if fp:
        print(f"  FP mean p_entail {st.mean(r['p'] for r in fp):.3f}")
    if fn:
        print(f"  FN mean p_entail {st.mean(r['p'] for r in fn):.3f}")


if __name__ == "__main__":
    main()
