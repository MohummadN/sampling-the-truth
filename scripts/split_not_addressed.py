#!/usr/bin/env python3
"""Split human-labelled `not_addressed` sentences into retrieval failure vs
reference-coverage limit.

For every sentence the human labelled `not_addressed`, scan EVERY 3-sentence
window of the entity's page — not just the top-5 the retriever returned. If the
page entails it somewhere, the top-5 retrieval missed it: a pipeline failure.
If nothing in the page entails it, the claim is either absent from the
reference or invented.

Searching all windows gives the NLI far more chances to fire than the top-5
did, and this verifier is known to over-call entailment. So the same scan runs
against a same-stratum WRONG page as a control: that rate is the noise floor,
and only the excess over it is real retrieval failure.

    python -m scripts.split_not_addressed

Writes outputs/na_split.csv.
"""
import csv
import json
import time

from src.data import load_entities, load_pages
from src.metrics.factuality import (THETA, WINDOW, entail_probs, load_nli)
from src.text import split_sentences

BATCH = 48
OUT = "outputs/na_split.csv"


def wrong_map(ents):
    by = {}
    for e in ents:
        by.setdefault(e["stratum"], []).append(e["entity"])
    out = {}
    for names in by.values():
        names = sorted(names)
        for i, n in enumerate(names):
            out[n] = names[(i + 1) % len(names)]
    return out


def windows_of(page):
    sents = split_sentences(page)
    if len(sents) <= WINDOW:
        return [" ".join(sents)] if sents else []
    return [" ".join(sents[i:i + WINDOW])
            for i in range(len(sents) - WINDOW + 1)]


def max_entail(hyp, wins, tok, model, entail_id):
    best = 0.0
    for i in range(0, len(wins), BATCH):
        chunk = wins[i:i + BATCH]
        probs = entail_probs(chunk, [hyp] * len(chunk), tok, model, entail_id)
        if probs:
            best = max(best, max(probs))
    return best


def main():
    labels = {r["id"]: r for r in csv.DictReader(open("outputs/labels.csv", encoding="utf-8"))}
    key = {r["id"]: r for r in csv.DictReader(open("outputs/labeling_key.csv", encoding="utf-8"))}
    sheet = {r["id"]: r for r in csv.DictReader(open("outputs/labeling_sheet.csv", encoding="utf-8"))}

    ids = [i for i, l in labels.items() if l["label"] == "not_addressed"]
    print(f"{len(ids)} not_addressed sentences to scan", flush=True)
    if not ids:
        return

    ents = load_entities()
    wrong = wrong_map(ents)
    pages = load_pages()
    tok, model, entail_id = load_nli()
    print(f"entail_id={entail_id} ({model.config.id2label[entail_id]})", flush=True)

    win_cache = {}
    rows = []
    t0 = time.perf_counter()
    for n, i in enumerate(ids, 1):
        ent = sheet[i]["entity"]
        claim = sheet[i]["sentence"]
        w_ent = wrong[ent]
        for e in (ent, w_ent):
            if e not in win_cache:
                win_cache[e] = windows_of(pages[e])

        # same hypothesis the pipeline uses (factuality.py:182), so these
        # numbers are comparable to the verifier's own scores
        hyp = f"{ent}: {claim}"
        p_true = max_entail(hyp, win_cache[ent], tok, model, entail_id)
        p_wrong = max_entail(hyp, win_cache[w_ent], tok, model, entail_id)

        rows.append({
            "id": i, "entity": ent, "wrong_entity": w_ent,
            "model": key[i]["model"], "arm": key[i]["arm"],
            "seed": key[i]["seed"], "stratum": key[i]["stratum"],
            "note": labels[i]["note"],
            "p_top5": key[i]["p_entail"],
            "p_allwindows": f"{p_true:.4f}",
            "p_wrongpage_allwindows": f"{p_wrong:.4f}",
            "n_windows": len(win_cache[ent]),
        })
        if n % 10 == 0 or n == len(ids):
            r = n / (time.perf_counter() - t0)
            print(f"  {n}/{len(ids)}  {r:.2f}/s  "
                  f"eta {(len(ids)-n)/r/60:.0f} min", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {OUT}\n")
    for th in (THETA, 0.9):
        t = sum(float(r["p_allwindows"]) >= th for r in rows) / len(rows)
        c = sum(float(r["p_wrongpage_allwindows"]) >= th for r in rows) / len(rows)
        print(f"threshold {th:.2f}:  true page {t:.1%}   wrong page {c:.1%}   "
              f"excess {t - c:+.1%}")
    print("\nexcess = retrieval failure; the remainder is coverage limit or "
          "hallucination")


if __name__ == "__main__":
    main()
