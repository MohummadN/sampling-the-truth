#!/usr/bin/env python3
"""Interactive labelling for the 100-sentence manual validation (Stage 7).

    python -m scripts.label

One item per screen. Progress is written after every answer, so you can stop
and resume freely. The verifier's verdict is never shown.

Keys — type the letter, optionally followed by a note:
    s          supported     — the evidence shown establishes the claim
    u          contradicted  — the evidence shown says otherwise
    a          not addressed — the evidence neither establishes nor contradicts

    No world knowledge is needed. You are not judging whether the claim is
    true, only what the evidence does with it.
    ?          unsure
    b          back one item
    q          save and quit

    e.g.  u fragment        or       s generic
"""
import csv
import os
import sys
import textwrap
from datetime import datetime

SHEET = "outputs/labeling_sheet.csv"
LABELS = "outputs/labels.csv"
VALID = {"s": "supported", "u": "contradicted",
         "a": "not_addressed", "?": "unsure"}
W = 96


def wrap(text, indent="  "):
    out = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, W, initial_indent=indent,
                                 subsequent_indent=indent) or [indent])
    return "\n".join(out)


def load_done():
    if not os.path.exists(LABELS):
        return {}
    with open(LABELS, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def save(done, order):
    tmp = LABELS + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "note", "labeled_at"])
        for i in order:
            if i in done:
                r = done[i]
                w.writerow([r["id"], r["label"], r["note"], r["labeled_at"]])
    os.replace(tmp, LABELS)


def main():
    with open(SHEET, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    order = [r["id"] for r in rows]
    done = load_done()

    redo = set()
    if "--redo" in sys.argv:
        want = sys.argv[sys.argv.index("--redo") + 1]
        redo = {k for k, v in done.items() if v["label"] == want}
        if not redo:
            print(f"  nothing labelled '{want}' to redo")
            return
        print(f"  redo mode: {len(redo)} item(s) labelled '{want}'")
        input("  press Enter to start ")

    i = 0
    while i < len(rows):
        r = rows[i]
        if r["id"] in done and r["id"] not in redo:
            i += 1
            continue

        os.system("clear")
        n_done = len(done)
        print(f"  [{n_done}/{len(rows)} labelled]   item {r['id']}")
        print("=" * W)
        print(f"\n  ENTITY: {r['entity']}\n")
        print("  CLAIM:")
        print(wrap(r["sentence"], "    "))
        print("\n  EVIDENCE SHOWN TO THE VERIFIER:")
        print(wrap(r["evidence"], "    "))
        print("\n" + "-" * W)
        print("  s=evidence establishes it    u=evidence contradicts it    "
              "a=evidence doesn't address it")
        print("  ?=can't decide    b=back    q=quit")
        try:
            ans = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not ans:
            continue
        key, _, note = ans.partition(" ")
        key = key.lower()

        if key == "q":
            break
        if key == "b":
            j = i - 1
            while j >= 0 and rows[j]["id"] not in done:
                j -= 1
            if j >= 0:
                del done[rows[j]["id"]]
                i = j
                save(done, order)
            continue
        if key not in VALID:
            continue

        redo.discard(r["id"])
        done[r["id"]] = {"id": r["id"], "label": VALID[key],
                         "note": note.strip(),
                         "labeled_at": datetime.now().isoformat(timespec="seconds")}
        save(done, order)
        i += 1

    save(done, order)
    print(f"\n  {len(done)}/{len(rows)} labelled -> {LABELS}")
    if len(done) < len(rows):
        print("  run `python -m scripts.label` again to continue where you left off")


if __name__ == "__main__":
    main()
