"""One-time build: FActScore entity pool + pinned Wikipedia reference pages.

Run once, from the repo root:
    srun -p studentkillable --time=01:00:00 python scripts/build_reference_pages.py

Writes:
    data/factscore_entities.txt   the 500 candidate names (committed — tiny)
    data/reference_pages.json     title -> page text (gitignored — build artifact)

Never run this again. Every downstream job reads the cached JSON. See
DECISIONS.md: the snapshot is pinned to 20231101.en because a later snapshot
could contain facts the models never saw during pretraining.
"""
from __future__ import annotations

import json
import os
import time

from datasets import load_dataset

SNAPSHOT = "20231101.en"
ENTITIES_TXT = "data/factscore_entities.txt"
PAGES_JSON = "data/reference_pages.json"


def fetch_entity_pool() -> list[str]:
    """The 500 FActScore biography entities. Each name is also its Wikipedia title."""
    d = load_dataset("dskar/FActScore")["test"]
    ents = list(dict.fromkeys(d["entity"]))       # de-dup, preserve order
    assert len(ents) == 500, f"expected 500 entities, got {len(ents)}"
    return ents


def main() -> None:
    os.makedirs("data", exist_ok=True)

    entities = fetch_entity_pool()
    with open(ENTITIES_TXT, "w") as f:
        f.write("\n".join(entities) + "\n")
    print(f"wrote {ENTITIES_TXT} ({len(entities)} names)")

    wanted = set(entities)
    pages: dict[str, str] = {}
    t0 = time.time()

    # One streaming pass. Set membership is O(1) — never scan the corpus per
    # title, which is what HW3's get_page does and would take hours here.
    ds = load_dataset("wikimedia/wikipedia", SNAPSHOT, split="train", streaming=True)
    for i, row in enumerate(ds, 1):
        if row["title"] in wanted:
            pages[row["title"]] = row["text"]
            if len(pages) == len(wanted):
                break
        if i % 500_000 == 0:
            print(f"  {i:,} rows, {len(pages)}/{len(wanted)} found, "
                  f"{time.time()-t0:.0f}s", flush=True)

    # Save BEFORE anything else can go wrong.
    with open(PAGES_JSON, "w") as f:
        json.dump(pages, f, ensure_ascii=False)
    print(f"wrote {PAGES_JSON} ({len(pages)} pages, {time.time()-t0:.0f}s)")

    missing = sorted(wanted - set(pages))
    if missing:
        print(f"\n{len(missing)} titles did not resolve in {SNAPSHOT}:")
        for m in missing:
            print("   ", m)
        print("These are excluded from the candidate pool at selection time.")


if __name__ == "__main__":
    main()