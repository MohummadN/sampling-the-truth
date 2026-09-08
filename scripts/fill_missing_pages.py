"""Second pass over the pinned snapshot for titles the first pass missed.

The first streaming pass found 454/500, with the misses randomly distributed
across famous and obscure entities alike — consistent with the streaming
reader dropping rows, not with the pages being absent. This re-runs the pass
against only the missing titles and merges the result into the cache.

    srun -p studentkillable --time=01:00:00 python scripts/fill_missing_pages.py
"""
from __future__ import annotations

import json
import time

from datasets import load_dataset

SNAPSHOT = "20231101.en"
PAGES_JSON = "data/reference_pages.json"
ENTITIES_TXT = "data/factscore_entities.txt"


def main() -> None:
    entities = open(ENTITIES_TXT).read().splitlines()
    pages = json.load(open(PAGES_JSON))
    wanted = {e for e in entities if e not in pages}
    if not wanted:
        print("nothing missing")
        return
    print(f"looking for {len(wanted)} missing titles")

    found: dict[str, str] = {}
    t0 = time.time()
    ds = load_dataset("wikimedia/wikipedia", SNAPSHOT, split="train", streaming=True)
    for i, row in enumerate(ds, 1):
        if row["title"] in wanted:
            found[row["title"]] = row["text"]
            print(f"  + {row['title']}  (row {i:,})", flush=True)
            if len(found) == len(wanted):
                break
        if i % 1_000_000 == 0:
            print(f"  {i:,} rows, {len(found)}/{len(wanted)}, {time.time()-t0:.0f}s",
                  flush=True)

    pages.update(found)
    with open(PAGES_JSON, "w") as f:
        json.dump(pages, f, ensure_ascii=False)
    print(f"\nfound {len(found)}, cache now {len(pages)}/500, {time.time()-t0:.0f}s")

    still = sorted(wanted - set(found))
    if still:
        print(f"\nstill missing ({len(still)}):")
        for s in still:
            print("   ", s)

if __name__ == "__main__":
    main()
