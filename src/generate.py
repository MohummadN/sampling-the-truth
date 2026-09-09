"""Generation: one model, one seed, every arm, every prompt -> one JSONL shard.

Sharding is by (model, seed). Loading Llama-3.2-3B off the netapp share takes
~6 minutes, so a job loads its model once and generates everything for it.
See DECISIONS.md.

    python -m src.generate --model gpt2 --seed 1234
    python -m src.generate --model gpt2 --seed 1234 --limit 3 \
        --arms greedy,beam4,nucleus0.9          # GATE 1

Output is append-only and resumable: each record is flushed and fsynced as it
is produced, and a re-run skips (model, decoding, seed, prompt_id) cells
already present. studentkillable jobs get pre-empted — nothing may be lost on
the way down. A lock file enforces one writer per shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

import torch
import transformers
from transformers import set_seed

from src.data import load_entities
from src.decoding import ARMS, COMMON, kwargs_for
from src.models import DTYPE, MODELS, load, revision
from src.text import split_sentences

TASK = "factscore_bio"          # matches the schema in 04 §6.4
OUT_DIR = "outputs"
MAX_NEW_TOKENS = int(COMMON["max_new_tokens"])


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# Resolved once at import: a subprocess per record costs ~25 ms x 8,100 records
# ~= 3.4 minutes, and would return "unknown" whenever a job's cwd is not the repo.
GIT_COMMIT = _git_commit()


# ---------------------------------------------------------------- resumability

def run_key(rec: dict) -> tuple:
    """The cell identity. One record per (model, decoding, seed, prompt)."""
    return (rec["model"], rec["decoding"], rec["seed"], rec["prompt_id"])


def load_done(path: str) -> set[tuple]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(run_key(json.loads(line)))
            except json.JSONDecodeError:
                continue        # tolerate a truncated last line from a kill
    return done


def append(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())    # survive pre-emption, not just a clean exit


def acquire_lock(path: str) -> str:
    """One writer per shard.

    The filename is keyed on (model, seed) but --arms is free, so two jobs with
    different arm subsets would append to the same file. An sc_k5 record is
    ~7 KB, well past the 4 KB atomic-append limit, so interleaved writes would
    corrupt lines rather than merely reorder them.
    """
    lock = path + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"{lock} exists — another job is writing {path}.\n"
            f"Shards are keyed on (model, seed): run one job per shard.\n"
            f"If a previous job was killed, delete the lock file by hand.")
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return lock


# --------------------------------------------------------------------- seeding

def cell_seed(arm: str, seed: int, prompt_id: str) -> int:
    """A distinct, reproducible seed per cell.

    hashlib, not hash() — Python's hash() is salted per process, so the same
    cell would draw different samples on every run and the study would not be
    reproducible.
    """
    key = f"{arm}|{seed}|{prompt_id}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) % (2 ** 31 - 1)


# ------------------------------------------------------------------ generation

def _sync() -> None:
    """CUDA kernels are async: without this, timing measures queueing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def new_token_counts(new: torch.Tensor, eos_id: int) -> list[int]:
    """Real generated length per sequence, up to and including the first EOS.

    With num_return_sequences > 1 the shorter sequences are padded to the
    longest, so tensor shape alone overstates what was actually produced.
    """
    counts = []
    for row in new:
        hit = (row == eos_id).nonzero()
        counts.append(int(hit[0].item()) + 1 if len(hit) else int(row.shape[-1]))
    return counts


@torch.no_grad()
def generate_one(tok, model, arm: str, entity: dict, seed: int) -> dict:
    kw = kwargs_for(arm)

    # set_seed covers random, numpy and torch (CPU + CUDA) in one call. Sampling
    # currently draws from the torch RNG, but seeding only torch would silently
    # stop being enough if that ever changes.
    set_seed(cell_seed(arm, seed, entity["prompt_id"]))

    inputs = tok(entity["prompt"], return_tensors="pt").to(model.device)
    _sync()
    t0 = time.perf_counter()
    out = model.generate(**inputs, **kw)
    _sync()
    dt = time.perf_counter() - t0

    new = out[:, inputs["input_ids"].shape[-1]:]        # strip prompt by INDEX
    counts = new_token_counts(new, tok.eos_token_id)
    texts = tok.batch_decode(new, skip_special_tokens=True)

    multi = kw.get("num_return_sequences", 1) > 1
    text = None if multi else texts[0]
    samples = texts if multi else None

    returned = sum(counts)
    # Beam search returns one sequence but decodes num_beams of them in
    # parallel, so returned tokens understate its cost by exactly that factor.
    # Matched inference compute is a headline claim — keep the two apart.
    compute = returned * int(kw.get("num_beams", 1))

    return {
        "task": TASK,
        "prompt_id": entity["prompt_id"],
        "entity": entity["entity"],
        "split": entity["split"],
        "stratum": entity["stratum"],
        "model": None,                  # filled by the caller
        "model_id": None,
        "model_revision": None,
        "dtype": str(DTYPE),
        "decoding": arm,
        "gen_kwargs": kw,
        "seed": seed,
        "prompt": entity["prompt"],
        "text": text,
        "samples": samples,
        "selected_index": None,         # sc_k5 only; Fooad's selector fills it
        "gen_tokens": returned,         # tokens actually returned
        "compute_tokens": compute,      # tokens actually decoded (cost axis)
        "gen_time_s": round(dt, 4),
        "hit_cap": any(c >= MAX_NEW_TOKENS for c in counts),
        "n_sentences": None if multi else len(split_sentences(text)),
        "run_id": None,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "git_commit": GIT_COMMIT,
    }


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated; default all")
    ap.add_argument("--split", default=None, choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="first N prompts only (GATE 1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        kwargs_for(a)               # fail fast on a typo, before loading 6 GB

    entities = load_entities()
    if args.split:
        entities = [e for e in entities if e["split"] == args.split]
    if args.limit:
        entities = entities[:args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or f"{OUT_DIR}/gen_{args.model}_seed{args.seed}.jsonl"
    lock = acquire_lock(out)
    try:
        done = load_done(out)
        todo = [(a, e) for a in arms for e in entities
                if (args.model, a, args.seed, e["prompt_id"]) not in done]
        print(f"{len(arms)} arms x {len(entities)} prompts = "
              f"{len(arms)*len(entities)} cells | {len(done)} already done | "
              f"{len(todo)} to run -> {out}", flush=True)
        if not todo:
            return

        tok, model = load(args.model)
        rev = revision(model)
        run_id = f"{args.model}-s{args.seed}-{int(time.time())}"
        print(f"loaded {args.model} rev={rev[:12]} on {model.device}", flush=True)

        t0 = time.perf_counter()
        for i, (arm, ent) in enumerate(todo, 1):
            rec = generate_one(tok, model, arm, ent, args.seed)
            rec["model"] = args.model
            rec["model_id"] = MODELS[args.model]
            rec["model_revision"] = rev
            rec["run_id"] = run_id
            append(out, rec)
            if i % 10 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(todo)}  {rate:.2f} cells/s  "
                      f"eta {(len(todo)-i)/rate/60:.0f} min", flush=True)

        print(f"done in {(time.perf_counter()-t0)/60:.1f} min")
    finally:
        os.unlink(lock)


if __name__ == "__main__":
    main()