import math

import pytest
import torch

from src.generate import append, cell_seed, generate_one, load_done, run_key
from src.models import load

ENT = {"prompt_id": "p000", "entity": "Ada Lovelace", "split": "dev",
       "stratum": 0, "prompt": "Tell me a bio of Ada Lovelace."}


# ------------------------------------------------------------------ no model

def test_cell_seed_is_stable_across_processes():
    """hashlib, not hash(): Python's hash() is salted per process, so the same
    cell would draw different samples on every run."""
    assert cell_seed("greedy", 1234, "p001") == cell_seed("greedy", 1234, "p001")
    assert cell_seed("greedy", 1234, "p001") == 244322643


def test_cell_seed_differs_per_cell():
    s = {cell_seed(a, 1234, p)
         for a in ("greedy", "nucleus0.9") for p in ("p001", "p002")}
    assert len(s) == 4


def test_roundtrip_and_resume(tmp_path):
    p = str(tmp_path / "g.jsonl")
    rec = {"model": "gpt2", "decoding": "greedy", "seed": 1, "prompt_id": "p000"}
    append(p, rec)
    assert load_done(p) == {("gpt2", "greedy", 1, "p000")}


def test_truncated_last_line_is_tolerated(tmp_path):
    """A pre-empted job can leave half a line. Resume must not crash on it."""
    p = str(tmp_path / "g.jsonl")
    rec = {"model": "gpt2", "decoding": "greedy", "seed": 1, "prompt_id": "p000"}
    append(p, rec)
    with open(p, "a") as f:
        f.write('{"model": "gpt2", "decod')
    assert load_done(p) == {("gpt2", "greedy", 1, "p000")}


# --------------------------------------------------------------- needs gpt2

@pytest.fixture(scope="module")
def gpt2_cpu():
    """CPU + fp32 so these run anywhere, including a login node."""
    return load("gpt2", dtype=torch.float32, device="cpu")


def test_greedy_is_seed_invariant(gpt2_cpu):
    """Greedy takes the argmax, so the seed must not change anything. If this
    fails, sampling is on somewhere it should not be."""
    tok, model = gpt2_cpu
    a = generate_one(tok, model, "greedy", ENT, seed=1)["text"]
    b = generate_one(tok, model, "greedy", ENT, seed=2)["text"]
    assert a == b


def test_sampling_is_seed_dependent(gpt2_cpu):
    """The counterpart: if sampling were silently disabled, every seed would
    return the same string and the whole diversity axis would be dead."""
    tok, model = gpt2_cpu
    outs = {generate_one(tok, model, "nucleus0.9", ENT, seed=s)["text"]
            for s in (1, 2, 3)}
    assert len(outs) > 1


def test_beam_compute_exceeds_returned_tokens(gpt2_cpu):
    """beam4 decodes num_beams sequences but returns one, so compute_tokens
    must be 4x gen_tokens. Conflating them understates beam search's cost by
    that factor, and matched inference compute is a headline claim."""
    tok, model = gpt2_cpu
    r = generate_one(tok, model, "beam4", ENT, seed=1)
    assert r["compute_tokens"] == 4 * r["gen_tokens"]


def test_greedy_compute_equals_returned(gpt2_cpu):
    tok, model = gpt2_cpu
    r = generate_one(tok, model, "greedy", ENT, seed=1)
    assert r["compute_tokens"] == r["gen_tokens"]
    assert r["gen_tokens"] <= 256      # prompt stripped by index, not string


# ------------------------------------------------------------------- locking

def test_lock_refuses_a_live_owner(tmp_path):
    """The collision the lock exists to prevent: two jobs appending to one
    shard. sc_k5 records exceed the 4 KB atomic-append limit, so concurrent
    writes corrupt lines rather than merely reordering them."""
    import os
    import socket

    from src.generate import acquire_lock, release_lock

    p = str(tmp_path / "shard.jsonl")
    lock = acquire_lock(p)              # this process is the live owner
    try:
        with pytest.raises(SystemExit, match="still running"):
            acquire_lock(p)
    finally:
        release_lock(lock)


def test_lock_breaks_a_stale_owner(tmp_path):
    """A pre-empted job cannot run its finally block. If its lock outlived it,
    the resubmit would be blocked — defeating the whole resume design."""
    import json

    from src.generate import acquire_lock, release_lock

    p = str(tmp_path / "shard.jsonl")
    with open(p + ".lock", "w") as f:
        json.dump({"slurm_job_id": None, "host": "a-host-that-is-not-this-one",
                   "pid": 1, "started": "x"}, f)
    lock = acquire_lock(p)              # must break it, not refuse
    release_lock(lock)


def test_lock_is_released_and_reacquirable(tmp_path):
    from src.generate import acquire_lock, release_lock

    p = str(tmp_path / "shard.jsonl")
    release_lock(acquire_lock(p))
    release_lock(acquire_lock(p))       # no leftover blocking the second run
