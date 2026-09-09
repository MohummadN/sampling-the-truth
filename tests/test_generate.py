import json

from src.generate import append, cell_seed, load_done, run_key


def test_cell_seed_is_stable_across_processes():
    """hashlib, not hash(): Python's hash() is salted per process, so seeds
    would differ between runs and the study would not be reproducible."""
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