import os

import pytest
import torch

from src.models import DTYPE, MODELS, load, revision


@pytest.fixture(scope="module")
def gpt2():
    return load("gpt2")


def test_unknown_model_raises(  ):
    with pytest.raises(KeyError, match="unknown model"):
        load("gpt-4")


def test_tokenizer_is_configured_for_generation(gpt2):
    """Left padding is mandatory for decoder-only generation: with right
    padding the model continues from <pad> tokens and produces garbage, with
    no error at all."""
    tok, _ = gpt2
    assert tok.padding_side == "left"
    assert tok.pad_token is not None


def test_greedy_is_deterministic(gpt2):
    """Catches a missing .eval() — with dropout active, greedy is not
    reproducible and nothing else in the project can be trusted."""
    tok, model = gpt2
    ids = tok("Tell me a bio of Ada Lovelace.", return_tensors="pt").input_ids.to(model.device)
    kw = dict(max_new_tokens=20, do_sample=False, temperature=1.0, top_p=1.0, top_k=0)
    with torch.no_grad():
        a = model.generate(ids, **kw)
        b = model.generate(ids, **kw)
    assert torch.equal(a, b)


def test_logits_are_finite_in_fp16(gpt2):
    """fp16 has a far narrower range than fp32 or bf16, so overflow to inf/nan
    is a real risk. It would surface as silent garbage, not an exception."""
    tok, model = gpt2
    ids = tok("Tell me a bio of Ada Lovelace.", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits
    assert torch.isfinite(logits).all()


def test_revision_is_recorded(gpt2):
    _, model = gpt2
    assert revision(model) != ""


@pytest.mark.skipif(not os.environ.get("SLOW"), reason="set SLOW=1 to download Llama")
def test_llama_loads_and_fits():
    """Run once with SLOW=1 to prove the gated repo, the token, and the 12.7 GB
    card all work for the largest model."""
    for name in ("llama-1b", "llama-3b"):
        tok, model = load(name)
        assert next(model.parameters()).dtype is DTYPE
        del model
        torch.cuda.empty_cache()
