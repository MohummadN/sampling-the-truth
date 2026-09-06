"""Fluency metric: perplexity under a fixed external scorer model.

The scorer must NOT be one of the generating models (gpt2, Llama-3.2-1B/3B),
or a model would find its own family's outputs unusually predictable (06 §6.2).
See DECISIONS.md.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCORER = "Qwen/Qwen2.5-0.5B"


def _device() -> str:
    """CUDA only if this torch build has kernels for the GPU actually present.

    The TAU login nodes carry GTX TITAN X (sm_52); torch 2.14+cu130 ships
    nothing below sm_75, so `.to("cuda")` succeeds and the first forward pass
    dies with `no kernel image is available`. Fall back to CPU instead.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    return "cpu"


def load_scorer(name: str = SCORER):
    """Load the perplexity scorer. Returns (model, tokenizer), eval mode."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.to(_device())
    model.eval()
    return model, tok


@torch.no_grad()
def perplexity(text: str, model, tok) -> float:
    """Base-e perplexity of one string. Lower = more predictable.

    Returns nan for strings of fewer than 2 tokens (nothing to condition on).
    """
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[-1] < 2:
        return float("nan")
    limit = min(
        getattr(model.config, "max_position_embeddings", 1024),
        tok.model_max_length,
    )
    ids = ids[:, :limit].to(model.device)
    out = model(input_ids=ids, labels=ids)   # HF shifts labels internally
    return torch.exp(out.loss).item()        # out.loss is mean NLL in nats