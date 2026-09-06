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
    """CUDA only if this torch build has kernels this GPU can actually run.

    CUDA guarantees binary compatibility upward within a major version: a cubin
    built for sm_X.y runs on sm_X.z for z >= y. The cluster's TITAN Xp is
    sm_61 and the cu126 wheel ships sm_60, which is therefore usable — a plain
    `"sm_61" in get_arch_list()` test would wrongly fall back to CPU.
    """
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    for arch in torch.cuda.get_arch_list():
        if not arch.startswith("sm_"):
            continue                        # skip compute_XX (PTX) entries
        code = "".join(c for c in arch[3:] if c.isdigit())   # sm_90a -> "90"
        if not code:
            continue
        if int(code[:-1]) == major and int(code[-1]) <= minor:
            return "cuda"
    return "cpu"


def load_scorer(name: str = SCORER):
    """Load the perplexity scorer. Returns (tokenizer, model), eval mode.

    Order matches src/models.py's load() -> (tok, model), so every loader in
    the repo unpacks the same way.
    """
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.to(_device())
    model.eval()
    return tok, model


@torch.no_grad()
def perplexity(text: str, tok, model) -> float:
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