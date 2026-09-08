"""Generator loading. One place, so all three models are configured identically."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.device import pick_device

MODELS = {
    "gpt2":     "gpt2",
    "llama-1b": "meta-llama/Llama-3.2-1B",
    "llama-3b": "meta-llama/Llama-3.2-3B",
}

DTYPE = torch.float16      # see DECISIONS.md — same for every model, or
                           # precision gets confounded with scale


def load(name: str, dtype=DTYPE, device: str | None = None):
    """Load a generator. Returns (tok, model), eval mode."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; expected one of {sorted(MODELS)}")
    hf_id = MODELS[name]

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token      # gpt2 and base Llama ship none
    tok.padding_side = "left"              # REQUIRED for decoder-only generation

    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
    model.to(device or pick_device()).eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return tok, model


def revision(model) -> str:
    """The exact model snapshot, for the JSONL record."""
    return getattr(model.config, "_commit_hash", None) or "unknown"
