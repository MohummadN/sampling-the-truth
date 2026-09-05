"""Decoding arms for the factuality-diversity grid.

Single source of truth mapping arm name -> generate() kwargs.
Every arm shares max_new_tokens=256; sampling arms set top_k=0
explicitly because HF defaults to top_k=50. Deterministic arms
carry explicit sampling fields (unused, triggers a harmless HF
warning) so nothing inherits from a model's generation_config.
"""

from transformers import GenerationConfig


# Every arm receives the same maximum generation length.
COMMON: dict[str, object] = dict(max_new_tokens=256)


# Explicit values prevent Llama's generation defaults from leaking in.
DETERMINISTIC: dict[str, object] = dict(
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
)


# top_k=0 disables top-k filtering on every sampling arm.
SAMPLING: dict[str, object] = dict(
    do_sample=True,
    top_k=0,
)


ARMS: dict[str, dict[str, object]] = {
    # Deterministic baseline with no inherited sampling settings.
    "greedy": {
        **DETERMINISTIC,
    },

    # Explicit length settings prevent beam-search length artifacts.
    "beam4": {
        **DETERMINISTIC,
        "num_beams": 4,
        "length_penalty": 1.0,
        "early_stopping": True,
    },

    # top_p=1.0 isolates the effect of low temperature.
    "temp0.7": {
        **SAMPLING,
        "temperature": 0.7,
        "top_p": 1.0,
    },

    # top_p=1.0 isolates the effect of high temperature.
    "temp1.3": {
        **SAMPLING,
        "temperature": 1.3,
        "top_p": 1.0,
    },

    # temperature=1.0 prevents Llama's temperature=0.6 from leaking in.
    "nucleus0.9": {
        **SAMPLING,
        "temperature": 1.0,
        "top_p": 0.9,
    },

    # The repetition penalty is matched by the greedy_reppen control.
    "dola": {
        **DETERMINISTIC,
        "dola_layers": "low",
        "repetition_penalty": 1.2,
    },

    # Five candidates support self-consistency and cost five times more.
    "sc_k5": {
        **SAMPLING,
        "temperature": 1.0,
        "top_p": 0.9,
        "num_return_sequences": 5,
    },

    # This control isolates DoLa's repetition-penalty effect.
    "greedy_reppen": {
        **DETERMINISTIC,
        "repetition_penalty": 1.2,
    },

    # This compares DoLa and nucleus under the same sampler.
    "dola_nucleus": {
        **SAMPLING,
        "temperature": 1.0,
        "top_p": 0.9,
        "dola_layers": "low",
    },
}


DESCRIBE_FIELDS = (
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "num_beams",
    "dola_layers",
    "repetition_penalty",
    "max_new_tokens",
    "num_return_sequences",
)


def kwargs_for(arm: str) -> dict[str, object]:
    """Return a fresh generate() kwargs dictionary for one arm."""
    if arm not in ARMS:
        valid = ", ".join(ARMS)
        raise KeyError(
            f"Unknown decoding arm {arm!r}. Valid arms: {valid}"
        )

    return {**COMMON, **ARMS[arm]}


def describe(model, arm: str) -> dict[str, object]:
    """Return the effective generation configuration for an arm."""
    cfg = GenerationConfig.from_model_config(model.config)
    cfg.update(**kwargs_for(arm))

    return {
        # Default None: dola_layers is absent on older transformers.
        field: getattr(cfg, field, None)
        for field in DESCRIBE_FIELDS
    }


if __name__ == "__main__":
    import torch

    from src.models import MODELS, load

    for model_name in MODELS:
        tokenizer, model = load(model_name)

        print(f"\nModel: {model_name}")
        for arm_name in ARMS:
            print(f"{arm_name:15} {describe(model, arm_name)}")

        del tokenizer
        del model
        torch.cuda.empty_cache()