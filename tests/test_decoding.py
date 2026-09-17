import os

import pytest

from src.decoding import ARMS, describe, kwargs_for
from src.models import load


@pytest.mark.skipif(not os.environ.get("SLOW"), reason="set SLOW=1 to load Llama")
def test_arms_resolve_identically_across_models():
    """The point of setting every field explicitly: gpt2 and Llama ship
    different generation_config.json defaults, so an unset field would resolve
    differently per model and the cross-model comparison would be invalid."""
    _, gpt2 = load("gpt2")
    _, llama = load("llama-1b")
    for arm in ARMS:
        assert describe(gpt2, arm) == describe(llama, arm), arm

def test_every_arm_resolves() -> None:
    for arm in ARMS:
        kwargs = kwargs_for(arm)
        assert kwargs["max_new_tokens"] == 256


def test_every_arm_sets_generation_controls_explicitly() -> None:
    required = {
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
    }

    for arm in ARMS:
        kwargs = kwargs_for(arm)
        missing = required - kwargs.keys()
        assert not missing, f"{arm} is missing explicit fields: {missing}"


def test_kwargs_for_returns_a_copy() -> None:
    first = kwargs_for("greedy")
    first["max_new_tokens"] = 1
    first["temperature"] = 99.0

    second = kwargs_for("greedy")

    assert second["max_new_tokens"] == 256
    assert second["temperature"] == 1.0


def test_unknown_arm_raises_clear_error() -> None:
    with pytest.raises(KeyError, match="Unknown decoding arm"):
        kwargs_for("wrong")

def test_sampling_arms_disable_top_k() -> None:
    for arm in ARMS:
        kwargs = kwargs_for(arm)
        if kwargs["do_sample"]:
            assert kwargs["top_k"] == 0, f"{arm}: top_k must be 0, got {kwargs['top_k']}"

def test_grid_default_excludes_dev_only_arms():
    """dola_high and dola_nucleus_high are dev-only sensitivity arms. If they
    reach generate.py's --arms default, a re-run of any shard silently writes
    11 arms into the frozen 8,100-record grid."""
    from src.decoding import GRID_ARMS
    from src.generate import DEFAULT_ARMS
    assert len(GRID_ARMS) == 9
    assert not ({"dola_high", "dola_nucleus_high"} & set(GRID_ARMS))
    assert DEFAULT_ARMS.split(",") == list(GRID_ARMS)


def test_dev_only_arms_still_resolve():
    """Keeping them out of the default must not make them unrunnable."""
    for arm in ("dola_high", "dola_nucleus_high"):
        assert kwargs_for(arm)["dola_layers"] == "high"
