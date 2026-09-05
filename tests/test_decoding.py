import pytest

from src.decoding import ARMS, kwargs_for


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