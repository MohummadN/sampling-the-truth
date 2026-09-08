"""Device selection, shared by every module that loads a model."""
from __future__ import annotations

import torch


def pick_device() -> str:
    """CUDA only if this torch build has kernels this GPU can actually run.

    CUDA guarantees binary compatibility upward within a major version: a cubin
    built for sm_X.y runs on sm_X.z for z >= y. The cluster's TITAN Xp is sm_61
    and the cu126 wheel ships sm_60, which is therefore usable — a plain
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
