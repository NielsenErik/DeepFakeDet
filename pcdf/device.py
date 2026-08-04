"""
Device resolution.

Every stage takes `device` from the config, and the default is `"auto"`: CUDA
if it is genuinely usable, else Apple MPS, else CPU.  "Genuinely usable" is not
`torch.cuda.is_available()` alone — this project has already hit the case where
the driver reports a device but any allocation fails (an unattended driver
upgrade left a stale kernel module, CUDA error 804).  So the probe actually
allocates, and a failure downgrades with a loud message instead of crashing a
six-hour pipeline on its first batch.

Explicit values (`cuda`, `cuda:1`, `mps`, `cpu`) are honoured as given; if an
explicit CUDA request is not usable, that is an error, not a silent fallback —
asking for a GPU and quietly getting a 40× slower CPU run is worse than
stopping.
"""
from __future__ import annotations

from typing import Optional

import torch

_RESOLVED: Optional[str] = None


def cuda_usable(index: int = 0) -> tuple[bool, str]:
    """(usable, reason) — probes with a real allocation, not just a flag."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        torch.zeros(8, device=f"cuda:{index}").sum().item()
        return True, torch.cuda.get_device_name(index)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


def mps_usable() -> bool:
    """
    MPS counts as usable only if the ops the circuit needs exist on it.
    `special.log_ndtr` (the interval/box query) was missing as of torch 2.10,
    and `pcdf.circuits.leaves.log_ndtr` now supplies a fallback — so this probe
    exercises the fallback path rather than the raw op, and stays honest if a
    future op goes missing too.
    """
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return False
    try:
        from .circuits.leaves import log_ndtr

        z = torch.tensor([-9.0, 0.0, 2.0], device="mps")
        out = log_ndtr(z)
        return bool(torch.isfinite(out).all())
    except Exception:                                          # noqa: BLE001
        return False


def resolve_device(spec: str = "auto", verbose: bool = True) -> str:
    """Map a config `device` value to a concrete torch device string."""
    global _RESOLVED
    spec = (spec or "auto").strip()

    if spec != "auto":
        if spec.startswith("cuda"):
            index = int(spec.split(":")[1]) if ":" in spec else 0
            ok, reason = cuda_usable(index)
            if not ok:
                raise RuntimeError(
                    f"device={spec!r} was requested but CUDA is not usable: {reason}. "
                    f"Use device=auto to fall back to CPU deliberately.")
        return spec

    ok, reason = cuda_usable()
    if ok:
        dev = "cuda"
        msg = f"[device] using CUDA ({reason})"
    elif mps_usable():
        dev = "mps"
        msg = "[device] using Apple MPS (no usable CUDA)"
    else:
        dev = "cpu"
        msg = f"[device] falling back to CPU — CUDA not usable: {reason}"
    if verbose and _RESOLVED != dev:
        print(msg, flush=True)
    _RESOLVED = dev
    return dev


def autocast_kwargs(device: str, enabled: bool = True) -> dict:
    """
    Arguments for `torch.amp.autocast` / `GradScaler` that are valid on this
    device.  Mixed precision is CUDA-only here: circuit evaluation is done in
    log space where fp16 range matters, and CPU autocast (bfloat16) buys
    nothing for these shapes.
    """
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return {"device_type": "cuda" if device.startswith("cuda") else "cpu",
            "enabled": bool(enabled and device.startswith("cuda")),
            "dtype": dtype}


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
