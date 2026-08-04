"""
Device portability.

The pipeline is developed on a Mac and run on a CUDA workstation, and it has
already been bitten by both failure modes this file guards:

  * a driver upgrade left `torch.cuda.is_available()` True while every
    allocation failed (CUDA error 804), so availability flags are not enough —
    `resolve_device` must probe with a real allocation;
  * `torch.special.log_ndtr` is missing on MPS, which would have silently cost
    the interval/box query rather than failing loudly at import.

So: whatever device is chosen, the circuit must produce the same numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdf.circuits.einsum_pc import EinsumPC  # noqa: E402
from pcdf.circuits.leaves import log_ndtr  # noqa: E402
from pcdf.circuits.structure import build_structure  # noqa: E402
from pcdf.device import cuda_usable, resolve_device  # noqa: E402


def available_devices():
    devs = ["cpu"]
    if cuda_usable()[0]:
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def test_resolve_device_honours_explicit_values():
    assert resolve_device("cpu", verbose=False) == "cpu"
    assert resolve_device("auto", verbose=False) in ("cpu", "cuda", "mps")
    if not cuda_usable()[0]:
        # asking for a GPU that cannot run must FAIL, not silently downgrade:
        # a six-hour run at 40× slower is worse than an error
        with pytest.raises(RuntimeError):
            resolve_device("cuda", verbose=False)


def test_log_ndtr_fallback_matches_native():
    z = torch.tensor([-40.0, -12.0, -8.0, -3.0, 0.0, 1.5, 6.0])
    from pcdf.circuits import leaves

    leaves._LOG_NDTR_NATIVE["cpu"] = False        # force the fallback branch
    try:
        fallback = log_ndtr(z)
    finally:
        leaves._LOG_NDTR_NATIVE.pop("cpu", None)
    native = torch.special.log_ndtr(z)
    assert torch.allclose(fallback, native, atol=1e-5, rtol=1e-4), \
        f"max err {(fallback - native).abs().max():.2e}"


def test_same_numbers_on_every_available_device():
    X = np.random.default_rng(0).normal(size=(96, 8)).astype(np.float32)
    rg = build_structure(X, method="chow_liu")
    ref = None
    for dev in available_devices():
        torch.manual_seed(0)
        pc = EinsumPC(rg, n_sum_components=3, n_input_components=3,
                      leaf_components=2, seed=0).to(dev)
        pc.fit_leaves(torch.from_numpy(X), seed=0)
        x = torch.from_numpy(X[:8]).to(dev)
        out = {
            "log_prob": pc.log_prob(x).detach().cpu(),
            "marginal": pc.log_marginal(x, [0, 3, 5]).detach().cpu(),
            "box": pc.log_box(x, {0: (-0.5, 1.0), 2: (-float("inf"), 0.3)}).detach().cpu(),
            "logZ": torch.tensor([float(pc.log_partition())]),
        }
        if ref is None:
            ref = out
            continue
        for k in ref:
            assert torch.allclose(ref[k], out[k], atol=1e-4), \
                f"{dev} disagrees with cpu on {k}: max |Δ| = {(ref[k] - out[k]).abs().max():.2e}"
