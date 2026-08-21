"""
`EinsumPC.log_ball` must be an exact probability MASS, not an approximation.

Three properties pin it, and each one would catch a different bug:

  small-eps   log P(box) - [log p(x) + sum_i log 2*eps_i] -> 0.
              Catches a wrong leaf CDF, a dropped Jacobian, or a box applied to
              the wrong feature: for a tiny box the mass is the density times
              the volume, and nothing else.

  large-eps   log P(box) -> 0 as the box swallows the support.
              Catches an unnormalized circuit or a leaf whose CDF does not
              reach 1 — the same property `log_partition` checks, reached from
              the other direction.

  monotone    a bigger box never has less mass.
              Catches sign errors and tail underflow in log(Phi(hi) - Phi(lo)),
              which is where a naive implementation breaks first.
"""
from __future__ import annotations

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdf.circuits.einsum_pc import EinsumPC  # noqa: E402
from pcdf.circuits.structure import build_structure  # noqa: E402


def _pc(d: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(256, 3))
    X = np.concatenate([z + 0.3 * rng.normal(size=(256, 3)),
                        2 * z + 0.3 * rng.normal(size=(256, 3)),
                        rng.normal(size=(256, d - 6))], axis=1).astype(np.float32)
    rg = build_structure(X, method="orc", max_arity=4, seed=seed)
    pc = EinsumPC(rg, n_sum_components=4, n_input_components=4,
                  leaf_components=3, seed=seed)
    pc.fit_leaves(torch.from_numpy(X))
    pc.eval()
    return pc, torch.from_numpy(X[:16])


# Measured float32 behaviour of |log P(box) - (log p + log vol)| (see
# `test_small_box_is_density_times_volume`):
#
#     eps    1e-1     1e-2     1e-3     1e-4     1e-5     1e-6
#     err    4.4e-1   4.7e-3   2.8e-4   1.8e-3   2.2e-2   5.3e-1
#
# A U: truncation error dominates above (the box is wide enough that the
# density is not constant across it), and cancellation in
# log(Phi(hi) - Phi(lo)) dominates below, where Phi(hi) - Phi(lo) ~ 2*eps*phi(x)
# is a difference of two nearly equal float32 numbers.  The identity is exact;
# float32 is not.  **Do not use eps below ~1e-3 in float32** — the score
# silently degrades rather than failing.
EPS_SWEET_SPOT = 1e-3
EPS_FLOOR = 1e-3


@torch.no_grad()
def test_small_box_is_density_times_volume():
    """log P(box) -> log p(x) + log vol as the box shrinks, down to the
    precision floor.  Catches a wrong leaf CDF, a dropped volume term, or a box
    applied to the wrong feature."""
    pc, x = _pc()
    d = x.shape[1]
    logp = pc.log_prob(x)

    def err(eps):
        return (pc.log_ball(x, eps) - (logp + d * np.log(2 * eps))).abs().max().item()

    # converging while truncation error dominates
    e1, e2, e3 = err(1e-1), err(1e-2), err(1e-3)
    assert e2 < e1, f"not converging: {e2} !< {e1}"
    assert e3 < e2, f"not converging: {e3} !< {e2}"
    # and the identity actually holds at the sweet spot
    assert e3 < 1e-3, f"log P(box) != log p(x) + log vol at eps=1e-3: {e3}"


@torch.no_grad()
def test_precision_floor_is_where_we_think_it_is():
    """Pins the lower end of the U.  If a future change (float64 leaves, a
    different CDF) moves this, the guidance in `log_ball` must move with it."""
    pc, x = _pc()
    d = x.shape[1]
    logp = pc.log_prob(x)

    def err(eps):
        return (pc.log_ball(x, eps) - (logp + d * np.log(2 * eps))).abs().max().item()

    assert err(1e-6) > err(EPS_SWEET_SPOT), \
        "no precision floor found — the guidance in log_ball may be stale"


@torch.no_grad()
def test_huge_box_has_unit_mass():
    pc, x = _pc()
    m = pc.log_ball(x, 1e4)
    assert m.abs().max().item() < 1e-4, f"total mass != 1: max |log P| = {m.abs().max()}"


@torch.no_grad()
def test_mass_is_monotone_in_eps():
    pc, x = _pc()
    ms = [pc.log_ball(x, e) for e in (1e-3, 1e-2, 1e-1, 1.0, 10.0)]
    for a, b in zip(ms, ms[1:]):
        assert (b >= a - 1e-6).all(), "a larger box lost mass"


@torch.no_grad()
def test_per_dimension_eps_matches_scalar():
    pc, x = _pc()
    d = x.shape[1]
    scalar = pc.log_ball(x, 0.1)
    vector = pc.log_ball(x, torch.full((d,), 0.1))
    matrix = pc.log_ball(x, torch.full(x.shape, 0.1))
    assert torch.allclose(scalar, vector, atol=1e-6)
    assert torch.allclose(scalar, matrix, atol=1e-6)
