"""
Regression test for the failure mode that silently invalidated the POC.

`fit_leaves(jitter=…)` in the old library jittered only scalar-μ leaves, so the
K sibling subtrees of every mixture started identical, received identical
gradients, and stayed identical forever: the circuit degenerated into a product
of marginals while the training loss looked perfectly healthy.  The published
diagnostic (POC.md, Lesson 2) is that a learned and a random structure then
produce identical NLL.

That diagnostic is only meaningful if the implementation can, in principle,
show a difference — so this file asserts the two things that must both hold:

  1. DEPTH BUYS SOMETHING.  A K>1 circuit must beat a K=1 circuit (a pure
     product of marginals) on data whose dependence is not captured by any
     single factorisation.  If it does not, the mixtures are dead.
  2. STRUCTURE BUYS SOMETHING.  On unwhitened, strongly block-correlated data a
     learned region graph (Chow-Liu / ORC) must beat a random one.

Data is a two-component Gaussian mixture with block covariance: each component
puts strong within-block correlation and near-zero across-block correlation, so
a structure that cuts *between* blocks is genuinely better than one that cuts
through them, and no product of marginals can represent the mixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdf.circuits.einsum_pc import EinsumPC  # noqa: E402
from pcdf.circuits.structure import build_structure  # noqa: E402

N_BLOCKS, BLOCK = 3, 4
D = N_BLOCKS * BLOCK


def blocky_mixture(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((n, D), np.float32)
    comp = rng.integers(0, 2, size=n)
    for b in range(N_BLOCKS):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        shared = rng.normal(size=(n, 1))
        # component-dependent sign flips the within-block correlation, so the
        # joint is not a single Gaussian and not a product of marginals
        sign = np.where(comp[:, None] == 0, 1.0, -1.0)
        out[:, sl] = (sign * shared + 0.35 * rng.normal(size=(n, BLOCK)))
        out[:, sl] += (2.0 if b == 1 else 0.0) * (comp[:, None] - 0.5)
    return out


def fit_nll(X: np.ndarray, Xval: np.ndarray, structure, K: int,
            epochs: int = 120, seed: int = 0) -> float:
    torch.manual_seed(seed)
    pc = EinsumPC(structure, n_sum_components=K, n_input_components=K,
                  leaf_components=3, weight_jitter=0.5, seed=seed)
    Xt, Xv = torch.from_numpy(X), torch.from_numpy(Xval)
    pc.fit_leaves(Xt, jitter=0.2, seed=seed)
    opt = torch.optim.Adam(pc.parameters(), lr=0.03)
    best = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        (-pc.log_prob(Xt).mean()).backward()
        opt.step()
        with torch.no_grad():
            best = min(best, float(-pc.log_prob(Xv).mean()))
    assert abs(float(pc.log_partition())) < 1e-3, "normalization broke during training"
    return best


def test_mixture_capacity_is_alive():
    """K>1 must beat K=1: if the sum units collapsed, this fails."""
    X, Xval = blocky_mixture(1500, 0), blocky_mixture(1500, 1)
    rg = build_structure(X, method="chow_liu")
    nll_k1 = fit_nll(X, Xval, rg, K=1)
    nll_k8 = fit_nll(X, Xval, rg, K=8)
    gain = nll_k1 - nll_k8
    print(f"\nK=1 {nll_k1:.3f}  K=8 {nll_k8:.3f}  gain {gain:.3f} nats")
    assert gain > 0.2, (
        f"K=8 barely beats a product of marginals ({gain:.3f} nats): the "
        f"mixtures have collapsed — this is the POC's Lesson-2 failure")


def test_learned_structure_beats_random():
    """The Chow-Liu / ORC cut must beat a random one on block-correlated data."""
    X, Xval = blocky_mixture(1500, 0), blocky_mixture(1500, 1)
    results = {}
    for method in ("random", "chow_liu", "orc"):
        results[method] = fit_nll(X, Xval, build_structure(X, method=method,
                                                           seed=0), K=8)
    print("\n" + "  ".join(f"{k} {v:.3f}" for k, v in results.items()))
    best_learned = min(results["chow_liu"], results["orc"])
    assert best_learned < results["random"] - 0.05, (
        f"learned structure ({best_learned:.3f}) does not beat random "
        f"({results['random']:.3f}) — either the structure learner is not "
        f"finding the blocks or the circuit cannot exploit them")


if __name__ == "__main__":
    test_mixture_capacity_is_alive()
    test_learned_structure_beats_random()
    print("both structure regressions pass")
