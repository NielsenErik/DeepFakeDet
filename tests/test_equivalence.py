"""
The contract test: EinsumPC must be the SAME circuit as the reference
`RegionGraphPC`, only faster.

Speed claims about a density model are worthless if the density changed, so
every structure family this project uses (Chow-Liu vtree, Ollivier-Ricci
region graph, multi-partition ORC, spectral, and the hierarchical image graph)
is built once, parameters are copied reference -> tensorized, and the two
implementations are compared on:

    log p(x)                exact density
    log p(x_O)              exact marginals, over random observation masks
    log Z                   normalization
    box query               censored / interval evidence

Tolerances are float32 round-off, not "close enough" fudge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdf.circuits.einsum_pc import EinsumPC  # noqa: E402
from pcdf.circuits.structure import build_structure  # noqa: E402
from pcdf.pclib import (  # noqa: E402
    GaussianMixtureLeaf,
    RegionGraphPC,
    eval_log_marginal,
    is_structured_decomposable_rg,
)

TOL = 2e-4


def _data(n=64, d=12, seed=0):
    rng = np.random.default_rng(seed)
    # correlated blocks so the structure learners have something to find
    z = rng.normal(size=(n, 3))
    X = np.concatenate([z + 0.3 * rng.normal(size=(n, 3)),
                        2 * z + 0.3 * rng.normal(size=(n, 3)),
                        rng.normal(size=(n, d - 6))], axis=1)
    return X.astype(np.float32)


def _pair(rg, K=3, I=3, M=2, seed=0):
    ref = RegionGraphPC(rg, n_sum_components=K, n_input_components=I,
                        leaf_factory=lambda i: GaussianMixtureLeaf(i, M),
                        weight_jitter=0.7, seed=seed)
    mine = EinsumPC(rg, n_sum_components=K, n_input_components=I,
                    leaf_components=M, weight_jitter=0.7, seed=seed)
    X = torch.from_numpy(_data(seed=seed + 1))
    ref.fit_leaves(X, jitter=0.3)
    mine.load_from_reference(ref)
    return ref, mine, X


STRUCTURES = ["chow_liu", "orc", "orc_multi", "spectral", "random", "forman"]


@pytest.mark.parametrize("method", STRUCTURES)
def test_matches_reference(method):
    X = _data()
    rg = build_structure(X, method=method)
    ref, mine, Xt = _pair(rg)
    x = Xt[:16]

    lp_ref = ref.log_prob(x)
    lp_mine = mine.log_prob(x)
    assert torch.allclose(lp_ref, lp_mine, atol=TOL, rtol=0), \
        f"{method}: max |Δ log p| = {(lp_ref - lp_mine).abs().max():.2e}"

    # exact marginals over random observation masks
    rng = np.random.default_rng(0)
    d = X.shape[1]
    for _ in range(5):
        marg = sorted(rng.choice(d, size=rng.integers(1, d), replace=False).tolist())
        m_ref = eval_log_marginal(ref.root, x, marg)
        m_mine = mine.log_marginal(x, marg)
        assert torch.allclose(m_ref, m_mine, atol=TOL, rtol=0), \
            f"{method}: marginal mismatch {(m_ref - m_mine).abs().max():.2e}"

    # normalization
    assert abs(float(mine.log_partition())) < TOL
    assert abs(float(ref.log_partition())) < TOL

    # box / censoring query
    boxes = {0: (-0.5, 1.0), 3: (-float("inf"), 0.2)}
    b_ref = eval_log_marginal(ref.root, x, (), boxes=boxes)
    b_mine = mine.log_box(x, boxes)
    assert torch.allclose(b_ref, b_mine, atol=TOL, rtol=0)


def test_structured_decomposability_flags():
    X = _data()
    assert is_structured_decomposable_rg(build_structure(X, method="chow_liu"))
    assert is_structured_decomposable_rg(build_structure(X, method="orc"))
    # multi-partition regions deliberately give it up — and must say so
    assert not is_structured_decomposable_rg(build_structure(X, method="orc_multi"))


def test_batched_region_marginals_match_loop():
    X = _data()
    rg = build_structure(X, method="chow_liu")
    _, mine, Xt = _pair(rg)
    x = Xt[:8]
    d = X.shape[1]
    masks = torch.zeros(4, d, dtype=torch.bool)
    for q in range(4):
        masks[q, q * 3:(q + 1) * 3] = True
    batched = mine.region_log_marginals(x, masks)
    for q in range(4):
        marg = [i for i in range(d) if not masks[q, i]]
        assert torch.allclose(batched[:, q], mine.log_marginal(x, marg), atol=TOL)


def test_gradients_flow_and_normalization_is_preserved_after_training():
    X = _data(n=256)
    rg = build_structure(X, method="orc")
    mine = EinsumPC(rg, n_sum_components=4, n_input_components=4, leaf_components=3)
    Xt = torch.from_numpy(X)
    mine.fit_leaves(Xt)
    opt = torch.optim.Adam(mine.parameters(), lr=0.05)
    first = None
    for _ in range(30):
        opt.zero_grad()
        nll = -mine.log_prob(Xt).mean()
        nll.backward()
        opt.step()
        first = first if first is not None else float(nll)
    assert float(nll) < first, "training did not reduce NLL"
    assert abs(float(mine.log_partition())) < TOL, "training broke normalization"
    mine.validate(Xt[:8])
