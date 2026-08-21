"""
The family mixture must stay a CIRCUIT, not become an ensemble.

What is actually at risk here: `p_mix = Σ_f π_f p_f` is only a valid circuit —
and the score only a true log-likelihood ratio — if every component is exactly
normalized over the SAME scope and shares the region graph.  If a component
silently gets its own structure, or if the mixture weights do not sum to one,
the "exact log-odds" claim quietly becomes false while every number still looks
plausible.  These tests pin that down.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pcdf.models.density_pc import PCConfig
from pcdf.models.family_mixture import (FamilyMixtureRatio,
                                        expected_calibration_error,
                                        risk_coverage)

GH = GW = 2
C = 4
D = GH * GW * C
FAMS = ["blend", "render"]


def _cfg() -> PCConfig:
    return PCConfig(device="cpu", seed=0, n_sum_components=2,
                    n_input_components=2, leaf_components=2, epochs=2,
                    batch_size=64, lr=1e-2, patience=5,
                    patch_method="kd", channel_method="chow_liu")


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    Zr = rng.normal(0, 1, (400, D)).astype(np.float32)
    Zf = {"blend": rng.normal(0.6, 1.0, (300, D)).astype(np.float32),
          "render": rng.normal(-0.6, 1.3, (300, D)).astype(np.float32)}
    mix = FamilyMixtureRatio(GH, GW, C, FAMS, _cfg())
    mix.fit(Zr, Zf, Zr[:100], verbose=False)
    return mix


def test_components_share_one_region_graph(fitted):
    """Any difference between components must come from parameters alone."""
    rg = fitted.real.region_graph
    for f in FAMS:
        assert fitted.comp[f].region_graph is rg
        assert fitted.comp[f].patch_regions is fitted.real.patch_regions


def test_every_component_is_normalized(fitted):
    """log Z = 0 for p_real and for each p_f, so the ratio is a real log-odds."""
    for pc in [fitted.real.pc] + [fitted.comp[f].pc for f in FAMS]:
        assert abs(float(pc.log_partition())) < 1e-3


def test_mixture_weights_are_a_distribution(fitted):
    assert np.isclose(np.exp(fitted.log_pi).sum(), 1.0, atol=1e-6)


def test_posterior_is_a_distribution_and_matches_the_ratio(fitted):
    rng = np.random.default_rng(1)
    Z = rng.normal(0, 1, (32, D)).astype(np.float32)
    s = fitted.score(Z)
    post = s["family_posterior"]
    assert post.shape == (32, len(FAMS))
    assert np.allclose(post.sum(1), 1.0, atol=1e-5)
    assert (post >= 0).all()

    # the detector score must equal logsumexp(log pi + log p_f) - log p_real
    L = fitted.component_log_probs(Z)
    expect = (torch.logsumexp(
        torch.from_numpy(L[:, :-1] + fitted.log_pi[None, :]), dim=1).numpy()
        - L[:, -1])
    assert np.allclose(s["ratio_mixture"], expect, atol=1e-4)


def test_mixture_dominates_its_own_best_component(fitted):
    """logsumexp >= max, so the mixture ratio can never be below the best one."""
    rng = np.random.default_rng(2)
    Z = rng.normal(0, 1, (24, D)).astype(np.float32)
    s = fitted.score(Z)
    assert (s["ratio_mixture"] >= s["ratio_best_family"] - 1e-5).all()


def test_region_posterior_is_exact_and_normalized(fitted):
    """
    Per-region mechanism attribution: (n, P, F), a distribution at every region.

    Also checks the marginals it is built from are the real thing — the region
    marginal of a full-scope observation must equal log p(z) when the region is
    the whole image, which is the cheapest available proof that
    `region_log_marginals` is not returning something merely plausible.
    """
    rng = np.random.default_rng(3)
    Z = rng.normal(0, 1, (8, D)).astype(np.float32)
    R = fitted.region_family_posterior(Z, batch=4)
    assert R.shape == (8, GH * GW, len(FAMS))
    assert np.allclose(R.sum(-1), 1.0, atol=1e-5)

    x = torch.from_numpy(Z).to(fitted.cfg.device)
    full = torch.ones(1, D, dtype=torch.bool, device=fitted.cfg.device)
    marg = fitted.real.pc.region_log_marginals(x, full).squeeze(1)
    assert torch.allclose(marg, fitted.real.pc.log_prob(x), atol=1e-4)


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 40000)
    y = (rng.uniform(0, 1, 40000) < p).astype(int)
    assert expected_calibration_error(p, y)["ece"] < 0.02


def test_ece_catches_overconfidence():
    rng = np.random.default_rng(5)
    p = np.full(4000, 0.99)
    y = (rng.uniform(0, 1, 4000) < 0.5).astype(int)
    assert expected_calibration_error(p, y)["ece"] > 0.4


def test_risk_decreases_as_we_abstain_more():
    """A usable confidence must buy accuracy: risk at 100% coverage should be
    no better than risk at 20%."""
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 3000)
    s = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.4, 3000)
    rc = risk_coverage(s, y)
    assert rc[0]["coverage"] == pytest.approx(1.0, abs=1e-6)
    assert rc[-1]["risk"] <= rc[0]["risk"]
