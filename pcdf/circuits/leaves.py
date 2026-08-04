"""
Tensorized leaf layer.

The reference library instantiates one `GaussianMixtureLeaf` nn.Module per
(feature, unit) pair; a region graph over d = 4096 features with I = 8 input
units is 32k Python modules, and a forward pass is 32k Python calls.  This
layer holds the *same* parameters as one (L, M) tensor triple and evaluates all
of them in three fused ops.

Semantics are copied verbatim from `GaussianMixtureLeaf` so that parameters can
be moved between the two implementations and the outputs agree to float
precision:

    sigma          = softplus(log_sigma) + 1e-5
    log f(v)       = logsumexp_m [ log_softmax(logits)_m + N(v; mu_m, sigma_m) ]
    log ∫ f        = 0                       (normalized by construction)
    log ∫_lo^hi f  = logsumexp_m [ log_softmax(logits)_m + log(Φ(z_hi)−Φ(z_lo)) ]

The three query modes (observed / marginalized / boxed) are selected per
(sample, feature) by masks, which is what makes batched exact marginals — the
whole localization story — cost one forward pass per mask set instead of one
Python traversal per query.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_LOG_NDTR_NATIVE: dict = {}


def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(max(y, 1e-8))))


def log_ndtr(z: torch.Tensor) -> torch.Tensor:
    """
    log Φ(z), portable across backends.

    `torch.special.log_ndtr` is missing on Apple MPS, and it is exactly what the
    interval/box query needs — so a device without it would silently lose the
    censoring query rather than the whole model.  The fallback is the erfc form
    with the standard asymptotic branch for the far left tail, where
    erfc(-z/√2) underflows to zero:

        log Φ(z) ≈ −z²/2 − log(−z·√(2π))     for z ≪ 0
    """
    kind = z.device.type
    native = _LOG_NDTR_NATIVE.get(kind)
    if native is None:
        try:
            torch.special.log_ndtr(torch.zeros(1, device=z.device))
            native = True
        except Exception:                                      # noqa: BLE001
            native = False
        _LOG_NDTR_NATIVE[kind] = native
    if native:
        return torch.special.log_ndtr(z)
    body = torch.log(0.5 * torch.erfc(-z.clamp(min=-8.0) / _SQRT2) + 1e-45)
    tail_z = z.clamp(max=-8.0)
    tail = -0.5 * tail_z * tail_z - torch.log(-tail_z * math.sqrt(2.0 * math.pi))
    return torch.where(z < -8.0, tail, body)


class GaussianMixtureLeafLayer(nn.Module):
    """
    L leaf units, each a univariate M-component Gaussian mixture over the
    feature `feature_of_unit[l]`.

    Args:
        feature_of_unit: (L,) int array — which input coordinate each unit reads.
        n_components:    M, components per unit.
        n_features:      d, the input dimension.
    """

    def __init__(self, feature_of_unit: np.ndarray, n_components: int = 4,
                 n_features: Optional[int] = None):
        super().__init__()
        feat = torch.as_tensor(np.asarray(feature_of_unit), dtype=torch.long)
        self.register_buffer("feature_of_unit", feat)
        self.n_units = int(feat.numel())
        self.n_components = int(n_components)
        self.n_features = int(n_features if n_features is not None else feat.max() + 1)

        L, M = self.n_units, self.n_components
        self.mus = nn.Parameter(torch.linspace(-1.0, 1.0, M).repeat(L, 1))
        self.log_sigmas = nn.Parameter(torch.zeros(L, M))
        self.logits = nn.Parameter(torch.zeros(L, M))

    @property
    def sigmas(self) -> torch.Tensor:
        return F.softplus(self.log_sigmas) + 1e-5

    # ── initialisation ───────────────────────────────────────────────────

    @torch.no_grad()
    def fit(self, X: torch.Tensor, jitter: float = 0.2, seed: int = 0) -> None:
        """
        Quantile-spread init per feature + symmetry-breaking jitter per unit.

        The jitter is load-bearing, not cosmetic: `fit` is a deterministic
        function of the data, so the I units of a leaf region would start
        identical, receive identical gradients, and stay identical forever —
        every mixture in the circuit collapses to a product of marginals and
        the structure stops mattering.  (POC.md, Lesson 2: the failure is
        silent; the training loss never reveals it.)  Jittering *per unit*
        here is what the reference library's `_fit_leaves_with_jitter` does
        module-by-module.
        """
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        Xc = X.detach().cpu().float()
        d, M = self.n_features, self.n_components
        qs = torch.as_tensor(
            np.quantile(Xc.numpy(), np.linspace(0.1, 0.9, M), axis=0),
            dtype=torch.float32).T                               # (d, M)
        spread = (Xc.std(0) + 1e-6) / max(M, 1)                  # (d,)

        feat = self.feature_of_unit.cpu()
        mus = qs[feat].clone()                                   # (L, M)
        sig = spread[feat].clamp_min(1e-3)                       # (L,)
        log_sig = torch.tensor([_inv_softplus(float(s)) for s in sig])
        log_sigmas = log_sig[:, None].repeat(1, M)
        logits = torch.zeros_like(mus)

        if jitter > 0:
            eps = torch.randn(mus.shape, generator=gen)
            mus += eps * jitter * (F.softplus(log_sigmas) + 1e-6)
            logits += 0.1 * torch.randn(logits.shape, generator=gen)

        dev = self.mus.device
        self.mus.copy_(mus.to(dev))
        self.log_sigmas.copy_(log_sigmas.to(dev))
        self.logits.copy_(logits.to(dev))

    # ── queries ──────────────────────────────────────────────────────────

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """log f_l(x_{feature(l)}) for every unit.  x: (B, d) -> (B, L)."""
        v = x[:, self.feature_of_unit]                            # (B, L)
        s = self.sigmas                                           # (L, M)
        z = (v.unsqueeze(-1) - self.mus) / s                      # (B, L, M)
        comp = -0.5 * z * z - torch.log(s) - 0.5 * LOG_2PI
        return torch.logsumexp(F.log_softmax(self.logits, dim=-1) + comp, dim=-1)

    def log_interval(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """log ∫_lo^hi f_l for every unit.  lo/hi: (B, d) -> (B, L)."""
        s = self.sigmas
        zl = (lo[:, self.feature_of_unit].unsqueeze(-1) - self.mus) / s
        zh = (hi[:, self.feature_of_unit].unsqueeze(-1) - self.mus) / s
        # log(Φ(zh) − Φ(zl)) = log Φ(zh) + log(1 − exp(log Φ(zl) − log Φ(zh))),
        # computed through log_ndtr so the far tails stay finite
        log_hi, log_lo = log_ndtr(zh), log_ndtr(zl)
        delta = torch.clamp(log_lo - log_hi, max=-1e-12)
        log_mass = log_hi + torch.log1p(-torch.exp(delta))
        log_mass = torch.where(torch.isfinite(log_mass), log_mass,
                               torch.full_like(log_mass, -60.0))
        return torch.logsumexp(F.log_softmax(self.logits, dim=-1) + log_mass, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        boxes: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        (B, L) leaf log-values under the three query modes.

        observed: (B, d) or (1, d) bool — False marginalizes that feature
                  (leaf contributes log ∫ f = 0, exactly).
        boxes:    optional (lo, hi, box_mask) each (B, d); where box_mask is
                  True the leaf contributes log ∫_lo^hi f.
        """
        out = self.log_density(x)
        if boxes is not None:
            lo, hi, box_mask = boxes
            bm = box_mask[:, self.feature_of_unit]
            if bm.any():
                out = torch.where(bm, self.log_interval(lo, hi), out)
        if observed is not None:
            obs = observed[:, self.feature_of_unit]
            out = torch.where(obs, out, torch.zeros_like(out))
        return out

    # ── parameter exchange with the reference implementation ─────────────

    @torch.no_grad()
    def load_from_reference(self, leaf_modules) -> None:
        """Copy parameters from a list of reference `GaussianMixtureLeaf`s."""
        for i, m in enumerate(leaf_modules):
            self.mus[i].copy_(m.mus.detach())
            self.log_sigmas[i].copy_(m.log_sigmas.detach())
            self.logits[i].copy_(m.logits.detach())

    @torch.no_grad()
    def store_to_reference(self, leaf_modules) -> None:
        for i, m in enumerate(leaf_modules):
            m.mus.copy_(self.mus[i].detach().cpu())
            m.log_sigmas.copy_(self.log_sigmas[i].detach().cpu())
            m.logits.copy_(self.logits[i].detach().cpu())
