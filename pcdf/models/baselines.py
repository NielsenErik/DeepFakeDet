"""
Real-only (one-class) baselines, fitted in the IDENTICAL feature space as the
circuit and scored with the identical aggregation.

This is the part of the study that decides whether the project is worth
pursuing, so the competitors are the actual state of the art for
"density/normality model over frozen patch features", not strawmen:

  Mahalanobis      Ledoit-Wolf shrunk full covariance.  Beat the PC in the POC.
  GMM-full         mixture of full-covariance Gaussians — itself a shallow PC
                   with multivariate leaves, which is exactly why it is the
                   sharpest comparison: it isolates what DEEP structure adds.
  PatchCore        coreset memory bank + kNN, the standard-setter in industrial
                   anomaly detection (Roth et al., CVPR 2022).  Non-parametric,
                   no likelihood, extremely strong on localized defects — the
                   hardest baseline to beat on localization.
  RealNVP flow     exact likelihood, no tractable marginals.  The scientifically
                   decisive comparison: if the flow matches the circuit on
                   detection, then the circuit's contribution is exactly the
                   query class it supports (marginals / conditionals /
                   localization), and that must be stated plainly rather than
                   dressed up as an accuracy win.
  KDE / kNN        cheap sanity references.

Every model exposes fit(Z) / score(Z) over patch-shaped arrays (N, P, c) and
returns the same aggregation family (image-level and per-patch), so no
comparison is confounded by aggregation choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..device import resolve_device


def _flat(Z: np.ndarray) -> np.ndarray:
    return Z.reshape(len(Z), -1)


def _aggregate(patch_scores: np.ndarray, topq: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Image scores from per-patch scores, matched to the circuit's family —
    including the two-sided variants, because a comparison in which only the
    circuit may score "too typical" regions would be rigged.
    """
    P = patch_scores.shape[1]
    k = max(1, int(round(topq * P)))
    srt = np.sort(patch_scores, axis=1)
    absz = np.abs(patch_scores)
    return {
        "patch_max": srt[:, -1],
        "patch_topq": srt[:, -k:].mean(1),
        "patch_mean": patch_scores.mean(1),
        "patch_absmax": absz.max(1),
        "patch_abstopq": np.sort(absz, axis=1)[:, -k:].mean(1),
        "patch_lowmax": (-patch_scores).max(1),
        "incoherence": patch_scores.std(1),
        "_patch": patch_scores,
    }


class BaseDetector:
    name = "base"
    device = "auto"

    # Fitted models are pickled and re-loaded by a later stage that may run on a
    # different device (or on a machine whose GPU is temporarily unusable), so
    # tensor state is always stored on CPU and moved at score time.
    def _dev(self) -> str:
        resolved = getattr(self, "_resolved_device", None)
        if resolved is None:
            resolved = self._resolved_device = resolve_device(self.device, verbose=False)
        return resolved

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_resolved_device", None)
        for key, val in list(state.items()):
            if isinstance(val, torch.Tensor):
                state[key] = val.detach().cpu()
            elif isinstance(val, nn.Module):
                state[key] = val.to("cpu")
        return state

    def fit(self, Z: np.ndarray) -> "BaseDetector":  # (N,P,c) real train
        raise NotImplementedError

    def score(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def calibrate(self, Zref: np.ndarray) -> None:
        """Per-patch standardization on real reference data — the same
        treatment the circuit's scores get, so the comparison is fair."""
        s = self._patch_scores(Zref)
        self._mu, self._sd = s.mean(0), s.std(0) + 1e-6

    def _standardize(self, s: np.ndarray) -> np.ndarray:
        if hasattr(self, "_mu"):
            return (s - self._mu) / self._sd
        return s


# ── Gaussian family ─────────────────────────────────────────────────────────

class MahalanobisDetector(BaseDetector):
    name = "mahalanobis"

    def __init__(self, per_patch: bool = True):
        self.per_patch = per_patch

    def fit(self, Z: np.ndarray) -> "MahalanobisDetector":
        from sklearn.covariance import LedoitWolf

        if self.per_patch:
            N, P, c = Z.shape
            # one shared shrunk covariance over patch vectors, plus per-patch
            # means: the spatial layout of a face is real structure, ignoring
            # it would handicap the baseline unfairly
            self.mu_ = Z.mean(0)                                  # (P,c)
            self.lw_ = LedoitWolf().fit((Z - self.mu_).reshape(-1, c))
        else:
            self.lw_ = LedoitWolf().fit(_flat(Z))
        return self

    def _patch_scores(self, Z: np.ndarray) -> np.ndarray:
        N, P, c = Z.shape
        return self.lw_.mahalanobis((Z - self.mu_).reshape(-1, c)).reshape(N, P)

    def score(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.per_patch:
            s = self.lw_.mahalanobis(_flat(Z))
            return {"image": s, "patch_max": s, "patch_topq": s, "patch_mean": s}
        return _aggregate(self._standardize(self._patch_scores(Z)))


class GmmDetector(BaseDetector):
    name = "gmm"

    def __init__(self, n_components: int = 8, per_patch: bool = True, seed: int = 0):
        self.n_components, self.per_patch, self.seed = n_components, per_patch, seed

    def fit(self, Z: np.ndarray) -> "GmmDetector":
        from sklearn.mixture import GaussianMixture

        X = Z.reshape(-1, Z.shape[-1]) if self.per_patch else _flat(Z)
        if len(X) > 200_000:
            X = X[np.random.default_rng(self.seed).choice(len(X), 200_000, replace=False)]
        self.gmm_ = GaussianMixture(self.n_components, covariance_type="full",
                                    reg_covar=1e-4, random_state=self.seed).fit(X)
        return self

    def _patch_scores(self, Z: np.ndarray) -> np.ndarray:
        N, P, c = Z.shape
        return -self.gmm_.score_samples(Z.reshape(-1, c)).reshape(N, P)

    def score(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.per_patch:
            s = -self.gmm_.score_samples(_flat(Z))
            return {"image": s, "patch_max": s, "patch_topq": s, "patch_mean": s}
        return _aggregate(self._standardize(self._patch_scores(Z)))


# ── PatchCore ───────────────────────────────────────────────────────────────

class PatchCoreDetector(BaseDetector):
    """
    Greedy-coreset memory bank of real patch features + k-NN distance.
    (Roth et al., CVPR 2022.)  No density, no likelihood — and still the model
    to beat on localized anomalies, which is the regime this project claims.
    """
    name = "patchcore"

    def __init__(self, coreset_frac: float = 0.02, k: int = 5, device: str = "auto",
                 max_bank: int = 200_000, seed: int = 0):
        self.coreset_frac, self.k, self.device = coreset_frac, k, device
        self.max_bank, self.seed = max_bank, seed

    def fit(self, Z: np.ndarray) -> "PatchCoreDetector":
        X = torch.from_numpy(Z.reshape(-1, Z.shape[-1]).astype(np.float32))
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_bank:
            X = X[torch.from_numpy(rng.choice(len(X), self.max_bank, replace=False))]
        n_keep = max(16, int(len(X) * self.coreset_frac))
        self.bank_ = self._greedy_coreset(X.to(self._dev()), n_keep).cpu()
        return self

    @torch.no_grad()
    def _greedy_coreset(self, X: torch.Tensor, n_keep: int) -> torch.Tensor:
        """k-center greedy: maximally spread subset, the standard PatchCore
        reduction (approximate but deterministic given the seed)."""
        idx = [int(torch.randint(len(X), (1,), generator=torch.Generator().manual_seed(0)))]
        dist = torch.cdist(X, X[idx[-1]:idx[-1] + 1]).squeeze(1)
        for _ in range(n_keep - 1):
            i = int(torch.argmax(dist))
            idx.append(i)
            dist = torch.minimum(dist, torch.cdist(X, X[i:i + 1]).squeeze(1))
        return X[idx].contiguous()

    @torch.no_grad()
    def _patch_scores(self, Z: np.ndarray) -> np.ndarray:
        N, P, c = Z.shape
        dev = self._dev()
        X = torch.from_numpy(Z.reshape(-1, c).astype(np.float32)).to(dev)
        bank = self.bank_.to(dev)
        out = []
        for i in range(0, len(X), 65536):
            d = torch.cdist(X[i:i + 65536], bank)
            out.append(d.topk(self.k, largest=False).values.mean(1).cpu())
        return torch.cat(out).numpy().reshape(N, P)

    def score(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        return _aggregate(self._standardize(self._patch_scores(Z)))


# ── Normalizing flow (exact likelihood, no marginals) ──────────────────────

class _CouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim * 2))

    def forward(self, x: torch.Tensor):
        xm = x * self.mask
        s, t = self.net(xm).chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(-1)


class FlowDetector(BaseDetector):
    """
    RealNVP over patch vectors: an exact-likelihood density model with the same
    inputs as the circuit and NO tractable marginals.  It is the control that
    separates "density estimation works here" from "circuits work here".
    """
    name = "flow"

    def __init__(self, n_layers: int = 8, hidden: int = 256, epochs: int = 40,
                 lr: float = 1e-3, batch: int = 4096, device: str = "auto", seed: int = 0):
        self.n_layers, self.hidden, self.epochs = n_layers, hidden, epochs
        self.lr, self.batch, self.device, self.seed = lr, batch, device, seed

    def fit(self, Z: np.ndarray) -> "FlowDetector":
        torch.manual_seed(self.seed)
        c = Z.shape[-1]
        masks = []
        for i in range(self.n_layers):
            m = torch.zeros(c)
            m[i % 2::2] = 1.0
            masks.append(m)
        dev = self._dev()
        self.layers_ = nn.ModuleList([_CouplingLayer(c, self.hidden, m) for m in masks])
        self.layers_.to(dev)
        X = torch.from_numpy(Z.reshape(-1, c).astype(np.float32))
        opt = torch.optim.Adam(self.layers_.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), self.batch):
                xb = X[perm[i:i + self.batch]].to(dev)
                opt.zero_grad(set_to_none=True)
                nll = -self._log_prob(xb).mean()
                nll.backward()
                opt.step()
        self.layers_.eval().cpu()
        return self

    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        logdet = torch.zeros(len(x), device=x.device)
        z = x
        for layer in self.layers_:
            z, ld = layer(z)
            logdet = logdet + ld
        base = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(-1)
        return base + logdet

    @torch.no_grad()
    def _patch_scores(self, Z: np.ndarray) -> np.ndarray:
        N, P, c = Z.shape
        dev = self._dev()
        self.layers_.to(dev)
        X = torch.from_numpy(Z.reshape(-1, c).astype(np.float32))
        out = []
        for i in range(0, len(X), 65536):
            out.append(-self._log_prob(X[i:i + 65536].to(dev)).cpu())
        return torch.cat(out).numpy().reshape(N, P)

    def score(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        return _aggregate(self._standardize(self._patch_scores(Z)))


BASELINES = {
    "mahalanobis": MahalanobisDetector,
    "gmm": GmmDetector,
    "patchcore": PatchCoreDetector,
    "flow": FlowDetector,
}


def build_baseline(name: str, **kwargs) -> BaseDetector:
    return BASELINES[name](**kwargs)
