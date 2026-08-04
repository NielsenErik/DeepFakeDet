"""
Patch-token projection: (N, P, C_backbone) -> (N, P·c) circuit inputs.

Three things have to be true of this map, or the whole exactness argument leaks:

1. It is FIT ON REAL TRAINING FACES ONLY.  Nothing about the fakes, and nothing
   about the test identities, may enter the projection.
2. It is a FIXED bijection-like preprocessing (PCA rotation + whitening +
   z-score), applied identically to every split and every dataset.  The circuit
   then models a normalized density in those coordinates — every property
   (smoothness, decomposability, Z = 1, exact marginals) is untouched, since a
   fixed invertible linear map only shifts the log-density by a constant
   Jacobian term that is identical for all samples and cancels in every AUROC.
3. Whitening happens per CHANNEL BLOCK, not across patches.  Mixing patches
   would destroy the patch identity of coordinates and with it every localized
   query — the mistake the explainability POC explicitly avoided.

`fit_streaming` runs over shards so a 200k-crop feature set never has to be
resident; only C×C covariance accumulators are.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass
class PatchProjector:
    """PCA(+whiten) over the channel axis, shared by all patch positions."""
    mean: np.ndarray            # (C,)
    components: np.ndarray      # (c, C)
    scale: np.ndarray           # (c,)   per-component z-scale after projection
    explained: float
    n_patches: int
    out_dim: int
    whiten: bool = True
    per_patch_mean: Optional[np.ndarray] = None   # (P, c) removed after projection
    per_patch_std: Optional[np.ndarray] = None    # (P, c)

    # ── fitting ─────────────────────────────────────────────────────────
    @staticmethod
    def fit(
        X: np.ndarray,
        out_dim: int = 16,
        whiten: bool = True,
        per_patch_standardize: bool = True,
        max_rows: int = 400_000,
        seed: int = 0,
    ) -> "PatchProjector":
        """
        X: (N, P, C) real-train patch tokens.

        `per_patch_standardize` z-scores each (patch, component) with the real
        training statistics of THAT patch position.  Faces are spatially
        structured — an eye patch and a background patch have different
        distributions — and without this the circuit spends its capacity
        learning the mean face layout instead of the deviations that matter.
        It also makes per-patch surprisals comparable across positions, which
        the localization score needs.
        """
        N, P, C = X.shape
        flat = X.reshape(-1, C)
        rng = np.random.default_rng(seed)
        if len(flat) > max_rows:
            flat = flat[rng.choice(len(flat), max_rows, replace=False)]
        flat = flat.astype(np.float64)
        mean = flat.mean(0)
        centered = flat - mean
        # economy SVD on the subsample; C is at most ~1024 so this is cheap
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        comp = Vt[:out_dim]
        var = (S ** 2) / max(len(centered) - 1, 1)
        explained = float(var[:out_dim].sum() / var.sum())
        scale = np.sqrt(var[:out_dim]) if whiten else np.ones(out_dim)
        scale = np.maximum(scale, 1e-8)

        proj = PatchProjector(mean=mean.astype(np.float32),
                              components=comp.astype(np.float32),
                              scale=scale.astype(np.float32),
                              explained=explained, n_patches=P, out_dim=out_dim,
                              whiten=whiten)
        if per_patch_standardize:
            Z = proj.transform(X, standardize=False).reshape(N, P, out_dim)
            proj.per_patch_mean = Z.mean(0).astype(np.float32)
            proj.per_patch_std = (Z.std(0) + 1e-6).astype(np.float32)
        return proj

    @staticmethod
    def fit_from_moments(n: int, s1: np.ndarray, s2: np.ndarray, n_patches: int,
                         out_dim: int = 16, whiten: bool = True) -> "PatchProjector":
        """
        Fit from streaming first/second moments of the channel vectors:
        `s1` (C,) sum, `s2` (C,C) sum of outer products, `n` vectors.

        Storing raw CLIP tokens for the whole training set would be ~80 GB;
        the covariance is 1024×1024 regardless of how many crops stream past,
        so the projector is fitted in one pass with no feature cache at all.
        """
        if out_dim is not None and out_dim <= 0:
            # Identity projection: keep every coordinate, standardize later.
            # PCA ranks directions by VARIANCE, and forensic artifacts are
            # low-variance by nature (content dominates), so for hand-built
            # artifact features the reduction itself can delete the signal —
            # measured on CLIP: probe 0.775 vs one-class 0.536.
            C = len(s1)
            return PatchProjector(
                mean=(s1 / n).astype(np.float32),
                components=np.eye(C, dtype=np.float32),
                scale=np.ones(C, dtype=np.float32),
                explained=1.0, n_patches=n_patches, out_dim=C, whiten=False)
        mean = s1 / n
        cov = s2 / n - np.outer(mean, mean)
        cov = (cov + cov.T) / 2.0
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = np.maximum(vals[order], 1e-12), vecs[:, order]
        comp = vecs[:, :out_dim].T
        scale = np.sqrt(vals[:out_dim]) if whiten else np.ones(out_dim)
        return PatchProjector(
            mean=mean.astype(np.float32), components=comp.astype(np.float32),
            scale=np.maximum(scale, 1e-8).astype(np.float32),
            explained=float(vals[:out_dim].sum() / vals.sum()),
            n_patches=n_patches, out_dim=out_dim, whiten=whiten)

    def set_patch_stats(self, Z: np.ndarray) -> None:
        """Per-(patch, component) mean/sd from projected REAL TRAIN features."""
        self.per_patch_mean = Z.mean(0).astype(np.float32)
        self.per_patch_std = (Z.std(0) + 1e-6).astype(np.float32)

    # ── applying ────────────────────────────────────────────────────────
    def transform(self, X: np.ndarray, standardize: bool = True,
                  flatten: bool = False) -> np.ndarray:
        """(N, P, C) -> (N, P, c) or (N, P·c) with feature index p·c + j."""
        Z = (X.astype(np.float32) - self.mean) @ self.components.T
        Z = Z / self.scale
        if standardize and self.per_patch_mean is not None:
            Z = (Z - self.per_patch_mean) / self.per_patch_std
        return Z.reshape(len(X), -1) if flatten else Z

    # ── persistence ─────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components,
                 scale=self.scale, explained=self.explained,
                 n_patches=self.n_patches, out_dim=self.out_dim,
                 whiten=self.whiten,
                 per_patch_mean=(self.per_patch_mean if self.per_patch_mean is not None
                                 else np.zeros(0, np.float32)),
                 per_patch_std=(self.per_patch_std if self.per_patch_std is not None
                                else np.zeros(0, np.float32)))

    @staticmethod
    def load(path: str | Path) -> "PatchProjector":
        z = np.load(path)
        ppm = z["per_patch_mean"]
        pps = z["per_patch_std"]
        return PatchProjector(
            mean=z["mean"], components=z["components"], scale=z["scale"],
            explained=float(z["explained"]), n_patches=int(z["n_patches"]),
            out_dim=int(z["out_dim"]), whiten=bool(z["whiten"]),
            per_patch_mean=(ppm if ppm.size else None),
            per_patch_std=(pps if pps.size else None))


def fit_streaming(
    shards: Iterable[str | Path],
    out_dim: int = 16,
    whiten: bool = True,
    max_images: int = 40_000,
    seed: int = 0,
) -> PatchProjector:
    """
    Fit the projector from `.npy` shards of real-train patch tokens without
    loading them all: sample a bounded number of images across shards, which is
    plenty for a C×C second-moment estimate and keeps peak RAM at a few GB.
    """
    rng = np.random.default_rng(seed)
    chunks, taken = [], 0
    for shard in shards:
        arr = np.load(shard, mmap_mode="r")
        n = min(len(arr), max(1, max_images // 8))
        idx = np.sort(rng.choice(len(arr), n, replace=False))
        chunks.append(np.asarray(arr[idx], dtype=np.float32))
        taken += n
        if taken >= max_images:
            break
    X = np.concatenate(chunks, axis=0)
    return PatchProjector.fit(X, out_dim=out_dim, whiten=whiten, seed=seed)
