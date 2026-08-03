"""
POC extension: EXPLAINABILITY by exact marginals.

The detection POC (poc_deepfake_pc.py) uses global forensic features, so its
NLL says "this image is fake" but not WHERE. Here the feature vector keeps
spatial identity — a 4x4 grid of patches, 3 forensic statistics per patch —
so the PC's exact marginals answer the localization question directly:

    per-patch anomaly  =  −log p(z_S)    S = the 3 features of patch i,

computed EXACTLY in one circuit pass per patch (smoothness + decomposability;
this is the Khosravi et al. "Why is this an outlier?" query). No gradients,
no sampling, no surrogate model: the heatmap is a set of exact probabilistic
statements under the real-face density.

Localization ground truth exists because we generate the fakes: the self-blend
mask says which pixels were actually replaced. We report patch-level AUROC
(anomalous-patch vs untouched-patch, pooled over fake images) and save
heatmap overlays for visual inspection.

NOTE: features are z-scored per dimension but NOT whitened/PCA'd — any linear
mixing would destroy the patch identity of the coordinates. The Chow-Liu vtree
handles the (strong) cross-patch correlations instead.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bak"))

from allinone_probabilistic_circuits import DensityPC, chow_liu_vtree  # noqa: E402
from poc_deepfake_pc import (  # noqa: E402
    JITTER, SEED, SymBrokenGMLeaf, auroc, load_lfw, self_blend)

GRID = 4                    # 4x4 patches on a 128x128 face
DIMS_PER_PATCH = 4
N_DIM = GRID * GRID * DIMS_PER_PATCH
EPOCHS = int(os.environ.get("POC_EPOCHS", "300"))
LR = 5e-3
OUT_DIR = os.path.join(os.path.dirname(HERE), "results_explain")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# Patch-identified forensic features
# ═══════════════════════════════════════════════════════════════════════════

def patch_features(images: np.ndarray) -> np.ndarray:
    """
    (N, 128, 128, 3) in [0,1] -> (N, 64): for each of the 4x4 patches
      [ high-frequency log-DCT energy,        (resampling fingerprint)
        noise-residual std,                   (sensor-noise disturbance)
        color shift vs whole image,           (blending color mismatch)
        mean cross-channel residual corr ].   (demosaicing/noise coherence)
    Feature j belongs to patch j // DIMS_PER_PATCH — the mapping the
    marginals rely on.
    """
    import cv2

    P = 128 // GRID
    fy, fx = np.mgrid[0:P, 0:P]
    high = np.sqrt(fy ** 2 + fx ** 2) > P // 2

    out = np.empty((len(images), N_DIM), np.float32)
    for n, img in enumerate(images):
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor((img * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        res = img - cv2.GaussianBlur(img, (5, 5), 0)
        gmean = img.reshape(-1, 3).mean(0)
        k = 0
        for gy in range(GRID):
            for gx in range(GRID):
                sl = np.s_[gy * P:(gy + 1) * P, gx * P:(gx + 1) * P]
                d = cv2.dct(gray[sl])
                out[n, k] = np.log(np.abs(d[high]) + 1e-8).mean()
                out[n, k + 1] = res[sl].std()
                out[n, k + 2] = np.abs(img[sl].reshape(-1, 3).mean(0)
                                       - gmean).sum()
                r = res[sl].reshape(-1, 3)
                cc = np.corrcoef(r, rowvar=False)
                out[n, k + 3] = (cc[0, 1] + cc[0, 2] + cc[1, 2]) / 3.0
                k += DIMS_PER_PATCH
    return out


def patch_scope(i: int) -> list:
    """Feature indices belonging to patch i."""
    return list(range(i * DIMS_PER_PATCH, (i + 1) * DIMS_PER_PATCH))


# ═══════════════════════════════════════════════════════════════════════════
# Exact per-patch marginal anomaly scores
# ═══════════════════════════════════════════════════════════════════════════

def patch_marginal_nll(pc: DensityPC, Z: torch.Tensor) -> np.ndarray:
    """(B, GRID*GRID): −log p(z_patch_i), exact, one circuit pass per patch."""
    scores = []
    with torch.no_grad():
        for i in range(GRID * GRID):
            keep = set(patch_scope(i))
            marg = [j for j in range(N_DIM) if j not in keep]
            scores.append(-pc.log_marginal(Z, marg))
    return torch.stack(scores, dim=1).numpy()


def patch_conditional_nll(pc: DensityPC, Z: torch.Tensor) -> np.ndarray:
    """
    (B, GRID*GRID): −log p(z_patch_i | z_rest) = −[log p(z) − log p(z_−i)],
    exact via two circuit passes. The right query for blending: a swapped
    patch can be individually plausible yet INCONSISTENT with the rest of
    the face, and the conditional is exactly that inconsistency.
    """
    scores = []
    with torch.no_grad():
        full = pc.log_prob(Z)
        for i in range(GRID * GRID):
            rest = pc.log_marginal(Z, patch_scope(i))   # z_i integrated out
            scores.append(-(full - rest))
    return torch.stack(scores, dim=1).numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import cv2

    os.makedirs(OUT_DIR, exist_ok=True)

    X, ident = load_lfw()
    ids = np.array(sorted(set(ident)))
    rng.shuffle(ids)
    cut = int(0.7 * len(ids))
    train_mask = np.isin(ident, ids[:cut])
    Xtr, Xte = X[train_mask], X[~train_mask]
    ident_te = ident[~train_mask]
    print(f"[data] train real {len(Xtr)}, test real {len(Xte)}")

    # fakes WITH ground-truth blend masks (test identities only)
    fakes, masks = [], []
    idx = np.arange(len(Xte))
    for i in idx:
        j = rng.choice(idx[ident_te != ident_te[i]])
        f, m = self_blend(Xte[i], Xte[j], rng, return_mask=True)
        fakes.append(f)
        masks.append(cv2.resize(m, (128, 128)))
    F = np.stack(fakes)
    M = np.stack(masks)

    t0 = time.time()
    Ztr_np = patch_features(Xtr)
    Zte_np = patch_features(Xte)
    Zfk_np = patch_features(F)
    print(f"[feat] patch features {N_DIM}d done ({time.time() - t0:.0f}s)")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Ztr_np)
    Ztr = torch.from_numpy(scaler.transform(Ztr_np).astype(np.float32))
    Zte = torch.from_numpy(scaler.transform(Zte_np).astype(np.float32))
    Zfk = torch.from_numpy(scaler.transform(Zfk_np).astype(np.float32))

    # PC on real train only (balance floor keeps the learned vtree shallow
    # enough for the K^depth subtree duplication to stay affordable)
    vtree = chow_liu_vtree(Ztr.numpy(), min_balance=0.45)
    pc = DensityPC(vtree, n_sum_components=2,
                   leaf_factory=lambda i: SymBrokenGMLeaf(i, 4))
    pc.fit_leaves(Ztr, jitter=JITTER)
    print(f"[pc] {sum(p.numel() for p in pc.parameters())} params")
    opt = torch.optim.Adam(pc.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        nll = -pc.log_prob(Ztr).mean()
        nll.backward()
        opt.step()
        if ep % 50 == 0 or ep == EPOCHS - 1:
            print(f"[pc] epoch {ep:4d}  NLL {nll.item():8.3f} "
                  f"({time.time() - t0:.0f}s)")
    pc.validate()
    assert abs(float(pc.log_partition().detach())) < 1e-4
    print("[audit] properties ✓  log Z = 0 ✓")

    # ── image-level detection ─────────────────────────────────────────────
    with torch.no_grad():
        s_te = -pc.log_prob(Zte).numpy()
        s_fk = -pc.log_prob(Zfk).numpy()
    print(f"\n[image-level] AUROC full log p(z): {auroc(s_te, s_fk):.3f}")

    # ── patch-level localization: exact marginals AND exact conditionals ──
    # ground truth: patch is 'manipulated' if >30% of its pixels were blended
    P128 = 128 // GRID
    gt = np.stack([
        np.array([M[b, gy * P128:(gy + 1) * P128, gx * P128:(gx + 1) * P128].mean()
                  for gy in range(GRID) for gx in range(GRID)])
        for b in range(len(M))
    ]) > 0.3

    H_fk = None
    for name, fn in [("marginal", patch_marginal_nll),
                     ("conditional", patch_conditional_nll)]:
        t0 = time.time()
        P_te = fn(pc, Zte)                      # (B_real, 16)
        P_fk = fn(pc, Zfk)                      # (B_fake, 16)
        # calibrate each patch position against the real-test distribution
        mu, sd = P_te.mean(0), P_te.std(0) + 1e-8
        h_fk = (P_fk - mu) / sd
        h_te = (P_te - mu) / sd
        loc_auc = auroc(h_fk[~gt], h_fk[gt])
        img_auc = auroc(h_te.max(1), h_fk.max(1))
        print(f"[explain:{name}] localization AUROC {loc_auc:.3f}  "
              f"image AUROC (max patch) {img_auc:.3f}  "
              f"({time.time() - t0:.0f}s)")
        if name == "conditional":
            H_fk = h_fk                         # conditionals drive overlays

    # ── visual overlays ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-H_fk.max(1))[:6]        # most confidently flagged
    fig, axes = plt.subplots(3, len(order), figsize=(3 * len(order), 9))
    for c, b in enumerate(order):
        img128 = cv2.resize(F[b], (128, 128))
        heat = H_fk[b].reshape(GRID, GRID)
        heat_up = cv2.resize(heat, (128, 128), interpolation=cv2.INTER_NEAREST)
        axes[0, c].imshow(np.clip(img128, 0, 1))
        axes[0, c].set_title(f"fake #{b}", fontsize=9)
        axes[1, c].imshow(np.clip(img128, 0, 1))
        im = axes[1, c].imshow(heat_up, cmap="inferno", alpha=0.55,
                               vmin=0, vmax=max(heat.max(), 3))
        axes[1, c].set_title("−log p(z_patch | z_rest), z-scored", fontsize=9)
        axes[2, c].imshow(M[b], cmap="gray")
        axes[2, c].set_title("true blend mask", fontsize=9)
        for ax in axes[:, c]:
            ax.axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8)
    out_png = os.path.join(OUT_DIR, "patch_marginal_heatmaps.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"[explain] overlays saved to {out_png}")


if __name__ == "__main__":
    main()
