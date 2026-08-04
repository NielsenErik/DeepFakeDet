"""
Exact localization — the query class that justifies the circuit.

For a flagged face, the circuit answers, exactly and in one batched pass:

    marginal      −log p(z_p)            how unusual is this patch on its own
    conditional   −log p(z_p | z_{−p})   how unusual is it GIVEN the rest of
                                          the face — i.e. is it inconsistent
                                          with its own context
    subset        −log p(z_S) for a set S chosen greedily to maximize
                  surprisal per patch (the Khosravi-style outlier-explanation
                  query): "the manipulated region is S"

No competing one-class detector produces these exactly.  A flow gives an exact
joint density but no marginals; PatchCore gives per-patch distances but no
probabilities and no conditioning; a VAE gives a bound.  So the evaluation here
is not "does the heatmap look nice" but a quantitative one against the derived
ground-truth masks:

    patch AUC      per-patch scores vs. the binarized mask, pooled over images
    PBCA           patch-level balanced accuracy at the image-adaptive threshold
    IoU@best       best achievable IoU over thresholds (an upper bound on what
                   a downstream user could extract)
    pointing hit   does the argmax patch fall inside the manipulated region

`mask_source` is recorded in every output: for FF++ these masks are DERIVED
from the frame-aligned real counterpart (this distribution ships no official
mask videos), and for self-blends they are the exact blending masks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_masks(cfg: Dict, index: Dict[str, np.ndarray], grid: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the derived masks for the indexed frames and pool them onto the patch
    grid.  Returns (masks (N, P) in [0,1], has_mask (N,) bool).
    """
    import cv2

    root = Path(cfg["root"])
    N = len(index["path"])
    out = np.zeros((N, grid * grid), np.float32)
    have = np.zeros(N, bool)
    for i, p in enumerate(index["path"]):
        mp = Path(p).parent / f"mask_{Path(p).stem}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.resize(m.astype(np.float32) / 255.0, (grid, grid),
                       interpolation=cv2.INTER_AREA)
        out[i] = m.reshape(-1)
        have[i] = True
    return out, have


def greedy_anomalous_subset(patch_z: np.ndarray, max_size: int = 8
                            ) -> List[int]:
    """
    Greedy selection of the most anomalous patch set for one image: repeatedly
    add the patch with the highest standardized conditional surprisal while the
    mean surprisal of the set keeps rising.  This is the practical form of the
    outlier-explanation query — "which minimal region explains the flag".
    """
    order = np.argsort(-patch_z)
    chosen: List[int] = []
    best_mean = -np.inf
    for p in order[:max_size]:
        cand = chosen + [int(p)]
        m = float(patch_z[cand].mean())
        if m < best_mean:
            break
        chosen, best_mean = cand, m
    return chosen


def localization_metrics(patch_scores: np.ndarray, masks: np.ndarray,
                         mask_threshold: float = 0.25) -> Dict[str, float]:
    """Pooled patch AUC + per-image IoU / pointing metrics."""
    from sklearn.metrics import roc_auc_score

    y = (masks > mask_threshold).astype(int)
    keep = (y.sum(1) > 0) & (y.sum(1) < y.shape[1])      # informative images only
    if keep.sum() == 0:
        return {"n_images": 0}
    ys, ss = y[keep], patch_scores[keep]

    pooled_auc = float(roc_auc_score(ys.reshape(-1), ss.reshape(-1)))
    per_image = [float(roc_auc_score(a, b)) for a, b in zip(ys, ss)
                 if 0 < a.sum() < len(a)]

    # best-threshold IoU, per image, over a shared quantile sweep
    ious = []
    for a, b in zip(ys, ss):
        best = 0.0
        for q in np.linspace(0.5, 0.95, 10):
            pred = b >= np.quantile(b, q)
            inter = float((pred & (a > 0)).sum())
            union = float((pred | (a > 0)).sum())
            best = max(best, inter / max(union, 1e-9))
        ious.append(best)

    hits = [float(a[int(np.argmax(b))] > 0) for a, b in zip(ys, ss)]
    return {
        "n_images": int(keep.sum()),
        "patch_auc_pooled": pooled_auc,
        "patch_auc_per_image": float(np.mean(per_image)) if per_image else float("nan"),
        "iou_best_mean": float(np.mean(ious)),
        "pointing_accuracy": float(np.mean(hits)),
        "manipulated_patch_fraction": float(ys.mean()),
    }


def run_localization(cfg: Dict, tag: str, n_images: int = 1000,
                     make_figures: bool = True, dataset: str = "ffpp") -> Dict:
    """
    Score fake test frames, compare the exact per-patch maps to the derived
    masks, and (optionally) write overlay figures.

    Baselines get the identical treatment on their own per-patch maps, because
    "the circuit localizes" only means something relative to what a memory bank
    or a Gaussian already localizes.
    """
    import pickle

    from ..models.density_pc import PCConfig, PCDetector
    from ..stages import _load_split

    root = Path(cfg["root"])
    from ..stages import _feature_dims

    grid, c = _feature_dims(cfg)
    Z, idx = _load_split(cfg, "test", dataset=dataset, label=1)
    if len(Z) > n_images:
        sel = np.random.default_rng(cfg["seed"]).choice(len(Z), n_images, replace=False)
        sel = np.sort(sel)
        Z, idx = Z[sel], {k: v[sel] for k, v in idx.items()}

    masks, have = load_masks(cfg, idx, grid)
    Z, idx = Z[have], {k: v[have] for k, v in idx.items()}
    masks = masks[have]
    print(f"[explain] {len(Z)} fake frames with derived masks")

    out: Dict = {"tag": tag, "dataset": dataset, "mask_source": "derived_frame_diff",
                 "n_images": int(len(Z)), "models": {}}

    pcc = PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"])
    det = PCDetector(grid, grid, c, pcc)
    det.load(root / "models" / f"pc_{tag}.pt", Z[:8].reshape(8, -1),
             structure_cache=root / "models" / f"structure_{tag}.pkl")
    s = det.score(Z.reshape(len(Z), -1))
    zc, zm = s["_patch_cond_z"], s["_patch_marg_z"]
    out["models"]["PC_conditional"] = localization_metrics(zc, masks)
    out["models"]["PC_marginal"] = localization_metrics(zm, masks)

    # per-method breakdown: NeuralTextures edits a small region, Deepfakes a
    # whole face — localization difficulty is not uniform and reporting one
    # number would hide that
    per_method = {}
    for meth in sorted(set(idx["method"].tolist())):
        m = idx["method"] == meth
        per_method[meth] = localization_metrics(zc[m], masks[m])
    out["per_method_conditional"] = per_method

    b_path = root / "models" / f"baselines_{tag}.pkl"
    if b_path.exists():
        with open(b_path, "rb") as fh:
            models = pickle.load(fh)
        for name, m in models.items():
            ps = m.score(Z).get("_patch")
            if ps is not None:
                out["models"][name] = localization_metrics(ps, masks)

    # the explanation query itself, on a handful of images
    subsets = [greedy_anomalous_subset(zc[i]) for i in range(min(32, len(zc)))]
    out["example_subsets"] = [
        {"video": str(idx["video"][i]), "frame": int(idx["frame"][i]),
         "method": str(idx["method"][i]), "patches": subsets[i],
         "mask_overlap": float(masks[i][subsets[i]].mean()) if subsets[i] else 0.0}
        for i in range(len(subsets))]
    out["summary"] = {k: v.get("patch_auc_pooled") for k, v in out["models"].items()}
    out["summary"]["subset_mask_overlap"] = float(
        np.mean([e["mask_overlap"] for e in out["example_subsets"]]))

    res_dir = root / "results" / tag
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "localization.json").write_text(json.dumps(out, indent=2, default=float))

    if make_figures:
        try:
            _figures(cfg, idx, zc, masks, res_dir / "localization_examples.png", grid)
        except Exception as exc:                     # noqa: BLE001
            print(f"[explain] figure generation skipped: {exc}")
    return out


def _figures(cfg: Dict, idx, patch_z: np.ndarray, masks: np.ndarray,
             out_path: Path, grid: int, n: int = 8) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-patch_z.max(1))[:n]
    fig, axes = plt.subplots(3, len(order), figsize=(2.1 * len(order), 6.4))
    for j, i in enumerate(order):
        img = cv2.cvtColor(cv2.imread(str(idx["path"][i])), cv2.COLOR_BGR2RGB)
        hm = cv2.resize(patch_z[i].reshape(grid, grid), img.shape[:2][::-1],
                        interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(masks[i].reshape(grid, grid), img.shape[:2][::-1],
                        interpolation=cv2.INTER_NEAREST)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{idx['method'][i]}", fontsize=7)
        axes[1, j].imshow(img)
        axes[1, j].imshow(hm, alpha=0.55, cmap="inferno")
        axes[2, j].imshow(gt, cmap="gray")
        for r in range(3):
            axes[r, j].axis("off")
    axes[1, 0].set_ylabel("−log p(z_p | z_-p), z-scored")
    fig.suptitle("Exact per-patch conditional surprisal vs derived manipulation mask",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
