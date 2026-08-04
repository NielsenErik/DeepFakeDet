"""
The three P0 experiments: does the circuit's UNIQUE capability pay for itself?

Everything measured so far says the circuit ties a GMM on detection once both
get the likelihood-ratio treatment.  So its case has to rest on queries no
competitor can answer.  Three of those, each with a competitor baseline that
gets every advantage we can fairly give it:

  1. LOCALIZATION BY PER-PATCH RATIO
     s_p = log p_blend(z_p | z_-p) − log p_real(z_p | z_-p)
     Inherits the ratio's nuisance cancellation AND exact conditioning on
     context.  Compared against PatchCore (0.670 patch-AUC), which currently
     wins with plain surprisal.

  2. OCCLUSION ROBUSTNESS
     Black out k% of patches.  The circuit marginalizes them EXACTLY — the
     query is native.  Baselines must impute (mean-fill) or drop, because a
     per-patch model has no joint to condition on.  Plots AUC vs k.
     This is the deployment argument: real forensic inputs are occluded,
     cropped and partially corrupted.

  3. REGION-WISE DISCRIMINATIVE INFORMATION
     Both circuits share a region graph, so they are COMPATIBLE, and
     information-theoretic quantities between them are tractable (Vergari et
     al., NeurIPS 2021).  For every patch region R we report
        D_R = E_real[ log p_real(z_R) − log p_blend(z_R) ]
     with both marginals computed EXACTLY and the expectation estimated on
     held-out real data.  This is a MODEL-level explanation — which face
     regions carry the discriminative information in general, not for one
     image — and no baseline can produce it at all.

    python scripts/p0_experiments.py --config configs/ffpp_sbi.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdf.cli import load_config  # noqa: E402
from pcdf.device import resolve_device  # noqa: E402
from pcdf.eval.metrics import video_keys, video_level  # noqa: E402
from pcdf.explain.localization import load_masks, localization_metrics  # noqa: E402
from pcdf.models.density_pc import PCConfig, PCDetector  # noqa: E402
from pcdf.models.ratio import PCRatioDetector  # noqa: E402
from pcdf.stages import _feature_dims, _load_split, _tag  # noqa: E402


def _auc(scores, y, vkey):
    from sklearn.metrics import roc_auc_score

    vs, vy = video_level(scores, vkey, y)
    return float(roc_auc_score(vy, vs))


# ── 1. localization by per-patch ratio ──────────────────────────────────────

def exp_localization(cfg, det_ratio, models, n_images=1500):
    grid, c = _feature_dims(cfg)
    Z, idx = _load_split(cfg, "test", label=1)
    rng = np.random.default_rng(cfg["seed"])
    if len(Z) > n_images:
        sel = np.sort(rng.choice(len(Z), n_images, replace=False))
        Z, idx = Z[sel], {k: v[sel] for k, v in idx.items()}
    masks, have = load_masks(cfg, idx, grid)
    Z, masks = Z[have], masks[have]
    idx = {k: v[have] for k, v in idx.items()}
    print(f"[p0.1] {len(Z)} manipulated frames with derived masks")

    out = {}
    s = det_ratio.score(Z.reshape(len(Z), -1))
    out["PC_ratio_conditional"] = localization_metrics(s["_patch_ratio_z"], masks)
    for name, m in models.items():
        ps = m.score(Z).get("_patch")
        if ps is not None:
            out[name] = localization_metrics(ps, masks)
    per_method = {}
    for meth in sorted(set(idx["method"].tolist())):
        sel = idx["method"] == meth
        per_method[meth] = localization_metrics(s["_patch_ratio_z"][sel], masks[sel])
    return {"models": out, "per_method_PC_ratio": per_method}


# ── 2. occlusion robustness ─────────────────────────────────────────────────

def exp_occlusion(cfg, det_ratio, models, fractions=(0.0, 0.125, 0.25, 0.375, 0.5),
                  n_test=6000):
    """
    The circuit marginalizes hidden patches exactly; the baselines cannot, so
    they get the standard practical remedy (mean imputation from real training
    data) — the fairest thing available to them.
    """
    grid, c = _feature_dims(cfg)
    P, d = grid * grid, grid * grid * c
    Ztr, _ = _load_split(cfg, "train", label=0)
    Zte, ite = _load_split(cfg, "test")
    rng = np.random.default_rng(cfg["seed"])
    if len(Zte) > n_test:
        sel = np.sort(rng.choice(len(Zte), n_test, replace=False))
        Zte, ite = Zte[sel], {k: v[sel] for k, v in ite.items()}
    y, vkey = ite["label"].astype(int), video_keys(ite)
    patch_mean = Ztr.mean(0)                       # (P, c) real-train mean

    dev = cfg["device"]
    rows = []
    for frac in fractions:
        n_hide = int(round(frac * P))
        hide = np.zeros((len(Zte), P), bool)
        if n_hide:
            for i in range(len(Zte)):
                hide[i, rng.choice(P, n_hide, replace=False)] = True

        # circuit: EXACT marginalization of the hidden patches
        obs = torch.from_numpy(np.repeat(~hide, c, axis=1)).to(dev)
        X = torch.from_numpy(np.ascontiguousarray(Zte.reshape(len(Zte), -1), np.float32))
        lr, lb = [], []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb, ob = X[i:i + 128].to(dev), obs[i:i + 128]
                lr.append(det_ratio.real.pc.log_prob_masked(xb, ob).cpu().numpy())
                lb.append(det_ratio.blend.pc.log_prob_masked(xb, ob).cpu().numpy())
        pc_auc = _auc(np.concatenate(lb) - np.concatenate(lr), y, vkey)

        # baselines: mean-impute the hidden patches (they have no joint model)
        Zimp = Zte.copy()
        Zimp[hide] = np.broadcast_to(patch_mean, Zte.shape)[hide]
        row = {"hidden_fraction": frac, "PC_ratio_exact_marginal": pc_auc}
        for name, (m_real, m_blend) in models.items():
            s = (m_real._patch_scores(Zimp).mean(1) - m_blend._patch_scores(Zimp).mean(1))
            row[f"{name}_ratio_imputed"] = _auc(s, y, vkey)
        rows.append(row)
        print(f"[p0.2] hidden {frac:5.1%}  " +
              "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "hidden_fraction"),
              flush=True)
    return rows


# ── 3. region-wise discriminative information ───────────────────────────────

def exp_region_information(cfg, det_ratio, n_ref=2000):
    """
    D_R = E_real[ log p_real(z_R) − log p_blend(z_R) ] for every patch region R.

    Both marginals are EXACT (the region is a node of the shared region graph);
    the expectation is a sample average over held-out real frames. High D_R =
    this region is where the two processes disagree most = where the evidence
    lives, as a property of the MODELS rather than of any single image.
    """
    grid, c = _feature_dims(cfg)
    P = grid * grid
    Zval, ival = _load_split(cfg, "val", label=0)
    Z = Zval[:n_ref].reshape(-1, grid * grid * c)
    dev = cfg["device"]
    keep, _ = det_ratio.real._patch_masks(dev)
    X = torch.from_numpy(np.ascontiguousarray(Z, np.float32))
    mr, mb = [], []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = X[i:i + 64].to(dev)
            mr.append(det_ratio.real.pc.region_log_marginals(xb, keep, 2048).cpu().numpy())
            mb.append(det_ratio.blend.pc.region_log_marginals(xb, keep, 2048).cpu().numpy())
    D = (np.concatenate(mr) - np.concatenate(mb)).mean(0)      # (P,)
    order = np.argsort(-D)
    return {
        "per_patch_nats": D.tolist(),
        "grid": grid,
        "top_regions": [{"patch": int(p), "row": int(p // grid), "col": int(p % grid),
                         "nats": float(D[p])} for p in order[:8]],
        "bottom_regions": [{"patch": int(p), "nats": float(D[p])} for p in order[-4:]],
        "note": ("exact marginals per region; expectation estimated on held-out "
                 "real frames. Higher = the two processes disagree more there."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--n-images", type=int, default=1500)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root = Path(cfg["root"])
    tag = _tag(cfg)
    grid, c = _feature_dims(cfg)

    Zseed, _ = _load_split(cfg, "val", label=0)
    det_ratio = PCRatioDetector(grid, grid, c,
                                PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"]))
    det_ratio.load(root / "models" / f"ratio_{tag}.pt", Zseed[:8].reshape(8, -1),
                   structure_cache=root / "models" / f"structure_{tag}.pkl")
    det_ratio.calibrate(Zseed[:1200].reshape(-1, grid * grid * c))

    results = {"tag": tag, "config": args.config}

    if "1" not in args.skip:
        import pickle
        with open(root / "models" / f"baselines_{tag}.pkl", "rb") as fh:
            base = pickle.load(fh)
        results["p0_1_localization"] = exp_localization(cfg, det_ratio, base, args.n_images)
        print(json.dumps(results["p0_1_localization"]["models"], indent=2, default=float))

    if "2" not in args.skip:
        from pcdf.models.baselines import GmmDetector, MahalanobisDetector

        Ztr, _ = _load_split(cfg, "train", label=0)
        Zbl, _ = _load_split(cfg, "train", label=0, perturb="blend")
        pairs = {}
        for name, make in [("gmm", lambda: GmmDetector(n_components=8)),
                           ("mahalanobis", lambda: MahalanobisDetector())]:
            pairs[name] = (make().fit(Ztr), make().fit(Zbl))
        results["p0_2_occlusion"] = exp_occlusion(cfg, det_ratio, pairs)

    if "3" not in args.skip:
        results["p0_3_region_information"] = exp_region_information(cfg, det_ratio)
        print("[p0.3] most discriminative regions (nats):")
        for r in results["p0_3_region_information"]["top_regions"]:
            print(f"    patch {r['patch']:3d} (row {r['row']}, col {r['col']}): {r['nats']:+.2f}")

    out = root / "results" / tag / "p0_experiments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[p0] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
