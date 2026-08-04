"""
Why does a supervised probe separate these features when a density model cannot?

There are three candidate explanations, and they demand different fixes, so
this module measures which one is true instead of arguing about it.

H1  DILUTION.  The shift is real but concentrated in a few of the d coordinates.
    A probe learns weights and can isolate them; a density model sums surprisal
    over ALL coordinates with equal footing, so 6000 nuisance dimensions bury a
    signal carried by 50.  Test: select the top-k discriminative coordinates on
    the VALIDATION split, then rescore the SAME fitted density restricted to
    that subset — exactly the marginal query a smooth+decomposable circuit
    supports natively.  If AUC jumps, H1 is confirmed and the fix is a
    likelihood RATIO or a selected scope, not a better circuit.

H2  DIRECTION.  The shift exists in the full space but points "inward": fakes
    are more typical.  Test: signed statistics — what fraction of manipulated
    frames have HIGHER joint likelihood than the real median, and does the
    two-sided score beat the one-sided one.

H3  ESTIMATION.  The density is simply badly fitted.  Test: compare the
    circuit's held-out NLL against a full-covariance Gaussian and a GMM in the
    same space; if the circuit is not clearly better, the model is the problem.

The dilution test is run through the circuit's own exact marginals, so its
result is a statement about the model that was actually fitted, not about a
surrogate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def discriminative_coordinates(Zval: np.ndarray, yval: np.ndarray,
                               top_k: int = 64) -> np.ndarray:
    """
    Rank flattened coordinates by how well each separates the classes on the
    VALIDATION split (absolute standardized mean difference — a per-coordinate
    t-statistic).  Selection never touches test data.
    """
    X = Zval.reshape(len(Zval), -1)
    a, b = X[yval == 0], X[yval == 1]
    diff = np.abs(a.mean(0) - b.mean(0))
    pooled = np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-8
    t = diff / pooled
    return np.argsort(-t)[:top_k]


def run_diagnosis(cfg: Dict, tag: str, top_ks: Tuple[int, ...] = (16, 64, 256, 1024),
                  n_eval: int = 8000) -> Dict:
    from sklearn.metrics import roc_auc_score

    from ..models.density_pc import PCConfig, PCDetector
    from ..stages import _feature_dims, _load_split
    from .metrics import video_keys, video_level

    root = Path(cfg["root"])
    grid, c = _feature_dims(cfg)
    Zval, ival = _load_split(cfg, "val")
    Zte, ite = _load_split(cfg, "test")
    rng = np.random.default_rng(cfg["seed"])
    if len(Zte) > n_eval:
        sel = np.sort(rng.choice(len(Zte), n_eval, replace=False))
        Zte, ite = Zte[sel], {k: v[sel] for k, v in ite.items()}
    yte = ite["label"].astype(int)
    vkey = video_keys(ite)

    pcc = PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"])
    det = PCDetector(grid, grid, c, pcc)
    det.load(root / "models" / f"pc_{tag}.pt", Zte[:8].reshape(8, -1),
             structure_cache=root / "models" / f"structure_{tag}.pkl")

    import torch

    d = grid * grid * c
    Xte = torch.from_numpy(np.ascontiguousarray(Zte.reshape(len(Zte), -1), np.float32))
    out: Dict = {"tag": tag, "d": d, "n_eval": int(len(Zte))}

    def _auc(s: np.ndarray) -> Dict[str, float]:
        vs, vy = video_level(s, vkey, yte)
        return {"auc_video": float(roc_auc_score(vy, vs)),
                "auc_frame": float(roc_auc_score(yte, s))}

    # ── H1: exact marginals over the most discriminative scope ───────────
    dev = cfg["device"]
    h1: Dict[str, Dict] = {}
    with torch.no_grad():
        full = det.pc.log_prob(Xte.to(dev), chunk=pcc.batch_size).cpu().numpy()
    h1["full_joint"] = _auc(-full)

    for k in top_ks:
        if k >= d:
            continue
        keep = discriminative_coordinates(Zval, ival["label"].astype(int), k)
        obs = torch.zeros((1, d), dtype=torch.bool, device=dev)
        obs[0, torch.as_tensor(np.sort(keep), device=dev)] = True
        with torch.no_grad():
            lp = det.pc.log_prob_masked(Xte.to(dev), obs, chunk=pcc.batch_size)
        s = -lp.cpu().numpy()
        # both directions: the selected scope may be atypical OR over-typical
        r = _auc(s)
        r_low = _auc(-s)
        h1[f"marginal_top{k}"] = {**r, "auc_video_inverted": r_low["auc_video"],
                                  "best": max(r["auc_video"], r_low["auc_video"])}
    out["H1_dilution"] = h1

    # ── H2: which direction is the shift ─────────────────────────────────
    med = float(np.median(full[yte == 0]))
    frac_higher = float((full[yte == 1] > med).mean())
    out["H2_direction"] = {
        "median_loglik_real": med,
        "median_loglik_fake": float(np.median(full[yte == 1])),
        "fraction_of_fakes_ABOVE_real_median": frac_higher,
        "reading": ("fakes are MORE likely than reals — the low-density "
                    "assumption is inverted" if frac_higher > 0.55 else
                    "fakes are less likely than reals — the assumption holds"),
        "one_sided_auc": _auc(-full)["auc_video"],
        "two_sided_auc": _auc(np.abs(full - med))["auc_video"],
    }

    # ── H3: is the density any good? ─────────────────────────────────────
    from sklearn.covariance import LedoitWolf

    Ztr, _ = _load_split(cfg, "train", label=0)
    sub = Ztr[np.sort(rng.choice(len(Ztr), min(20000, len(Ztr)), replace=False))]
    Xtr_flat = sub.reshape(len(sub), -1)
    Zv_real = Zval[ival["label"] == 0][:4000].reshape(-1, d)
    lw = LedoitWolf().fit(Xtr_flat)
    gauss_nll = float(-lw.score(Zv_real))
    with torch.no_grad():
        pc_nll = float(-det.pc.log_prob(
            torch.from_numpy(np.ascontiguousarray(Zv_real, np.float32)).to(dev),
            chunk=pcc.batch_size).mean())
    out["H3_estimation"] = {
        "pc_val_nll": pc_nll,
        "full_covariance_gaussian_val_nll": gauss_nll,
        "pc_better_by_nats": gauss_nll - pc_nll,
        "reading": ("the circuit fits better than a full-covariance Gaussian"
                    if pc_nll < gauss_nll else
                    "the circuit does NOT beat a Gaussian — estimation is the problem"),
    }

    best_marg = max((v.get("best", 0.0) for k, v in h1.items()
                     if k.startswith("marginal")), default=0.0)
    out["verdict"] = _verdict(h1["full_joint"]["auc_video"], best_marg,
                              out["H2_direction"]["fraction_of_fakes_ABOVE_real_median"])
    return out


def _verdict(full_auc: float, best_marginal_auc: float, frac_higher: float) -> str:
    gain = best_marginal_auc - full_auc
    if gain > 0.05:
        return (f"H1 CONFIRMED: restricting the exact marginal to the most "
                f"discriminative coordinates gains {gain:+.3f} AUC over the full "
                f"joint. The signal is real but diluted across dimensions — the "
                f"fix is a likelihood RATIO or a selected scope, not a better "
                f"density model.")
    if frac_higher > 0.55:
        return ("H2 CONFIRMED: fakes are more likely than reals under the fitted "
                "density, so one-sided NLL cannot work by construction.")
    return ("Neither dilution nor inversion explains the gap; suspect the "
            "estimator or the features.")


def cmd_diagnose(cfg: Dict, args) -> None:
    from ..stages import _tag

    res = run_diagnosis(cfg, _tag(cfg), n_eval=args.n_eval)
    out = Path(cfg["root"]) / "results" / _tag(cfg) / "diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
