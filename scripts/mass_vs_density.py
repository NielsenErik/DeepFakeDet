"""
Score by probability MASS instead of density.

THE ARGUMENT.  A density is mass per unit volume, and the two come apart
precisely where likelihood-based detection fails.  Data confined to a thin,
low-dimensional sheet gets a large density and almost no mass.  Kamkari et al.
(ICML 2024, arXiv:2403.18910) show this is the mechanism behind the likelihood
OOD paradox — models assign OOD inputs high likelihood yet never generate them,
because the mass around them is ~0 — and win by thresholding likelihood
together with an estimate of Local Intrinsic Dimension, LID standing in for
volume.  They need the proxy because a flow or a diffusion model cannot
integrate its own density.

A smooth, decomposable circuit with tractable-CDF leaves can, exactly, in one
pass: `EinsumPC.log_ball`.  So this script asks the question the proxy was
invented to approximate, directly.

It also answers Le Lan & Dinh (Entropy 2021, arXiv:2012.03808): a density is not
reparametrization-invariant, so its ORDERING carries less information than
anomaly detection assumes.  Probability mass over a region is invariant.

WHAT IS MEASURED, and what each row can rule out:

  density        -log p(x).  The existing one-class score, reproduced here so
                 every number below is on identical crops and identical code.
  mass(eps)      -log P(|u - x| <= eps), swept.  If this equals `density` at
                 every eps, the leaves are too smooth to resolve the sheet and
                 the whole idea is dead — that is the real risk and the sweep
                 is what exposes it.
  lid            the slope of log P(box) against log(2*eps): for small eps
                 log M ~ log p + d_local * log(2*eps), so the slope IS the local
                 dimension, computed rather than estimated.  Kamkari's
                 prediction is that forgeries have LOWER local dimension than
                 real faces.
  mass - density the two-sided combination.  Their dual threshold, exactly.

INTERPRETING IT.  Compare `auc_video` across rows against the circuit's own
one-class NLL (results/<tag>/scores.json) and its exact log-ratio
(results/<tag>/ratio.json).  Mass is only interesting if it beats the density it
is derived from; beating the RATIO is a stronger and much less likely result.

Output: results/<tag>/mass_vs_density.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdf.cli import load_config  # noqa: E402
from pcdf.device import resolve_device  # noqa: E402
from pcdf.models.density_pc import PCConfig, PCDetector  # noqa: E402
from pcdf.stages import _feat_dir, _feature_dims, _load_split, _tag  # noqa: E402

# Below this the float32 CDF difference cancels and the score degrades
# silently — see `tests/test_mass.py` and the note in `log_ball`.
EPS_FLOOR = 1e-3


@torch.no_grad()
def mass_sweep(det: PCDetector, Z: np.ndarray, epss: np.ndarray,
               scale: np.ndarray, batch: int = 64, chunk: int = 32
               ) -> tuple[np.ndarray, np.ndarray]:
    """(N,) log p(x) and (N, len(epss)) log P(box) for every eps."""
    dev = det.cfg.device
    sc = torch.from_numpy(np.ascontiguousarray(scale, np.float32)).to(dev)
    logps, masses = [], []
    for i in range(0, len(Z), batch):
        x = torch.from_numpy(np.ascontiguousarray(Z[i:i + batch], np.float32)).to(dev)
        logps.append(det.pc.log_prob(x, chunk=chunk).cpu().numpy())
        row = [det.pc.log_ball(x, sc * float(e), chunk=chunk).cpu().numpy()
               for e in epss]
        masses.append(np.stack(row, axis=1))
        if (i + batch) % 2048 < batch:
            print(f"[mass] {min(i + batch, len(Z))}/{len(Z)}", flush=True)
    return np.concatenate(logps), np.concatenate(masses)


def local_dimension(log_mass: np.ndarray, epss: np.ndarray,
                    scale_mean: float) -> np.ndarray:
    """
    Per-sample local dimension: the slope of log P(box) against log(2*eps).

    Fitted on the SMALLEST eps in the sweep that are still above the float32
    floor, because the identity log M ~ log p + d*log(2*eps) is a small-box
    statement; wide boxes bend the line and would bias the slope downward.
    """
    use = epss * scale_mean >= EPS_FLOOR
    if use.sum() < 3:
        use = np.zeros_like(epss, bool)
        use[:3] = True
    k = min(4, int(use.sum()))
    idx = np.where(use)[0][:k]
    xs = np.log(2 * epss[idx] * scale_mean)
    ys = log_mass[:, idx]
    xc = xs - xs.mean()
    return (ys - ys.mean(1, keepdims=True)) @ xc / (xc @ xc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--set", "-s", action="append", dest="overrides", default=[])
    # The grid must reach scales where the box stops being infinitesimal.  At
    # small eps EVERY smooth density looks full-dimensional (mass = density x
    # volume exactly), so the local dimension reads d and mass ranks samples
    # identically to density.  Manifold structure only appears once the box is
    # comparable to the thickness of the sheet, which is what the upper end is
    # for.
    ap.add_argument("--eps", type=float, nargs="*",
                    default=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0,
                             3.0, 10.0, 30.0, 100.0])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N test crops (smoke test)")
    a = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    from pcdf.eval.metrics import video_keys, video_level

    cfg = load_config(a.config, a.overrides)
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root, tag = Path(cfg["root"]), _tag(cfg)
    grid, C = _feature_dims(cfg)

    Ztr, _ = _load_split(cfg, "train", label=0)
    Zte, idx = _load_split(cfg, "test")
    if a.limit:
        # Stratified, because the index is ordered by method: a head slice is
        # all reals and every AUC comes back NaN.
        rs = np.random.default_rng(0)
        sel = rs.permutation(len(Zte))[:a.limit]
        sel.sort()
        Zte, idx = Zte[sel], {k: v[sel] for k, v in idx.items()}
    Ztr = Ztr.reshape(len(Ztr), -1)
    Zte = Zte.reshape(len(Zte), -1)
    y = idx["label"].astype(int)
    vkey = video_keys(idx)
    print(f"[mass] {tag}: {len(Zte)} test crops, d={Zte.shape[1]}, "
          f"{len(set(vkey.tolist()))} videos", flush=True)

    det = PCDetector(grid, grid, C, PCConfig(device=cfg["device"],
                                             seed=cfg["seed"], **cfg["pc"]))
    det.load(root / "models" / f"pc_{tag}.pt", Ztr,
             structure_cache=root / "models" / f"structure_{tag}.pkl")

    # eps is expressed in units of the per-feature TRAINING spread, so one eps
    # means the same thing in every coordinate even if the projection did not
    # leave them identically scaled.
    scale = Ztr.std(0) + 1e-6
    epss = np.asarray(sorted(a.eps), dtype=float)
    absolute = epss * float(scale.mean())
    print(f"[mass] eps x mean feature std = "
          f"{np.array2string(absolute, precision=4)}")
    if (absolute < EPS_FLOOR).any():
        print(f"[mass] WARNING: eps below the float32 floor {EPS_FLOOR} are "
              f"unreliable and are reported but not used for the LID slope")

    logp, logm = mass_sweep(det, Zte, epss, scale, a.batch, a.chunk)
    lid = local_dimension(logm, epss, float(scale.mean()))

    def report(score: np.ndarray) -> dict:
        """score: HIGHER = more likely to be a forgery."""
        vs, vy = video_level(score, vkey, y)
        per = {}
        for meth in sorted(set(idx["method"][y == 1].tolist())):
            m = (y == 0) | (idx["method"] == meth)
            v, ly = video_level(score[m], vkey[m], y[m])
            per[meth] = float(roc_auc_score(ly, v))
        return {"auc_video": float(roc_auc_score(vy, vs)),
                "auc_frame": float(roc_auc_score(y, score)),
                "per_method": per}

    rows: dict[str, dict] = {"density": report(-logp)}
    for j, e in enumerate(epss):
        r = report(-logm[:, j])
        # If mass is a monotone function of density the ranking is unchanged and
        # the AUC is identical to `density` — the leaves are then too smooth to
        # see the sheet, which is the hypothesis this whole script risks.
        r["spearman_vs_density"] = float(
            np.corrcoef(np.argsort(np.argsort(-logm[:, j])),
                        np.argsort(np.argsort(-logp)))[0, 1])
        r["eps_absolute"] = float(absolute[j])
        r["below_float32_floor"] = bool(absolute[j] < EPS_FLOOR)
        rows[f"mass_eps{e:g}"] = r
    rows["lid"] = report(-lid)                      # low dimension = forged
    rows["lid"]["mean_real"] = float(lid[y == 0].mean())
    rows["lid"]["mean_fake"] = float(lid[y == 1].mean())
    rows["lid"]["d_total"] = int(Zte.shape[1])
    # The dimension read off each ADJACENT pair of eps: d_local as a function of
    # scale.  A flat curve at d means the sweep never left the linear regime and
    # no manifold structure was resolved -- the honest way to see that.
    la = np.log(2 * absolute)
    curve = []
    for j in range(len(epss) - 1):
        slope = (logm[:, j + 1] - logm[:, j]) / (la[j + 1] - la[j])
        curve.append({"eps_lo": float(absolute[j]), "eps_hi": float(absolute[j + 1]),
                      "d_real": float(slope[y == 0].mean()),
                      "d_fake": float(slope[y == 1].mean()),
                      "auc_video": float(roc_auc_score(
                          *video_level(-slope, vkey, y)[::-1]))})
    rows["lid"]["scale_curve"] = curve
    # Kamkari's dual criterion as a single score: high density but low mass.
    jbig = int(np.argmax(epss))
    rows["density_minus_mass"] = report(logp - logm[:, jbig])

    prior = {}
    for name in ("scores.json", "ratio.json"):
        p = root / "results" / tag / name
        if p.exists():
            prior[name] = json.loads(p.read_text())
    baseline = {}
    if "scores.json" in prior:
        try:
            baseline["circuit_nll"] = \
                prior["scores.json"]["summary"]["PC"]["ffpp"]["auc_video"]
        except (KeyError, TypeError):
            pass
    if "ratio.json" in prior:
        baseline["circuit_log_ratio"] = prior["ratio.json"]["best"]["auc_video"]

    out = {"tag": tag, "n_test_crops": int(len(Zte)), "d": int(Zte.shape[1]),
           "eps_grid": epss.tolist(), "eps_absolute": absolute.tolist(),
           "eps_float32_floor": EPS_FLOOR,
           "scores": rows, "reference_scores": baseline}

    print(f"\n{'score':<24}{'video AUC':>10}{'frame AUC':>11}{'rho vs density':>16}")
    print("-" * 61)
    for k, v in rows.items():
        rho = v.get("spearman_vs_density")
        print(f"{k:<24}{v['auc_video']:>10.4f}{v['auc_frame']:>11.4f}"
              f"{('' if rho is None else f'{rho:.4f}'):>16}")
    print("-" * 61)
    for k, v in baseline.items():
        print(f"{k:<24}{v:>10.4f}   (previously measured)")
    print(f"\nlocal dimension  real {rows['lid']['mean_real']:.2f}  "
          f"fake {rows['lid']['mean_fake']:.2f}  of d={Zte.shape[1]}")
    print(f"\n{'eps band':<22}{'d real':>9}{'d fake':>9}{'video AUC':>11}")
    for c in rows["lid"]["scale_curve"]:
        band = f"{c['eps_lo']:.3g}-{c['eps_hi']:.3g}"
        print(f"{band:<22}{c['d_real']:>9.1f}{c['d_fake']:>9.1f}"
              f"{c['auc_video']:>11.4f}")
    best = max((k for k in rows), key=lambda k: rows[k]["auc_video"])
    print(f"best here: {best} at {rows[best]['auc_video']:.4f}")

    dest = root / "results" / tag / "mass_vs_density.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"[mass] wrote {dest}")


if __name__ == "__main__":
    main()
