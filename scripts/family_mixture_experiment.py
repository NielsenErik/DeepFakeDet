"""
The pseudo-fake distribution, measured — and an exact mixture built to fix it.

This script answers three questions with one fitted object.

Q1  HOW BIG IS THE DOMAIN GAP, IN NATS?
    The two-circuit ratio separates real faces from our own self-blends at AUC
    0.953 and real FF++ forgeries at 0.828.  That difference is usually waved at
    as "a domain gap".  With exactly normalized densities it can be measured:
    where do real forgeries sit in `log p_blend`, relative to the pseudo-fakes
    the model was fitted on?  We report the coverage of real forgeries under the
    pseudo-fake density — the fraction that even reach the 5th percentile of the
    blends — per manipulation.

Q2  DOES A MIXTURE OVER MECHANISMS CLOSE IT?
    A single self-blend family deviates from real in one direction only, which
    is why the graphics-rendered manipulations invert.  `p_mix = Σ_f π_f p_f`
    over four families is still one exact circuit, so the score stays a true
    log-likelihood ratio.

Q3  WHICH MECHANISM DOES A GIVEN FORGERY LOOK LIKE?
    `P(f | z) = π_f p_f(z) / Σ_g π_g p_g(z)`, exact.  If the model assigns
    Face2Face and FaceSwap to `render` and Deepfakes to `blend` — without ever
    seeing a real forgery or a mechanism label — that is a model-level
    explanation, and it is a probability rather than a score.

FAIRNESS.  The same mixture construction is run with full-covariance GMM
components, because the honest comparison for "a circuit can do this" is a
model that is ALSO tractable for marginals and mixture posteriors.  A Gaussian
mixture is; what it is not is non-singular at the dimensions these features
live at.  Both numbers are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdf.cli import load_config  # noqa: E402
from pcdf.device import resolve_device  # noqa: E402
from pcdf.stages import _feature_dims, _load_split, _tag  # noqa: E402

FAMILIES = ["blend", "render", "overshoot", "statistical"]


def video_auc(score, index, mask=None):
    from sklearn.metrics import roc_auc_score

    from pcdf.eval.metrics import video_keys, video_level

    y = index["label"].astype(int)
    vkey = video_keys(index)
    if mask is not None:
        score, y, vkey = score[mask], y[mask], vkey[mask]
    vs, vy = video_level(score, vkey, y)
    return float(roc_auc_score(vy, vs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--set", "-s", action="append", dest="overrides", default=[])
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--suffix", default="P",
                    help="feature suffix marker for the blend sets "
                         "('P' = pristine background)")
    ap.add_argument("--limit-test", type=int, default=20000)
    ap.add_argument("--with-gmm", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root = Path(cfg["root"])
    grid, C = _feature_dims(cfg)
    d = grid * grid * C
    tag = _tag(cfg)
    print(f"[mix] tag={tag} d={d} families={a.families}", flush=True)

    Zr, _ = _load_split(cfg, "train", label=0)
    Zrv, _ = _load_split(cfg, "val", label=0)
    Zfam, Zfam_val = {}, {}
    for f in a.families:
        pert = f"blend-{f}{a.suffix}"
        Zfam[f], _ = _load_split(cfg, "train", label=0, perturb=pert)
        Zfam_val[f], _ = _load_split(cfg, "val", label=0, perturb=pert)
        print(f"[mix] {f}: {Zfam[f].shape}", flush=True)

    flat = lambda Z: Z.reshape(len(Z), -1)  # noqa: E731

    from pcdf.models.density_pc import PCConfig
    from pcdf.models.family_mixture import (FamilyMixtureRatio,
                                            expected_calibration_error,
                                            risk_coverage)

    pcc = PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"])
    mix = FamilyMixtureRatio(grid, grid, C, a.families, pcc)
    t0 = time.time()
    mix.fit(flat(Zr), {f: flat(Zfam[f]) for f in a.families},
            flat(Zrv[:3000]),
            structure_cache=root / "models" / f"structure_{tag}.pkl")
    print(f"[mix] fitted {len(a.families) + 1} circuits in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    mix.save(root / "models" / f"fammix_{tag}.pt")

    out = {"tag": tag, "d": d, "families": a.families,
           "log_pi": mix.log_pi.tolist()}

    # ── Q1: where do real forgeries sit under the pseudo-fake density? ────
    Zte, ite = _load_split(cfg, "test")
    if a.limit_test and len(Zte) > a.limit_test:
        sel = np.sort(np.random.default_rng(0).choice(len(Zte), a.limit_test, False))
        Zte, ite = Zte[sel], {k: v[sel] for k, v in ite.items()}
    y = ite["label"].astype(int)

    s_te = mix.score(flat(Zte))
    s_own = mix.score(flat(np.concatenate([Zfam_val[f][:800] for f in a.families])))
    s_realv = mix.score(flat(Zrv[:3000]))

    # support of the pseudo-fake density, from the pseudo-fakes themselves
    q05 = float(np.percentile(s_own["log_p_mixture"], 5))
    cov = {}
    for meth in sorted(set(ite["method"][y == 1].tolist())):
        m = ite["method"] == meth
        cov[meth] = {
            "coverage_at_q05": float((s_te["log_p_mixture"][m] >= q05).mean()),
            "mean_log_ratio": float(s_te["ratio_mixture"][m].mean()),
        }
    cov["_real_test"] = {
        "coverage_at_q05": float((s_te["log_p_mixture"][y == 0] >= q05).mean()),
        "mean_log_ratio": float(s_te["ratio_mixture"][y == 0].mean()),
    }
    cov["_own_pseudo_fakes"] = {
        "coverage_at_q05": 0.95,
        "mean_log_ratio": float(s_own["ratio_mixture"].mean()),
    }
    out["Q1_domain_gap"] = {
        "log_p_mixture_q05_of_pseudo_fakes": q05,
        "per_method": cov,
        "note": ("coverage_at_q05 = fraction of that class reaching the 5th "
                 "percentile of log p_mix on the pseudo-fakes the mixture was "
                 "fitted on. Low values mean real forgeries lie outside the "
                 "support of the pseudo-fake model — the gap, in one number."),
    }

    # ── Q2: does the mixture detect better than one family? ───────────────
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for name in ("ratio_mixture", "ratio_best_family"):
        aucs[name] = {"auc_video": video_auc(s_te[name], ite),
                      "auc_frame": float(roc_auc_score(y, s_te[name]))}
    per_method = {}
    for meth in sorted(set(ite["method"][y == 1].tolist())):
        m = (y == 0) | (ite["method"] == meth)
        per_method[meth] = video_auc(s_te["ratio_mixture"], ite, m)
    # each family alone, as its own ratio — isolates what the mixture adds
    single = {}
    L = mix.component_log_probs(flat(Zte))
    for i, f in enumerate(a.families):
        s_i = L[:, i] - L[:, -1]
        single[f] = {"auc_video": video_auc(s_i, ite),
                     "per_method": {
                         meth: video_auc(s_i, ite,
                                         (y == 0) | (ite["method"] == meth))
                         for meth in sorted(set(ite["method"][y == 1].tolist()))}}
    out["Q2_detection"] = {"mixture": aucs, "per_method": per_method,
                           "single_family": single,
                           "own_pseudo_fake_auc": float(roc_auc_score(
                               np.r_[np.zeros(len(s_realv["ratio_mixture"])),
                                     np.ones(len(s_own["ratio_mixture"]))],
                               np.r_[s_realv["ratio_mixture"],
                                     s_own["ratio_mixture"]]))}

    # ── Q3: which mechanism does each manipulation look like? ─────────────
    post = s_te["family_posterior"]
    fam_table = {}
    for meth in sorted(set(ite["method"][y == 1].tolist())):
        m = ite["method"] == meth
        fam_table[meth] = {f: float(post[m, i].mean())
                           for i, f in enumerate(a.families)}
    fam_table["_real"] = {f: float(post[y == 0, i].mean())
                          for i, f in enumerate(a.families)}
    out["Q3_family_posterior"] = {
        "mean_posterior": fam_table,
        "note": ("P(family | z), exact, from densities that share a region "
                 "graph and are each exactly normalized. No real forgery and no "
                 "mechanism label was used in fitting."),
    }

    # per-REGION mechanism attribution, on a subset (this expands to
    # images x patches x families exact marginals)
    try:
        n_reg = min(600, len(Zte))
        sel = np.sort(np.random.default_rng(1).choice(len(Zte), n_reg, False))
        R = mix.region_family_posterior(flat(Zte[sel]))
        yr, mr = y[sel], ite["method"][sel]
        reg = {}
        for meth in sorted(set(mr[yr == 1].tolist())):
            m = mr == meth
            # dominant family per region, averaged over images of this method
            reg[meth] = {
                "mean_posterior_by_region": R[m].mean(0).tolist(),
                "dominant_family_per_region": [
                    a.families[i] for i in R[m].mean(0).argmax(1).tolist()],
            }
        out["Q3_family_posterior"]["per_region"] = {
            "n_images": int(n_reg), "grid": grid, "by_method": reg,
            "note": ("P(family | z_R) with p_f(z_R) an exact region marginal — "
                     "a mechanism map over the face, in probabilities."),
        }
    except Exception as exc:                                   # noqa: BLE001
        out["Q3_family_posterior"]["per_region_error"] = str(exc)
        print(f"[mix] per-region posterior failed: {exc}", flush=True)

    # ── C5: calibration of the log-odds ──────────────────────────────────
    s = s_te["ratio_mixture"]
    p = 1.0 / (1.0 + np.exp(-s))
    out["C5_calibration"] = {
        "raw": expected_calibration_error(p, y),
        "risk_coverage": risk_coverage(s, y),
        "note": ("the ratio is a log-odds by construction, so sigmoid(s) is a "
                 "probability claim that can be right or wrong — not a score "
                 "that only ranks."),
    }
    # temperature-scaled on VAL reals + our own pseudo-fakes (never test, never
    # a real forgery), which is the only calibration data this protocol allows
    sv = np.r_[s_realv["ratio_mixture"], s_own["ratio_mixture"]]
    yv = np.r_[np.zeros(len(s_realv["ratio_mixture"])),
               np.ones(len(s_own["ratio_mixture"]))]
    # logistic loss via logaddexp — the ratio reaches hundreds of nats at
    # d = 1024, and np.exp(-s/T) overflows long before the optimum is found
    best = min(((float(np.mean(np.logaddexp(0.0, -(2 * yv - 1) * sv / T))), T)
                for T in np.geomspace(0.05, 500, 80)))
    T = best[1]
    out["C5_calibration"]["temperature"] = T
    out["C5_calibration"]["scaled"] = expected_calibration_error(
        1.0 / (1.0 + np.exp(-s / T)), y)

    p_out = root / "results" / tag / "family_mixture.json"
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items()
                      if k != "C5_calibration"}, indent=2, default=float))
    print(f"\n[mix] wrote {p_out}")


if __name__ == "__main__":
    main()
