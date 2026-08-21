"""
Does discriminative training break the tie with the GMM?

Measured baseline to beat: with the likelihood-ratio treatment, PC 0.828 and
GMM 0.830 on FF++ (SBI features) — the circuit's extra capacity bought nothing,
because both densities were fitted to model EVERYTHING while only a few
coordinates distinguish the classes.

This sweeps the hybrid objective

    L = λ·[generative NLL] + (1−λ)·[discriminative BCE on the log-ratio]

from λ=1 (what we did) to λ=0 (pure discriminative circuit pair). The
prediction from the PC literature (Gens & Domingos 2012: the tractable
DISCRIMINATIVE class is broader than the generative one) is that lower λ should
help, and that a deep circuit has far more discriminative capacity to exploit
than a GMM does.

Reported per λ: FF++ test AUC (the number that matters), the sanity AUC on
real-vs-own-blends, and localization patch-AUC — because a purely
discriminative fit might win detection while destroying the density semantics
that the localization and explanation claims rest on. That trade is the whole
point of the sweep.

    python scripts/hybrid_sweep.py --config configs/ffpp_sbi.yaml
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
from pcdf.models.density_pc import PCConfig  # noqa: E402
from pcdf.models.ratio import PCRatioDetector  # noqa: E402
from pcdf.stages import _feature_dims, _load_split, _tag  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--lams", nargs="*", type=float, default=[1.0, 0.7, 0.3, 0.1, 0.0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=8000)
    ap.add_argument("--n-loc", type=int, default=1200)
    ap.add_argument("--blend-suffix", default="blend",
                    help="which pseudo-fake feature set to fit p_blend on; "
                         "'blend-blendP' is the leak-free (pristine background) "
                         "version")
    ap.add_argument("--out-name", default="hybrid_sweep.json")
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root, tag = Path(cfg["root"]), _tag(cfg)
    grid, c = _feature_dims(cfg)
    d = grid * grid * c

    Ztr, _ = _load_split(cfg, "train", label=0)
    Zbl, _ = _load_split(cfg, "train", label=0, perturb=args.blend_suffix)
    Zvr, _ = _load_split(cfg, "val", label=0)
    Zvb, _ = _load_split(cfg, "val", label=0, perturb=args.blend_suffix)
    print(f"[hybrid] p_blend fitted on {args.blend_suffix!r}: {Zbl.shape}",
          flush=True)
    Zte, ite = _load_split(cfg, "test")
    rng = np.random.default_rng(cfg["seed"])
    if len(Zte) > args.n_test:
        sel = np.sort(rng.choice(len(Zte), args.n_test, replace=False))
        Zte, ite = Zte[sel], {k: v[sel] for k, v in ite.items()}
    y, vkey = ite["label"].astype(int), video_keys(ite)

    # localization subset (fakes with derived masks)
    Zf, idxf = _load_split(cfg, "test", label=1)
    if len(Zf) > args.n_loc:
        s2 = np.sort(rng.choice(len(Zf), args.n_loc, replace=False))
        Zf, idxf = Zf[s2], {k: v[s2] for k, v in idxf.items()}
    masks, have = load_masks(cfg, idxf, grid)
    Zf, masks = Zf[have], masks[have]

    from sklearn.metrics import roc_auc_score

    def auc(sc):
        vs, vy = video_level(sc, vkey, y)
        return float(roc_auc_score(vy, vs))

    flat = lambda Z: Z.reshape(len(Z), -1)
    results = {}
    for lam in args.lams:
        print(f"\n===== λ = {lam} =====", flush=True)
        det = PCRatioDetector(grid, grid, c,
                              PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"]))
        hist = det.fit_hybrid(flat(Ztr), flat(Zbl), Zvr[:1500].reshape(-1, d),
                              Zvb[:1500].reshape(-1, d),
                              structure_cache=root / "models" / f"structure_{tag}.pkl",
                              lam=lam, epochs=args.epochs)
        det.calibrate(Zvr[:1200].reshape(-1, d))

        s = det.score(flat(Zte))
        entry = {"ffpp_auc": {k: auc(v) for k, v in s.items() if not k.startswith("_")},
                 "val_real_vs_blend_auc": hist["val_auc"][-1]["auc"] if hist["val_auc"] else None,
                 "final_gen_nll": hist["gen"][-1], "final_disc_bce": hist["disc"][-1]}
        entry["best_ffpp"] = max(entry["ffpp_auc"].values())

        sl = det.score(flat(Zf))
        entry["localization"] = localization_metrics(sl["_patch_ratio_z"], masks)

        # does it still behave like a density?
        with torch.no_grad():
            lz_r = float(det.real.pc.log_partition())
            lz_b = float(det.blend.pc.log_partition())
        entry["log_partition"] = {"real": lz_r, "blend": lz_b}

        results[str(lam)] = entry
        print(f"[λ={lam}] FF++ best {entry['best_ffpp']:.4f}   "
              f"loc patchAUC {entry['localization'].get('patch_auc_pooled', 0):.4f}   "
              f"logZ {lz_r:+.1e}/{lz_b:+.1e}", flush=True)
        det.save(root / "models" / f"ratio_hybrid_lam{lam}_{tag}.pt")

    out = root / "results" / tag / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print("\n" + "=" * 66)
    print(f"{'lambda':>8} {'FF++ AUC':>10} {'loc patchAUC':>14} {'real-vs-blend':>14}")
    for lam, e in results.items():
        print(f"{lam:>8} {e['best_ffpp']:10.4f} "
              f"{e['localization'].get('patch_auc_pooled', 0):14.4f} "
              f"{(e['val_real_vs_blend_auc'] or 0):14.4f}")
    print(f"\n[hybrid] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
