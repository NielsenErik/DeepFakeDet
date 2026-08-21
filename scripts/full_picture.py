"""
Every stage of the system, on one dataset, in one table.

WHY THIS EXISTS.  The project's numbers are spread across `gap_waterfall.json`,
`scores.json`, `ratio.json`, `mass_vs_density.json` and
`sbi_eval_*_*.json`, each produced by a different script on a different
occasion.  That is how a target measured on FF++ RAW (Finding 11) sat next to
c23 measurements for two weeks, and how "our encoder is weak" survived four
months without anyone scoring the encoder cross-dataset.  This runs every stage
in one process, on one dataset, on the SAME crops, so the rows are comparable
by construction.

STAGES, in the order evidence is lost:

  published SBI              the reported number for this dataset, if any
  official SBI encoder       their released weights, our crops, end to end
  our SBI encoder            our checkpoint, same crops
  probe on projected feats   supervised ceiling for the coordinates the
                             circuit sees -- fitted on FF++ train, so it is a
                             CROSS-DATASET probe when the test set is not FF++
  circuit, one-class NLL     the density score family
  circuit, exact log-ratio   the two-circuit repair
  circuit, probability mass  Finding 9's score
  local dimension            Finding 9's geometric quantity

Everything below the encoder rows is scored from the SAME feature file, so a
difference between two of them is a difference in SCORING RULE and nothing else.

    python scripts/full_picture.py --dataset celebdf --target 0.9287
    python scripts/full_picture.py --dataset ffpp

Output: results/full_picture_<dataset>.json
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
from pcdf.stages import _feat_dir, _feature_dims, _load_split, _tag  # noqa: E402


def video_auc(score: np.ndarray, vkey: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    vids = sorted(set(vkey.tolist()))
    vs = np.array([score[vkey == v].mean() for v in vids])
    vy = np.array([int(y[vkey == v].max()) for v in vids])
    return float(roc_auc_score(vy, vs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--set", "-s", action="append", dest="overrides", default=[])
    ap.add_argument("--dataset", default="celebdf")
    ap.add_argument("--target", type=float, default=None,
                    help="published number for this dataset, for the top row")
    ap.add_argument("--eps", type=float, default=1.0,
                    help="box half-width for the mass score, in units of the "
                         "per-feature training std (Finding 9 sweep)")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from pcdf.models.density_pc import PCConfig, PCDetector
    from pcdf.models.ratio import PCRatioDetector

    cfg = load_config(a.config, a.overrides)
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root, tag = Path(cfg["root"]), _tag(cfg)
    grid, C = _feature_dims(cfg)

    fd = _feat_dir(cfg)
    if not (fd / f"{a.dataset}_test.npy").exists():
        raise SystemExit(f"[full] no features for {a.dataset} under {fd}\n"
                         f"       run: pcdf -c {a.config} features "
                         f"--dataset {a.dataset}")

    Ztr, _ = _load_split(cfg, "train", dataset="ffpp", label=0)
    Zte, idx = _load_split(cfg, "test", dataset=a.dataset)
    Ztr = Ztr.reshape(len(Ztr), -1)
    Zte = Zte.reshape(len(Zte), -1)
    y = idx["label"].astype(int)
    vkey = np.array([f"{m}/{Path(p).parent.name}"
                     for m, p in zip(idx["method"], idx["path"])])
    print(f"[full] {a.dataset}: {len(y)} crops, {len(set(vkey.tolist()))} videos, "
          f"{int((y == 1).sum())} forged, d={Zte.shape[1]}", flush=True)

    rows: list[tuple[str, float | None, str]] = []
    if a.target is not None:
        rows.append(("published SBI (reported)", a.target, "external"))

    # encoder rows come from the artefacts official_sbi_eval.py already wrote
    for who, label in (("official", "official SBI encoder, our crops"),
                       ("ours", "our SBI encoder")):
        p = root / "results" / f"sbi_eval_{who}_{a.dataset}.json"
        if p.exists():
            d = json.loads(p.read_text())
            best = max(v["auc_video_mean"] for v in d["by_crop_rule"].values())
            rows.append((label, best, "end-to-end"))

    # supervised ceiling for these coordinates: trained on FF++ train reals+fakes
    Ztr_lab, itr = _load_split(cfg, "train", dataset="ffpp")
    Ztr_lab = Ztr_lab.reshape(len(Ztr_lab), -1)
    ytr = itr["label"].astype(int)
    n = min(len(Ztr_lab), 40000)
    sel = np.random.default_rng(0).permutation(len(Ztr_lab))[:n]
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(Ztr_lab[sel], ytr[sel])
    s_probe = probe.decision_function(Zte)
    rows.append(("linear probe on projected features", video_auc(s_probe, vkey, y),
                 "supervised, fitted on FF++ train"))

    det = PCDetector(grid, grid, C, PCConfig(device=cfg["device"],
                                             seed=cfg["seed"], **cfg["pc"]))
    det.load(root / "models" / f"pc_{tag}.pt", Ztr,
             structure_cache=root / "models" / f"structure_{tag}.pkl")
    sc = det.score(Zte)
    nll_family = {k: video_auc(v, vkey, y) for k, v in sc.items()
                  if not k.startswith("_")}
    best_nll = max(nll_family, key=nll_family.get)
    rows.append((f"circuit, one-class ({best_nll})", nll_family[best_nll],
                 "one-class"))
    rows.append(("circuit, raw NLL", nll_family["nll"], "one-class"))

    rp = root / "models" / f"ratio_{tag}.pt"
    ratio_family = {}
    if rp.exists():
        rd = PCRatioDetector(grid, grid, C, PCConfig(device=cfg["device"],
                                                     seed=cfg["seed"], **cfg["pc"]))
        rd.load(rp, Ztr, structure_cache=root / "models" / f"structure_{tag}.pkl")
        rs = rd.score(Zte)
        ratio_family = {k: video_auc(v, vkey, y) for k, v in rs.items()
                        if not k.startswith("_")}
        b = max(ratio_family, key=ratio_family.get)
        rows.append((f"circuit, exact log-ratio ({b})", ratio_family[b], "ratio"))

    # Finding 9: probability mass and local dimension, same circuit
    scale = torch.from_numpy(Ztr.std(0) + 1e-6).to(cfg["device"])
    logp, m_lo, m_hi = [], [], []
    with torch.no_grad():
        for i in range(0, len(Zte), 64):
            x = torch.from_numpy(np.ascontiguousarray(Zte[i:i + 64], np.float32)
                                 ).to(cfg["device"])
            logp.append(det.pc.log_prob(x, chunk=32).cpu().numpy())
            m_lo.append(det.pc.log_ball(x, scale * a.eps / 3, chunk=32).cpu().numpy())
            m_hi.append(det.pc.log_ball(x, scale * a.eps, chunk=32).cpu().numpy())
    logp = np.concatenate(logp)
    m_lo, m_hi = np.concatenate(m_lo), np.concatenate(m_hi)
    lid = (m_hi - m_lo) / np.log(3.0)
    rows.append((f"circuit, probability mass (eps={a.eps:g})",
                 video_auc(-m_hi, vkey, y), "mass"))
    rows.append(("circuit, density - mass", video_auc(logp - m_hi, vkey, y), "mass"))
    rows.append(("circuit, local dimension", video_auc(-lid, vkey, y), "mass"))

    print(f"\n{'stage':<46}{'video AUC':>10}   note")
    print("-" * 78)
    for name, v, note in rows:
        print(f"{name:<46}{v:>10.4f}   {note}")
    print("-" * 78)
    print(f"local dimension: real {lid[y == 0].mean():.1f}  "
          f"fake {lid[y == 1].mean():.1f}  of d={Zte.shape[1]}")

    out = {"dataset": a.dataset, "tag": tag, "n_crops": int(len(y)),
           "d": int(Zte.shape[1]), "eps": a.eps,
           "rows": [{"stage": n, "auc_video": v, "note": t} for n, v, t in rows],
           "nll_family": nll_family, "ratio_family": ratio_family,
           "lid_real": float(lid[y == 0].mean()),
           "lid_fake": float(lid[y == 1].mean())}
    dest = root / "results" / f"full_picture_{a.dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"[full] wrote {dest}")


if __name__ == "__main__":
    main()
