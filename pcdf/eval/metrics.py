"""
Evaluation: frame- and video-level detection metrics, per manipulation method
and across datasets.

Conventions chosen to match the deepfake literature, so the numbers here are
comparable to published ones rather than to themselves only:

* AUC is the headline, computed at VIDEO level with the frame scores averaged
  per video (the FF++ / Celeb-DF convention).  Frame-level AUC is reported next
  to it because a large gap between them is informative: it means the detector
  fires on some frames only, which is exactly what a localized artifact model
  should do.
* Cross-dataset evaluation never re-fits anything.  The circuit and every
  baseline are fitted once on FF++ real training faces; Celeb-DF / DF40 are
  scored by the same frozen models.
* When a test set ships no reals of its own (DF40 mirrors are fake-only), the
  real side is FF++ test reals and the result is flagged `mixed_reals=True`,
  because a real/fake pair from different sources can be separated by source
  statistics alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── primitive metrics ───────────────────────────────────────────────────────

def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def eer(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + (1 - tpr[i])) / 2)


def video_level(scores: np.ndarray, videos: np.ndarray, labels: np.ndarray,
                how: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate frame scores to one score per video."""
    out_s, out_y = [], []
    for v in np.unique(videos):
        m = videos == v
        s = scores[m]
        if how == "mean":
            out_s.append(float(s.mean()))
        elif how == "max":
            out_s.append(float(s.max()))
        else:                                    # top-quartile mean: robust max
            out_s.append(float(np.sort(s)[-max(1, len(s) // 4):].mean()))
        out_y.append(int(labels[m][0]))
    return np.array(out_s), np.array(out_y)


def video_keys(index: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Aggregation key for video-level metrics.

    FF++ names every manipulation of a source pair identically —
    `Deepfakes/000_003` and `Face2Face/000_003` are both stem `000_003` — so
    grouping by the stem alone would pool frames from DIFFERENT forgeries into
    one "video" and quietly corrupt every video-level number.  The key is
    therefore (dataset, method, stem).
    """
    return np.array([f"{d}/{m}/{v}" for d, m, v in
                     zip(index["dataset"], index["method"], index["video"])])


def metric_block(scores: np.ndarray, labels: np.ndarray, videos: np.ndarray
                 ) -> Dict[str, float]:
    vs, vy = video_level(scores, videos, labels)
    return {
        "auc_video": auc(vy, vs),
        "auc_frame": auc(labels, scores),
        "ap_video": average_precision(vy, vs),
        "eer_video": eer(vy, vs),
        "n_frames": int(len(scores)),
        "n_videos": int(len(vs)),
    }


# ── orchestration ───────────────────────────────────────────────────────────

def _score_all_models(cfg: Dict, tag: str, Z: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """Every model's full score family on one feature block."""
    import pickle

    import torch  # noqa: F401

    from ..models.density_pc import PCConfig, PCDetector

    root = Path(cfg["root"])
    from ..stages import _feature_dims

    grid, c = _feature_dims(cfg)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    pc_path = root / "models" / f"pc_{tag}.pt"
    if pc_path.exists():
        pcc = PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"])
        det = PCDetector(grid, grid, c, pcc)
        det.load(pc_path, Z[:8].reshape(8, -1),
                 structure_cache=root / "models" / f"structure_{tag}.pkl")
        s = det.score(Z.reshape(len(Z), -1))
        out["PC"] = {k: v for k, v in s.items() if not k.startswith("_")}

    b_path = root / "models" / f"baselines_{tag}.pkl"
    if b_path.exists():
        with open(b_path, "rb") as fh:
            models = pickle.load(fh)
        for name, m in models.items():
            s = m.score(Z)
            out[name] = {k: v for k, v in s.items() if not k.startswith("_")}
    return out


def _score_sbi(cfg: Dict, index: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """The supervised real-only competitor, scored on the same crops."""
    from ..models.supervised import score_sbi

    ckpt = Path(cfg["root"]) / "models" / "sbi_effnetb4.pt"
    if not ckpt.exists():
        return None
    items = [(p, None) for p in index["path"]]
    return score_sbi(cfg, ckpt, items)


def evaluate_all(cfg: Dict, tag: str, datasets: Sequence[str] = ("ffpp",),
                 robustness: bool = False) -> Dict:
    from ..stages import _load_split

    root = Path(cfg["root"])
    results: Dict = {"tag": tag, "config": cfg, "per_dataset": {}, "summary": {}}

    # FF++ test reals double as the real side for fake-only cross-datasets
    Zr_ffpp, idx_r_ffpp = _load_split(cfg, "test", dataset="ffpp", label=0)

    for ds in datasets:
        try:
            Z, idx = _load_split(cfg, "test", dataset=ds)
        except FileNotFoundError:
            print(f"[eval] no features for {ds}, skipping")
            continue
        mixed = bool((idx["label"] == 0).sum() == 0)
        if mixed:
            Z = np.concatenate([Zr_ffpp, Z])
            idx = {k: np.concatenate([idx_r_ffpp[k], v]) for k, v in idx.items()}

        vkey = video_keys(idx)
        scored = _score_all_models(cfg, tag, Z)
        sbi = _score_sbi(cfg, idx)
        if sbi is not None:
            scored["SBI"] = {"prob": sbi}
        block: Dict = {"mixed_reals": mixed, "models": {}}
        methods = sorted(set(idx["method"][idx["label"] == 1].tolist()))

        for model, family in scored.items():
            per_score: Dict[str, Dict] = {}
            for score_name, s in family.items():
                entry = {"all": metric_block(s, idx["label"], vkey)}
                for meth in methods:
                    m = (idx["label"] == 0) | (idx["method"] == meth)
                    entry[meth] = metric_block(s[m], idx["label"][m], vkey[m])
                per_score[score_name] = entry
            block["models"][model] = per_score
        results["per_dataset"][ds] = block

    if robustness:
        results["robustness"] = _robustness_sweep(cfg, tag, results)

    # summary: best score function per model per dataset, by video AUC
    summary: Dict[str, Dict[str, float]] = {}
    for ds, block in results["per_dataset"].items():
        for model, per_score in block["models"].items():
            best = max(per_score.items(),
                       key=lambda kv: (kv[1]["all"]["auc_video"]
                                       if np.isfinite(kv[1]["all"]["auc_video"]) else -1))
            summary.setdefault(model, {})[ds] = {
                "best_score": best[0],
                "auc_video": best[1]["all"]["auc_video"],
                "auc_frame": best[1]["all"]["auc_frame"],
            }
    results["summary"] = summary
    return results


def _robustness_sweep(cfg: Dict, tag: str, results: Dict) -> Dict:
    """
    Re-score the FF++ test split under each perturbed feature set that exists
    on disk (`pcdf features --perturb <name>` writes them), with every model
    frozen.

    The score function is PINNED to whatever won on clean data — re-picking the
    best score per perturbation would let a model launder a collapse into an
    apparent success by silently switching criteria.  Degradation is reported
    in absolute AUC points against clean, because that is what a deployment
    cares about.
    """
    from ..stages import _load_split
    from ..data.transforms import PERTURBATIONS

    clean = results["per_dataset"].get("ffpp", {}).get("models", {})
    pinned = {}
    for model, per_score in clean.items():
        best = max(per_score.items(),
                   key=lambda kv: kv[1]["all"]["auc_video"]
                   if np.isfinite(kv[1]["all"]["auc_video"]) else -1)
        pinned[model] = (best[0], best[1]["all"]["auc_video"])

    out: Dict[str, Dict] = {}
    for pert in PERTURBATIONS:
        if pert == "clean":
            continue
        try:
            Z, idx = _load_split(cfg, "test", dataset="ffpp", perturb=pert)
        except FileNotFoundError:
            continue
        vkey = video_keys(idx)
        scored = _score_all_models(cfg, tag, Z)
        sbi = _score_sbi(cfg, idx)
        if sbi is not None:
            scored["SBI"] = {"prob": sbi}
        entry = {}
        for model, family in scored.items():
            name, clean_auc = pinned.get(model, (None, float("nan")))
            if name not in family:
                continue
            m = metric_block(family[name], idx["label"], vkey)
            entry[model] = {"score": name, "auc_video": m["auc_video"],
                            "auc_clean": clean_auc,
                            "delta": m["auc_video"] - clean_auc}
        out[pert] = entry
        print(f"[eval] robustness {pert}: " +
              ", ".join(f"{k} {v['auc_video']:.3f} ({v['delta']:+.3f})"
                        for k, v in entry.items()))
    return out


def save_results(results: Dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float))
    return path
