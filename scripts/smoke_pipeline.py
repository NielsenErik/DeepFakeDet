"""
End-to-end smoke test of every post-feature stage, on synthetic data, on CPU.

Purpose: catch integration bugs (shape mismatches, index/key drift, missing
files, silent NaNs) in minutes on a laptop instead of after an hour of GPU
feature extraction.  It fabricates a feature set with the exact on-disk layout
the real pipeline produces — `features/<backbone>/ffpp_<split>.npy` plus an
index JSON, crops with derived masks — then runs fit-pc, baselines, evaluate,
explain, ablate-structure and report against it.

The fake data has REAL structure: fakes get a localized perturbation in a few
patches of a few frames, so a working pipeline must produce AUC > 0.5 and
localization above chance.  If this script reports chance-level numbers, the
bug is in the code, not in the representation.

    python scripts/smoke_pipeline.py [--root /tmp/pcdf_smoke]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GRID = 4                      # 4x4 = 16 patches
CDIM = 6                      # channels per patch -> d = 96
N_VIDEOS = {"train": 40, "val": 12, "test": 24}
FRAMES = 4
METHODS = ["Deepfakes", "Face2Face"]


def synth_features(root: Path, rng: np.random.Generator) -> None:
    """Correlated real patches; fakes get a localized shift in 2-3 patches."""
    fdir = root / "features" / "clip"
    fdir.mkdir(parents=True, exist_ok=True)
    P = GRID * GRID

    # a shared low-rank structure so the channel dependencies are non-trivial
    W = rng.normal(size=(3, CDIM))

    def sample_real(n: int) -> np.ndarray:
        z = rng.normal(size=(n, P, 3))
        base = z @ W
        return (base + 0.4 * rng.normal(size=(n, P, CDIM))).astype(np.float32)

    for split, n_vid in N_VIDEOS.items():
        Zs, index = [], {k: [] for k in ("video", "method", "label", "split",
                                         "dataset", "frame", "path")}
        # reals
        for v in range(n_vid):
            Z = sample_real(FRAMES)
            for f in range(FRAMES):
                Zs.append(Z[f])
                _add(index, f"real{v:03d}", "real", 0, split, f,
                     str(root / "crops" / "ffpp" / "real" / f"real{v:03d}" / f"{f:03d}.jpg"))
        # fakes (test/val only, mirroring the real-only training protocol)
        if split != "train":
            for meth in METHODS:
                for v in range(n_vid // 2):
                    Z = sample_real(FRAMES)
                    hot = rng.choice(P, size=3, replace=False)
                    Z[:, hot] += rng.normal(1.8, 0.3, size=(FRAMES, 3, CDIM))
                    for f in range(FRAMES):
                        Zs.append(Z[f])
                        vid = f"{meth}{v:03d}"
                        _add(index, vid, meth, 1, split, f,
                             str(root / "crops" / "ffpp" / meth / vid / f"{f:03d}.jpg"))
                        _write_mask(root, meth, vid, f, hot)
        np.save(fdir / f"ffpp_{split}.npy", np.stack(Zs).astype(np.float16))
        (fdir / f"ffpp_{split}_index.json").write_text(json.dumps(index))
        print(f"[smoke] {split}: {len(Zs)} frames")


def _add(index, video, method, label, split, frame, path) -> None:
    index["video"].append(video)
    index["method"].append(method)
    index["label"].append(label)
    index["split"].append(split)
    index["dataset"].append("ffpp")
    index["frame"].append(frame)
    index["path"].append(path)


def _write_mask(root: Path, method: str, video: str, frame: int,
                hot: np.ndarray) -> None:
    """Derived-mask stand-in at the same 64x64 resolution the ingester writes."""
    import cv2

    d = root / "crops" / "ffpp" / method / video
    d.mkdir(parents=True, exist_ok=True)
    m = np.zeros((GRID, GRID), np.uint8)
    for p in hot:
        m[p // GRID, p % GRID] = 255
    cv2.imwrite(str(d / f"mask_{frame:03d}.png"),
                cv2.resize(m, (64, 64), interpolation=cv2.INTER_NEAREST))
    if not (d / f"{frame:03d}.jpg").exists():        # figures need an image
        cv2.imwrite(str(d / f"{frame:03d}.jpg"),
                    np.full((64, 64, 3), 127, np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/pcdf_smoke")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if root.exists() and not args.keep:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    synth_features(root, rng)

    from pcdf.cli import DEFAULTS, _deep_update

    cfg = json.loads(json.dumps(DEFAULTS))
    _deep_update(cfg, {
        "root": str(root), "device": "cpu",
        "features": {"backbone": "clip", "grid": GRID, "out_dim": CDIM},
        "pc": {"n_sum_components": 4, "n_input_components": 4,
               "leaf_components": 3, "epochs": 8, "batch_size": 64,
               "patch_method": "kd", "channel_method": "chow_liu", "patience": 4},
        "baselines": ["mahalanobis", "gmm", "patchcore", "flow"],
    })

    class A:                                  # stand-in for argparse namespaces
        def __init__(self, **kw):
            self.__dict__.update(kw)

    from pcdf import stages

    print("\n=== fit-pc ===")
    stages.cmd_fit_pc(cfg, A())
    print("\n=== baselines ===")
    stages.cmd_baselines(cfg, A(only=None))
    print("\n=== evaluate ===")
    stages.cmd_evaluate(cfg, A(datasets=["ffpp"], robustness=False))
    print("\n=== explain ===")
    stages.cmd_explain(cfg, A(n_images=48, no_figures=False))
    print("\n=== ablate-structure ===")
    stages.cmd_ablate_structure(cfg, A(variants=["random/random", "kd/chow_liu", "kd/orc"],
                                       epochs=6, limit_test=400))
    print("\n=== report ===")
    stages.cmd_report(cfg, A())

    tag = stages._tag(cfg)
    report = root / "results" / tag / "REPORT.md"
    print("\n" + "=" * 70)
    print(report.read_text())

    verdict = json.loads((root / "results" / tag / "verdict.json").read_text())
    scores = json.loads((root / "results" / tag / "scores.json").read_text())
    pc_auc = scores["summary"].get("PC", {}).get("ffpp", {}).get("auc_video")
    print(f"\n[smoke] PC video AUC on synthetic data: {pc_auc}")
    ok = pc_auc is not None and pc_auc > 0.6
    print(f"[smoke] {'PASS' if ok else 'FAIL'} — "
          f"{'pipeline recovers the planted signal' if ok else 'planted signal NOT recovered'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
