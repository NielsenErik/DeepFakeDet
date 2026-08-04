"""
`pcdf` — the experiment pipeline.

Stages are separate commands with explicit inputs and outputs, so a run is
resumable and every artefact on disk is traceable to the command that made it:

    pcdf manifest   videos           -> manifests/<dataset>.csv
    pcdf ingest     videos+manifest  -> crops/ (+ landmarks, pseudo-masks)
    pcdf features   crops            -> features/<backbone>/*.npy (+ projector)
    pcdf fit-pc     features         -> models/pc_<tag>.pt (+ structure cache)
    pcdf baselines  features         -> models/baselines_<tag>.pkl
    pcdf evaluate   models+features  -> results/<tag>/scores.json
    pcdf explain    models+features  -> results/<tag>/localization.json + maps
    pcdf report     results          -> results/REPORT.md  (+ decision rubric)
    pcdf bench      -                -> results/bench.json (circuit throughput)

Everything is driven by a YAML config (see configs/) plus command-line
overrides; the resolved config is written next to every artefact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ── config ──────────────────────────────────────────────────────────────────

DEFAULTS: Dict = {
    "root": str(Path.home() / "deepfake_data"),
    "seed": 0,
    "device": "auto",
    "data": {
        "ffpp_root": "raw/ffpp/FaceForensics++_C23",
        "splits_dir": ".",
        "celebdf_root": "raw/celebdf",
        "df40_root": "raw/df40",
        "n_frames_train": 32,
        "n_frames_test": 32,
        "crop_size": 256,
        "margin": 1.3,
        "detector": "mediapipe",
        "workers": 12,
        "include_dfd": False,
    },
    "features": {
        "backbone": "clip",
        "batch": 64,
        "out_dim": 16,
        "grid": 8,
        "whiten": True,
        "max_projector_images": 20000,
    },
    "pc": {
        "n_sum_components": 8,
        "n_input_components": 8,
        "leaf_components": 4,
        "patch_method": "kd",
        "channel_method": "orc",
        "epochs": 60,
        "batch_size": 256,
        "lr": 5e-3,
        "patience": 8,
        "topq": 0.05,
    },
    "baselines": ["mahalanobis", "gmm", "patchcore", "flow"],
    "eval": {
        "test_datasets": ["ffpp", "celebdf", "df40"],
        "robustness": ["clean", "jpeg70", "jpeg50", "resize0.5", "blur"],
    },
}


def load_config(path: Optional[str], overrides: List[str]) -> Dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        import yaml

        with open(path) as fh:
            user = yaml.safe_load(fh) or {}
        _deep_update(cfg, user)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_path(cfg, key.split("."), _coerce(val))
    return cfg


def _deep_update(base: Dict, extra: Dict) -> Dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_path(cfg: Dict, keys: List[str], value) -> None:
    for k in keys[:-1]:
        cfg = cfg.setdefault(k, {})
    cfg[keys[-1]] = value


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.startswith("["):
        return json.loads(v)
    return v


def resolve(root: str | Path, rel: str | Path) -> Path:
    rel = Path(rel)
    return rel if rel.is_absolute() else Path(root) / rel


# ── stage: manifest ─────────────────────────────────────────────────────────

def cmd_manifest(cfg: Dict, args) -> None:
    from .data.manifest import (build_celebdf_manifest, build_df40_manifest,
                                build_ffpp_manifest, assert_identity_disjoint,
                                assert_no_fakes_in_train, summarize, write_manifest)

    root = Path(cfg["root"])
    out_dir = root / "manifests"
    dsets = args.datasets or ["ffpp"]

    for name in dsets:
        if name == "ffpp":
            recs = build_ffpp_manifest(
                resolve(root, cfg["data"]["ffpp_root"]),
                resolve(root, cfg["data"]["splits_dir"]),
                include_dfd=cfg["data"]["include_dfd"])
            assert_identity_disjoint([r for r in recs if r.method != "DeepFakeDetection"])
        elif name == "celebdf":
            recs = build_celebdf_manifest(resolve(root, cfg["data"]["celebdf_root"]))
        elif name == "df40":
            recs = build_df40_manifest(resolve(root, cfg["data"]["df40_root"]))
        else:
            raise KeyError(name)
        write_manifest(recs, out_dir / f"{name}.csv")
        print(f"\n=== {name} ===\n{summarize(recs)}")
        print(f"[manifest] wrote {out_dir / f'{name}.csv'}")


# ── stage: ingest (videos -> face crops) ────────────────────────────────────

def _ingest_one(task):
    """Worker: one video -> crops.  Imports inside so it is fork-safe."""
    # One thread per worker.  ONNX Runtime and OpenCV each default to using
    # every core, so N workers spawn N×24 threads and spend their time in the
    # scheduler — this was a 5× throughput loss before it was pinned.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import cv2

    cv2.setNumThreads(1)
    import numpy as np  # noqa: F401
    from .data.faces import ExtractConfig, FaceDetector, extract_video

    rec, out_dir, cfg_kwargs, ref = task
    global _DET
    try:
        det = _DET
    except NameError:
        # Give each worker its own BLOCK of cores.  Unpinned, N workers each
        # grab all 24 cores and the machine thrashes (5× throughput loss);
        # pinned to a single core, MediaPipe's internally threaded inference
        # loses ~4× as well.  A disjoint block per worker avoids both.
        try:
            import multiprocessing as _mp

            ident = _mp.current_process()._identity
            wid = (ident[0] - 1) if ident else 0
            ncpu = os.cpu_count() or 1
            n_workers = int(os.environ.get("PCDF_N_WORKERS", "1"))
            per = max(1, ncpu // max(n_workers, 1))
            start = (wid * per) % ncpu
            os.sched_setaffinity(0, {(start + j) % ncpu for j in range(per)})
        except (AttributeError, OSError, IndexError):
            pass
        det = _DET = FaceDetector(backend=cfg_kwargs.pop("detector", "auto"),
                                  device="cpu")
    ecfg = ExtractConfig(**{k: v for k, v in cfg_kwargs.items()
                            if k in ExtractConfig.__dataclass_fields__})
    if (Path(out_dir) / "meta.json").exists():
        return json.loads((Path(out_dir) / "meta.json").read_text())["n_frames"]
    meta = extract_video(rec.video, out_dir, ecfg, det, reference_video=ref)
    return meta["n_frames"]


def cmd_ingest(cfg: Dict, args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from .data.faces import crop_dir_for
    from .data.manifest import read_manifest, write_manifest

    root = Path(cfg["root"])
    man_path = root / "manifests" / f"{args.dataset}.csv"
    recs = read_manifest(man_path)
    if args.methods:
        recs = [r for r in recs if r.method in args.methods]
    if args.limit:
        recs = recs[:args.limit]
    crops_root = root / "crops"

    dcfg = cfg["data"]
    tasks = []
    for r in recs:
        n_frames = (dcfg["n_frames_train"] if r.split == "train"
                    else dcfg["n_frames_test"])
        kwargs = dict(n_frames=n_frames, size=dcfg["crop_size"],
                      margin=dcfg["margin"], detector=dcfg["detector"])
        # frame-aligned real counterpart -> derived localization mask
        ref = None
        if r.dataset == "ffpp" and r.label == 1 and args.masks:
            ref = str(resolve(root, dcfg["ffpp_root"]) / "real" / f"{r.ident}.mp4")
        tasks.append((r, str(crop_dir_for(crops_root, r)), kwargs, ref))

    t0 = time.time()
    done = 0
    os.environ["PCDF_N_WORKERS"] = str(dcfg["workers"])
    with ProcessPoolExecutor(max_workers=dcfg["workers"]) as pool:
        for rec, n in zip(recs, pool.map(_ingest_one, tasks, chunksize=4)):
            rec.n_frames = int(n)
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"[ingest] {done}/{len(recs)} videos  {rate:.1f} vid/s  "
                      f"eta {(len(recs) - done) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)

    kept = [r for r in recs if r.n_frames > 0]
    write_manifest(kept, root / "manifests" / f"{args.dataset}_ingested.csv")
    print(f"[ingest] {len(kept)}/{len(recs)} videos yielded crops "
          f"({sum(r.n_frames for r in kept)} frames) in {(time.time() - t0) / 60:.1f} min")
    if args.purge_videos:
        _purge(recs)


def _purge(recs) -> None:
    """Crop-then-purge: the crops are the dataset from here on, and the disk
    budget does not fit both."""
    freed = 0
    for r in recs:
        p = Path(r.video)
        if r.n_frames > 0 and p.exists():
            freed += p.stat().st_size
            p.unlink()
    print(f"[ingest] purged source videos, freed {freed / 1e9:.1f} GB")


# ── entry point ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("pcdf")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--set", "-s", action="append", dest="overrides", default=[],
                   help="override config, e.g. -s pc.epochs=100")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="build dataset manifests")
    m.add_argument("--datasets", nargs="*", default=["ffpp"])
    m.set_defaults(fn=cmd_manifest)

    i = sub.add_parser("ingest", help="videos -> face crops")
    i.add_argument("--dataset", default="ffpp")
    i.add_argument("--methods", nargs="*", default=None)
    i.add_argument("--limit", type=int, default=None)
    i.add_argument("--masks", action="store_true",
                   help="derive pseudo ground-truth masks from the aligned real video")
    i.add_argument("--purge-videos", action="store_true")
    i.set_defaults(fn=cmd_ingest)

    from .stages import register_stages
    register_stages(sub)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    # resolve "auto" ONCE, here, so every stage and every worker sees the same
    # concrete device string in the config it writes next to its artefacts
    from .device import resolve_device

    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    Path(cfg["root"]).mkdir(parents=True, exist_ok=True)
    args.fn(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
