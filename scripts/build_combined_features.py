"""
Build the `combined` arm by concatenating two arms that already exist on disk.

`configs/ffpp_combined.yaml` describes this arm and `build_extractor` has no
branch for it — the config was written, the extractor never was.  It does not
need to be: the two sources are stored as per-patch arrays over the SAME crops
in the SAME order, so the arm is a concatenation along the channel axis, not a
re-extraction.  The config's own comment says as much: d = 64 x (16 + 95) = 7104.

  sbi       (N, 64, 16)  blending geometry; the only representation that closed
                         the probe-minus-one-class gap (0.047)
  spectral  (N, 64, 95)  Corvi et al. residual spectrum; far more uniform across
                         manipulation types (probe 0.73-0.84 vs CLIP 0.59-0.92)

WHY STANDARDIZE.  The two sources are on completely different scales — `sbi` is
PCA-whitened to unit variance, `spectral` is raw band energies and moments.
Concatenated raw, the circuit's structure learner and its Gaussian leaves would
be driven almost entirely by whichever source has the larger numbers, and the
arm would silently be a slower copy of that source.  Each channel is
standardized with statistics computed on TRAIN ONLY, then applied unchanged to
val/test/blend, so nothing about the evaluation splits leaks into the transform.

ALIGNMENT IS CHECKED, NOT ASSUMED.  If the two indices ever disagree by one row
the concatenation is meaningless and every number downstream is garbage, with
nothing in the output to show it.  The script refuses to write in that case.

    python scripts/build_combined_features.py
    pcdf -c configs/ffpp_combined.yaml fit-pc
    python scripts/mass_vs_density.py -c configs/ffpp_combined.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdf.cli import load_config  # noqa: E402

SOURCES = ("sbi", "spectral")
SPLITS = ("train", "val", "test", "train_blend", "val_blend")
CHUNK = 4096


def _paths(fd: Path, split: str) -> tuple[Path, Path]:
    return fd / f"ffpp_{split}.npy", fd / f"ffpp_{split}_index.json"


def channel_stats(feat_root: Path, src: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over TRAIN, streamed so a 1.6 GB array is never
    materialised twice."""
    Z = np.load(feat_root / src / "ffpp_train.npy", mmap_mode="r")
    n, P, C = Z.shape
    tot = n * P
    s = np.zeros(C, np.float64)
    ss = np.zeros(C, np.float64)
    for i in range(0, n, CHUNK):
        blk = np.asarray(Z[i:i + CHUNK], np.float64).reshape(-1, C)
        s += blk.sum(0)
        ss += (blk * blk).sum(0)
    mean = s / tot
    var = np.maximum(ss / tot - mean * mean, 0.0)
    return mean.astype(np.float32), (np.sqrt(var) + 1e-6).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_combined.yaml")
    ap.add_argument("--set", "-s", action="append", dest="overrides", default=[])
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing combined arm")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    feat_root = Path(cfg["root"]) / "features"
    out = feat_root / cfg["features"]["backbone"]
    if out.exists() and any(out.glob("*.npy")) and not a.force:
        raise SystemExit(f"[combined] {out} already has features; pass --force "
                         f"to rebuild")
    out.mkdir(parents=True, exist_ok=True)

    for src in SOURCES:
        if not (feat_root / src / "ffpp_train.npy").exists():
            raise SystemExit(f"[combined] source arm '{src}' is not built "
                             f"({feat_root / src}) — run `pcdf features` for it first")

    stats = {src: channel_stats(feat_root, src) for src in SOURCES}
    for src, (m, s) in stats.items():
        print(f"[combined] {src}: {len(m)} channels, "
              f"mean in [{m.min():+.3g}, {m.max():+.3g}], "
              f"std in [{s.min():.3g}, {s.max():.3g}]")

    written = []
    for split in SPLITS:
        srcs = []
        for src in SOURCES:
            npy, idx = _paths(feat_root / src, split)
            if not (npy.exists() and idx.exists()):
                break
            srcs.append((src, npy, idx))
        if len(srcs) != len(SOURCES):
            print(f"[combined] {split}: not present in every source, skipped")
            continue

        # alignment: same rows, same order, or the concatenation is nonsense
        indices = [json.loads(p.read_text()) for _, _, p in srcs]
        if any(ix["path"] != indices[0]["path"] for ix in indices[1:]):
            raise SystemExit(
                f"[combined] {split}: source indices disagree — the arms were "
                f"extracted over different crops or in a different order. "
                f"Refusing to write; re-extract one arm against the other's manifest.")

        arrs = [np.load(p, mmap_mode="r") for _, p, _ in srcs]
        n, P = arrs[0].shape[0], arrs[0].shape[1]
        if any(z.shape[0] != n or z.shape[1] != P for z in arrs):
            raise SystemExit(f"[combined] {split}: shape mismatch "
                             f"{[z.shape for z in arrs]}")
        C = sum(z.shape[2] for z in arrs)

        dest = out / f"ffpp_{split}.npy"
        mm = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32,
                                       shape=(n, P, C))
        for i in range(0, n, CHUNK):
            cols, off = [], 0
            for (src, _, _), z in zip(srcs, arrs):
                m, s = stats[src]
                cols.append((np.asarray(z[i:i + CHUNK], np.float32) - m) / s)
                off += z.shape[2]
            mm[i:i + CHUNK] = np.concatenate(cols, axis=2)
        mm.flush()
        del mm
        shutil.copyfile(srcs[0][2], out / f"ffpp_{split}_index.json")
        print(f"[combined] {split}: ({n}, {P}, {C}) -> {dest.name} "
              f"({dest.stat().st_size / 1e9:.2f} GB)")
        written.append((split, n, P, C))

    if not written:
        raise SystemExit("[combined] nothing written")

    # `_feature_dims` reads out_dim from here whenever features.out_dim <= 0,
    # which is what this arm uses ("keep every coordinate").  Without it the tag
    # becomes ..._g8c0 and the circuit is built with zero channels.
    C = written[0][3]
    np.savez(out / "projector.npz",
             mean=np.zeros(C, np.float32),
             components=np.eye(C, dtype=np.float32),
             scale=np.ones(C, np.float32),
             explained=np.float32(1.0), n_patches=written[0][2],
             out_dim=C, whiten=False,
             per_patch_mean=np.zeros(0, np.float32),
             per_patch_std=np.zeros(0, np.float32))
    (out / "config.json").write_text(json.dumps({
        "backbone": "combined",
        "built_by": "scripts/build_combined_features.py",
        "sources": list(SOURCES),
        "channels_per_source": {src: int(len(stats[src][0])) for src in SOURCES},
        "standardized": "per channel, statistics from ffpp_train only",
        "splits": {s: {"n": n, "patches": P, "channels": c}
                   for s, n, P, c in written},
    }, indent=2))
    print(f"[combined] projector.npz out_dim={C}; wrote {out}")
    print(f"[combined] next: pcdf -c {a.config} fit-pc")


if __name__ == "__main__":
    main()
