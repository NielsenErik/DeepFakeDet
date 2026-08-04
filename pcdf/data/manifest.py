"""
Dataset manifests — the single source of truth about what is trained on and
what is evaluated on.

A manifest is a CSV with one row per *video* (frames are derived), carrying:

    dataset   ffpp | celebdf | dfdcp | dfd | df40 | standin
    method    real | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
              FaceShifter | DeepFakeDetection | <df40 method> ...
    label     0 = real, 1 = fake
    split     train | val | test
    video     path to the source video
    ident     identity / source-video group used for split disjointness
    n_frames  frames sampled from it (filled in by the face extractor)

Two rules this file exists to enforce, because breaking either invalidates
every number downstream:

1. FF++ splits are IDENTITY-disjoint and are the official ones (720/140/140
   video pairs from the FaceForensics repo).  A fake video derived from source
   pair (target, source) belongs to the split of that pair — never to the split
   of the frame it happens to resemble.
2. The headline protocol never lets a real fake into training.  `split=train`
   rows with `label=1` are only produced when a config explicitly asks for a
   supervised baseline, and the PC / SBI paths assert their absence.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
                "FaceShifter", "DeepFakeDetection")

FIELDS = ("dataset", "method", "label", "split", "video", "ident", "n_frames")


@dataclass
class VideoRecord:
    dataset: str
    method: str
    label: int
    split: str
    video: str
    ident: str
    n_frames: int = 0

    def as_row(self) -> Dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


def write_manifest(records: Sequence[VideoRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in records:
            w.writerow(r.as_row())
    return path


def read_manifest(path: str | Path) -> List[VideoRecord]:
    out: List[VideoRecord] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(VideoRecord(
                dataset=row["dataset"], method=row["method"],
                label=int(row["label"]), split=row["split"],
                video=row["video"], ident=row["ident"],
                n_frames=int(row.get("n_frames") or 0)))
    return out


def assert_no_fakes_in_train(records: Iterable[VideoRecord]) -> None:
    bad = [r for r in records if r.split == "train" and r.label == 1]
    if bad:
        raise AssertionError(
            f"{len(bad)} fake videos are in the TRAIN split — the real-only "
            f"protocol is violated (first: {bad[0].video})")


def assert_identity_disjoint(records: Iterable[VideoRecord]) -> None:
    by_split: Dict[str, set] = {}
    for r in records:
        by_split.setdefault(r.split, set()).add(r.ident)
    splits = sorted(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise AssertionError(
                    f"identity leakage between {a} and {b}: "
                    f"{len(overlap)} shared identities, e.g. {sorted(overlap)[:5]}")


# ── FaceForensics++ ─────────────────────────────────────────────────────────

def _ffpp_split_map(split_dir: str | Path) -> Dict[str, str]:
    """
    Official FF++ splits: each entry is a (target, source) pair of video ids.
    Both ids of a pair are in the same split, and every fake named
    `<target>_<source>` inherits it.
    """
    split_dir = Path(split_dir)
    mapping: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        pairs = json.loads((split_dir / f"ffpp_split_{split}.json").read_text())
        for a, b in pairs:
            mapping[str(a)] = split
            mapping[str(b)] = split
    return mapping


def build_ffpp_manifest(
    root: str | Path,
    split_dir: str | Path,
    methods: Sequence[str] = FFPP_METHODS,
    include_dfd: bool = False,
) -> List[VideoRecord]:
    """
    Manifest for the `FaceForensics++_C23/{real,fake/<Method>}` layout.

    Real videos are `real/<id>.mp4`; fakes are `<target>_<source>.mp4`, whose
    identity group is the TARGET id — the same identity the corresponding real
    video belongs to, which is what keeps the splits identity-disjoint across
    the real/fake boundary.

    DeepFakeDetection (the DFD actor subset) is excluded by default: its real
    counterparts are not in this distribution, so including its fakes would
    compare fakes of one source population against reals of another, and any
    detector would win for the wrong reason.  `include_dfd=True` marks them
    `split=test` for use as an explicitly caveated cross-source probe.
    """
    root = Path(root)
    smap = _ffpp_split_map(split_dir)
    records: List[VideoRecord] = []

    for vid in sorted((root / "real").glob("*.mp4")):
        ident = vid.stem
        split = smap.get(ident)
        if split is None:
            continue
        records.append(VideoRecord("ffpp", "real", 0, split, str(vid), ident))

    for method in methods:
        mdir = root / "fake" / method
        if not mdir.exists():
            continue
        if method == "DeepFakeDetection":
            if not include_dfd:
                continue
            for vid in sorted(mdir.glob("*.mp4")):
                actor = vid.stem.split("_")[0]
                records.append(VideoRecord("ffpp", method, 1, "test", str(vid),
                                           f"dfd_{actor}"))
            continue
        for vid in sorted(mdir.glob("*.mp4")):
            parts = vid.stem.split("_")
            target = parts[0]
            split = smap.get(target)
            if split is None:
                continue
            records.append(VideoRecord("ffpp", method, 1, split, str(vid), target))

    return records


# ── Celeb-DF v2 ─────────────────────────────────────────────────────────────

def build_celebdf_manifest(root: str | Path) -> List[VideoRecord]:
    """
    Official Celeb-DF-v2 layout: `Celeb-real/`, `YouTube-real/`,
    `Celeb-synthesis/` plus `List_of_testing_videos.txt`.  The standard
    protocol evaluates on that test list only, so everything else is dropped
    rather than silently used.
    """
    root = Path(root)
    test_list = root / "List_of_testing_videos.txt"
    if not test_list.exists():
        raise FileNotFoundError(
            f"{test_list} not found — Celeb-DF-v2 must ship its official test "
            f"list; without it the numbers are not comparable to the literature")
    records: List[VideoRecord] = []
    for line in test_list.read_text().strip().splitlines():
        label_str, rel = line.split()
        # the file uses 1 = real, 0 = fake — inverted w.r.t. our convention
        label = 0 if int(label_str) == 1 else 1
        path = root / rel
        ident = Path(rel).stem.split("_")[0]
        method = "real" if label == 0 else "Celeb-synthesis"
        records.append(VideoRecord("celebdf", method, label, "test", str(path), ident))
    return records


# ── DF40 (diffusion / modern generators) ────────────────────────────────────

def build_df40_manifest(root: str | Path, methods: Optional[Sequence[str]] = None,
                        split: str = "test") -> List[VideoRecord]:
    """
    DF40 mirrors are laid out as `<method>/<video_id>/*.png|mp4`.  Everything
    is test-only here: DF40 exists in this project to answer "does a detector
    trained on 2019-era FF++ forgeries survive modern generators", so training
    on it would destroy the question.
    """
    root = Path(root)
    records: List[VideoRecord] = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if methods and mdir.name not in methods:
            continue
        for vid in sorted(list(mdir.glob("*.mp4")) + [d for d in mdir.iterdir() if d.is_dir()]):
            records.append(VideoRecord("df40", mdir.name, 1, split, str(vid), vid.stem))
    return records


def summarize(records: Sequence[VideoRecord]) -> str:
    counts: Dict[tuple, int] = {}
    for r in records:
        counts[(r.dataset, r.split, r.method, r.label)] = \
            counts.get((r.dataset, r.split, r.method, r.label), 0) + 1
    lines = [f"{'dataset':<10} {'split':<6} {'method':<20} {'label':<6} {'videos':>7}"]
    for k in sorted(counts):
        lines.append(f"{k[0]:<10} {k[1]:<6} {k[2]:<20} {k[3]:<6} {counts[k]:>7}")
    lines.append(f"{'TOTAL':<10} {'':<6} {'':<20} {'':<6} {len(records):>7}")
    return "\n".join(lines)
