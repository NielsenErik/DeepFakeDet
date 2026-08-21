"""
Score OUR crops with the OFFICIAL SBI weights, end to end.

WHY.  `gap_waterfall.py` measured that our encoder reaches 0.8312 on FF++ where
published SBI reports 0.9964, and that **98% of the total gap is the encoder** —
the projection is not lossy and the circuit gives back almost nothing.  Every
recipe-level attempt to close that has failed (Findings 0, 3, 8).  But
Shiohara & Yamasaki released their trained weights, including one trained on
FF++ c23, so the encoder does not have to be reproduced.  It can be downloaded:

    python -m gdown 1X0-NYT8KPursLZZdxduRQju6E52hauV0 -O FFc23.tar   # c23
    python -m gdown 12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV -O FFraw.tar   # raw

THE QUESTION THIS ANSWERS, AND WHY IT IS WORTH 10 MINUTES.  Their crops come
from a dlib/RetinaFace 81-landmark pipeline; ours come from mediapipe with
margin 1.3.  If their encoder scores near 0.99 on OUR crops, the conventions are
compatible and adopting their encoder is only a re-extraction.  If it scores
far lower, the crop geometry itself matters and their preprocessing has to come
with the weights.  Either answer is worth knowing BEFORE spending hours
extracting features for 183k crops.

FOUR WAYS TO GET THIS SILENTLY WRONG, all verified against their code:

  1. It is `efficientnet_pytorch` (lukemelas), NOT timm.  The state dict is
     `net._conv_stem.weight`, `net._bn0.*`, `net._fc.*` — 706 tensors that share
     no names with timm's `tf_efficientnet_b4`.  Loading it into our
     `SbiEncoderExtractor` with `strict=False` would drop nearly every weight
     and NOT raise.
  2. `num_classes=2`, not 1: `_fc.weight` is (2, 1792).  The fake probability is
     `model(x).softmax(1)[:, 1]` (`src/inference/inference_dataset.py`).
  3. **No ImageNet normalization.**  They feed `.float()/255` and nothing else
     (`advprop=True` only chose the initialization, before fine-tuning).  Our
     own encoder DOES use mean/std, so the two are not interchangeable.
  4. RGB, 380x380, bilinear (`cv2.cvtColor(..., COLOR_BGR2RGB)` then
     `cv2.resize(..., dsize=(380,380))` with cv2's default INTER_LINEAR).

Video aggregation follows their protocol: max over faces within a frame, then
mean over frames.  We store one crop per frame, so that reduces to the mean; the
max-over-frames variant is reported alongside because it is a common alternative
and the difference is worth seeing rather than assuming.

Output: results/official_sbi_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdf.cli import load_config  # noqa: E402
from pcdf.device import resolve_device  # noqa: E402
from pcdf.stages import _feat_dir  # noqa: E402

INPUT_SIZE = 380


class OfficialSbiDetector(nn.Module):
    """`src/inference/model.py` verbatim, minus the pretrained download."""

    def __init__(self) -> None:
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        # from_name, not from_pretrained: every weight is about to be
        # overwritten, and from_pretrained would fetch 75 MB to throw away.
        self.net = EfficientNet.from_name("efficientnet-b4", num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_official(checkpoint: Path, device: str) -> OfficialSbiDetector:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = OfficialSbiDetector()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False is needed only for num_batches_tracked bookkeeping; anything
    # else missing means the architecture does not match and every number below
    # would be noise dressed up as a measurement.
    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing or unexpected:
        raise SystemExit(
            f"[official] state dict does not match the architecture\n"
            f"  missing:    {real_missing[:6]}{' ...' if len(real_missing) > 6 else ''}"
            f" ({len(real_missing)})\n"
            f"  unexpected: {list(unexpected)[:6]}"
            f"{' ...' if len(unexpected) > 6 else ''} ({len(unexpected)})")
    print(f"[official] loaded {len(sd)} tensors from {checkpoint.name} "
          f"(epoch {blob.get('epoch', '?')}), 0 unmatched")
    return model.eval().to(device)


def index_from_crops(root: Path, dataset: str, crops_dirname: str) -> dict:
    """
    Build a path/label/method/video index straight from an ingested manifest.

    Checking a published cross-dataset number does not need features, a
    projector or a circuit — only crops and labels — and requiring the feature
    pipeline first would put three more stages between us and the answer.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcdf.data.manifest import read_manifest

    man = root / "manifests" / f"{dataset}_ingested.csv"
    if not man.exists():
        raise SystemExit(
            f"[official] {man} not found — run\n"
            f"    pcdf -c <config> manifest --datasets {dataset}\n"
            f"    pcdf -c <config> ingest --dataset {dataset}")
    cols = {k: [] for k in ("path", "label", "method", "video")}
    n_missing = 0
    for r in read_manifest(man):
        d = root / crops_dirname / r.dataset / r.method / Path(r.video).stem
        if not d.exists():
            n_missing += 1
            continue
        stem = Path(r.video).stem
        for q in sorted(d.glob("[0-9]*.jpg")):
            cols["path"].append(str(q))
            cols["label"].append(r.label)
            cols["method"].append(r.method)
            cols["video"].append(f"{r.method}/{stem}")
    if n_missing:
        print(f"[official] {n_missing} manifest rows had no crop directory")
    if not cols["path"]:
        raise SystemExit(f"[official] no crops found under {root / crops_dirname / dataset}")
    return {k: np.array(v) for k, v in cols.items()}


_LMK_CACHE: dict = {}


def _landmarks_for(path: Path):
    """Dense landmarks for one stored crop, in CROP coordinates.

    `landmarks.npy` is (n_frames, 478, 2) indexed by the position of the file in
    the sorted listing — the same convention `collect_real_items` uses.
    """
    d = path.parent
    ent = _LMK_CACHE.get(d)
    if ent is None:
        lp = d / "landmarks.npy"
        order = {q.name: j for j, q in enumerate(sorted(d.glob("[0-9]*.jpg")))}
        ent = _LMK_CACHE[d] = (np.load(lp) if lp.exists() else None, order)
    lmks, order = ent
    j = order.get(path.name)
    if lmks is None or j is None or j >= len(lmks):
        return None
    return lmks[j]


def official_recrop(img: np.ndarray, lmk: np.ndarray) -> np.ndarray:
    """
    `crop_face(..., crop_by_bbox=True, margin=False, phase='test')` applied
    inside our stored crop.

    Their rule: take the detector bbox, add w/4 and h/4 on each side, then halve
    those margins for test -> bbox + w/8 and h/8.  The result is NOT square and
    is resized to 380x380 anyway, so faces arrive at the network STRETCHED.
    Ours is a square of side max(w,h)*1.3, resized without distortion.  This
    isolates that difference: same videos, same detector, same frames, only the
    crop rule changes.

    LIMIT OF THIS TEST.  We re-crop inside a stored 256px crop, so their tighter
    region is upsampled from perhaps 150x180 rather than taken at native
    resolution from the frame.  That can only cost accuracy, so this UNDERSTATES
    the benefit of their geometry; a positive result here is a lower bound.
    Their bbox also comes from RetinaFace where ours is the hull of mediapipe's
    dense landmarks, so the boxes are similar but not identical.
    """
    import cv2

    H, W = img.shape[:2]
    x0, y0 = float(lmk[:, 0].min()), float(lmk[:, 1].min())
    x1, y1 = float(lmk[:, 0].max()), float(lmk[:, 1].max())
    w, h = x1 - x0, y1 - y0
    if w <= 1 or h <= 1:
        return img
    xa = max(0, int(x0 - w / 8))
    ya = max(0, int(y0 - h / 8))
    xb = min(W, int(x1 + w / 8) + 1)
    yb = min(H, int(y1 + h / 8) + 1)
    if xb - xa < 8 or yb - ya < 8:
        return img
    return img[ya:yb, xa:xb]


class OurEncoderWrapper(nn.Module):
    """Our timm EfficientNet-B4 behind the same (B,3,H,W) in [0,1] interface.

    Ours was trained with ImageNet normalization (`SbiFrameDataset`) and has a
    single logit; theirs takes raw [0,1] and has two. The normalization is
    applied here so `score_crops` stays identical for both, and the single
    logit is mapped to the same "probability of fake" the official softmax
    gives.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.net((x - self.mean) / self.std).squeeze(1))
        return torch.stack([1 - p, p], dim=1)     # match softmax(1)[:,1]


def load_ours(cfg, checkpoint=None) -> nn.Module:
    import timm

    from pcdf.models.supervised import SbiConfig

    ck = Path(checkpoint or cfg["features"]["backbone_kwargs"]["checkpoint"])
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    scfg = SbiConfig(**{k: v for k, v in blob.get("cfg", {}).items()
                        if k in SbiConfig.__dataclass_fields__})
    net = timm.create_model(scfg.arch, pretrained=False, num_classes=1)
    net.load_state_dict(blob["model"])
    print(f"[ours] loaded {ck.name} (val AUC "
          f"{blob.get('best_val_auc_video', float('nan')):.4f})")
    return OurEncoderWrapper(net).eval().to(cfg["device"])


@torch.no_grad()
def score_crops(model: nn.Module, paths, device: str, batch: int = 64,
                crop_rule: str = "stored") -> np.ndarray:
    import cv2

    out, buf, n_recropped = [], [], 0
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            buf.append(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if crop_rule == "official":
                lmk = _landmarks_for(Path(p))
                if lmk is not None:
                    img = official_recrop(img, lmk)
                    n_recropped += 1
            buf.append(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)))  # INTER_LINEAR
        if len(buf) == batch or i == len(paths) - 1:
            x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).to(device)
            x = x.float().div_(255.0)                      # no mean/std — theirs
            out.append(model(x).softmax(1)[:, 1].float().cpu().numpy())
            buf = []
            if (i + 1) % 5000 < batch:
                print(f"[official] {i + 1}/{len(paths)}", flush=True)
    if crop_rule == "official":
        print(f"[official] re-cropped {n_recropped}/{len(paths)} "
              f"({100 * n_recropped / max(len(paths), 1):.1f}% had landmarks)")
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--set", "-s", action="append", dest="overrides", default=[])
    ap.add_argument("--checkpoint", default=None,
                    help="default: <root>/models/official_sbi/FFc23.tar")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default="ffpp",
                    help="ffpp reads the feature index (identical crops to the "
                         "probe and circuit); any other dataset is read "
                         "straight from its ingested manifest + crop dirs, so "
                         "no feature extraction is needed to check a published "
                         "cross-dataset number.")
    ap.add_argument("--crops-dir", default="crops")
    ap.add_argument("--our-encoder", action="store_true",
                    help="score with OUR checkpoint instead of the official "
                         "one. Different architecture (timm) AND different "
                         "input normalization (ImageNet mean/std, where theirs "
                         "is raw [0,1]) — the two are not interchangeable.")
    ap.add_argument("--crop-rule", choices=["stored", "official", "both"],
                    default="both",
                    help="stored: our square margin-1.3 crop as saved. "
                         "official: their bbox+w/8,h/8 non-square crop, "
                         "stretched to 380 — re-derived inside the stored crop.")
    ap.add_argument("--target", type=float, default=None,
                    help="published number to check against, e.g. 0.9287 for "
                         "Celeb-DF-v2 with the FF-c23 weights (repo table)")
    ap.add_argument("--ours", type=float, default=0.8312,
                    help="our encoder's measured FF++ video AUC (gap_waterfall)")
    ap.add_argument("--published", type=float, default=0.9964,
                    help="their reported FF++ c23 video AUC")
    a = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    from pcdf.eval.metrics import video_keys

    cfg = load_config(a.config, a.overrides)
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root = Path(cfg["root"])
    ckpt = Path(a.checkpoint or (root / "models" / "official_sbi" / "FFc23.tar"))
    if not ckpt.exists():
        raise SystemExit(f"[official] no checkpoint at {ckpt}; see the docstring "
                         f"for the gdown commands")

    if a.dataset == "ffpp":
        # the SAME index the probe and the circuit see, so this is a difference
        # in encoder and nothing else
        index = json.loads((_feat_dir(cfg) / "ffpp_test_index.json").read_text())
        index = {k: np.array(v) for k, v in index.items()}
    else:
        index = index_from_crops(root, a.dataset, a.crops_dir)
    if a.limit:
        rs = np.random.default_rng(0)
        sel = np.sort(rs.permutation(len(index["path"]))[:a.limit])
        index = {k: v[sel] for k, v in index.items()}
    y = index["label"].astype(int)
    vkey = video_keys(index) if a.dataset == "ffpp" else index["video"]
    print(f"[official] {len(y)} crops, {len(set(vkey.tolist()))} videos, "
          f"{int((y == 1).sum())} forged", flush=True)

    if a.our_encoder:
        model = load_ours(cfg, a.checkpoint)
    else:
        model = load_official(ckpt, cfg["device"])
    rules = ["stored", "official"] if a.crop_rule == "both" else [a.crop_rule]
    scores = {r: score_crops(model, index["path"], cfg["device"], a.batch, r)
              for r in rules}

    def evaluate(sc):
        r = {"auc_frame": float(roc_auc_score(y, sc))}
        for name, fn in (("mean", np.mean), ("max", np.max)):
            vids = sorted(set(vkey.tolist()))
            vs = np.array([fn(sc[vkey == v]) for v in vids])
            vy = np.array([int(y[vkey == v].max()) for v in vids])
            r[f"auc_video_{name}"] = float(roc_auc_score(vy, vs))
        per = {}
        for meth in sorted(set(index["method"][y == 1].tolist())):
            m = (y == 0) | (index["method"] == meth)
            vids = sorted(set(vkey[m].tolist()))
            vs = np.array([sc[m][vkey[m] == v].mean() for v in vids])
            vy = np.array([int(y[m][vkey[m] == v].max()) for v in vids])
            per[meth] = float(roc_auc_score(vy, vs))
        r["per_method"] = per
        r["recovered_fraction_of_gap"] = \
            (r["auc_video_mean"] - a.ours) / (a.published - a.ours)
        return r

    res = {"checkpoint": ("ours" if a.our_encoder else str(ckpt)),
           "dataset": a.dataset, "n_crops": int(len(y)),
           "ours_same_crops": a.ours, "published_reported": a.published,
           "by_crop_rule": {r: evaluate(sc) for r, sc in scores.items()}}

    print(f"\n{'crop rule':<12}{'frame':>9}{'video mean':>12}{'video max':>11}"
          f"{'% of gap':>10}")
    print("-" * 54)
    for r, v in res["by_crop_rule"].items():
        print(f"{r:<12}{v['auc_frame']:>9.4f}{v['auc_video_mean']:>12.4f}"
              f"{v['auc_video_max']:>11.4f}"
              f"{100 * v['recovered_fraction_of_gap']:>9.0f}%")
    print("-" * 54)
    print(f"{'our encoder':<12}{'':>9}{a.ours:>12.4f}")
    print(f"{'published':<12}{'':>9}{a.published:>12.4f}")

    print("\nper method (video, mean agg):")
    meths = sorted(next(iter(res["by_crop_rule"].values()))["per_method"])
    hdr = "".join(f"{r:>12}" for r in res["by_crop_rule"])
    print(f"{'':<16}{hdr}")
    for m in meths:
        row = "".join(f"{res['by_crop_rule'][r]['per_method'][m]:>12.4f}"
                      for r in res["by_crop_rule"])
        print(f"{m:<16}{row}")

    if a.target is not None:
        res["target"] = a.target
        got = max(v["auc_video_mean"] for v in res["by_crop_rule"].values())
        res["vs_target"] = got - a.target
        print(f"\npublished target      {a.target:.4f}")
        print(f"ours (best crop rule) {got:.4f}   ({got - a.target:+.4f})")
        if abs(got - a.target) <= 0.02:
            print("  VERDICT: reproduces the published number. The pipeline is "
                  "correct end to end -- crops, model, protocol.")
        elif got < a.target - 0.02:
            print("  VERDICT: does NOT reproduce. The defect is in our pipeline "
                  "and is now bounded by a published reference instead of "
                  "guessed at.")
        else:
            print("  VERDICT: above the published number -- check the test list "
                  "and the aggregation before believing it.")

    best = max(res["by_crop_rule"], key=lambda r: res["by_crop_rule"][r]["auc_video_mean"])
    v = res["by_crop_rule"][best]["auc_video_mean"]
    if "official" in res["by_crop_rule"] and "stored" in res["by_crop_rule"]:
        delta = (res["by_crop_rule"]["official"]["auc_video_mean"]
                 - res["by_crop_rule"]["stored"]["auc_video_mean"])
        res["crop_rule_delta"] = delta
        print(f"\ncrop rule alone: {delta:+.4f} "
              f"(their geometry vs ours, same weights, same frames)")
        if delta > 0.05:
            print("  VERDICT: crop geometry is a large part of the gap. Port "
                  "`crop_face` into ingest and re-extract at native resolution "
                  "-- this test upsamples from a 256px crop, so it is a LOWER "
                  "bound on the benefit.")
        elif delta > 0.02:
            print("  VERDICT: crop geometry matters but does not explain the "
                  "gap on its own. The detector (RetinaFace vs mediapipe) and "
                  "the evaluation protocol are the remaining suspects.")
        else:
            print("  VERDICT: crop geometry is NOT the explanation. Suspect the "
                  "detector or the published protocol; run their full inference "
                  "pipeline on our videos to separate those.")

    who = "ours" if a.our_encoder else "official"
    dest = root / "results" / f"sbi_eval_{who}_{a.dataset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2, default=float))
    print(f"[official] wrote {dest}")


if __name__ == "__main__":
    main()
