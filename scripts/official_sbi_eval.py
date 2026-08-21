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


@torch.no_grad()
def score_crops(model: nn.Module, paths, device: str, batch: int = 64) -> np.ndarray:
    import cv2

    out, buf = [], []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            buf.append(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            buf.append(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)))  # INTER_LINEAR
        if len(buf) == batch or i == len(paths) - 1:
            x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).to(device)
            x = x.float().div_(255.0)                      # no mean/std — theirs
            out.append(model(x).softmax(1)[:, 1].float().cpu().numpy())
            buf = []
            if (i + 1) % 5000 < batch:
                print(f"[official] {i + 1}/{len(paths)}", flush=True)
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--set", "-s", action="append", dest="overrides", default=[])
    ap.add_argument("--checkpoint", default=None,
                    help="default: <root>/models/official_sbi/FFc23.tar")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
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

    # the SAME index the probe and the circuit see, so this is a difference in
    # encoder and nothing else
    index = json.loads((_feat_dir(cfg) / "ffpp_test_index.json").read_text())
    index = {k: np.array(v) for k, v in index.items()}
    if a.limit:
        rs = np.random.default_rng(0)
        sel = np.sort(rs.permutation(len(index["path"]))[:a.limit])
        index = {k: v[sel] for k, v in index.items()}
    y = index["label"].astype(int)
    vkey = video_keys(index)
    print(f"[official] {len(y)} crops, {len(set(vkey.tolist()))} videos, "
          f"{int((y == 1).sum())} forged", flush=True)

    model = load_official(ckpt, cfg["device"])
    s = score_crops(model, index["path"], cfg["device"], a.batch)

    def agg(fn):
        vids = sorted(set(vkey.tolist()))
        vs = np.array([fn(s[vkey == v]) for v in vids])
        vy = np.array([int(y[vkey == v].max()) for v in vids])
        return vs, vy

    res = {"checkpoint": str(ckpt), "n_crops": int(len(y)),
           "auc_frame": float(roc_auc_score(y, s))}
    for name, fn in (("mean", np.mean), ("max", np.max)):
        vs, vy = agg(fn)
        res[f"auc_video_{name}"] = float(roc_auc_score(vy, vs))
    per = {}
    for meth in sorted(set(index["method"][y == 1].tolist())):
        m = (y == 0) | (index["method"] == meth)
        vids = sorted(set(vkey[m].tolist()))
        vs = np.array([s[m][vkey[m] == v].mean() for v in vids])
        vy = np.array([int(y[m][vkey[m] == v].max()) for v in vids])
        per[meth] = float(roc_auc_score(vy, vs))
    res["per_method"] = per
    res["ours_same_crops"] = a.ours
    res["published_reported"] = a.published

    v = res["auc_video_mean"]
    res["recovered_fraction_of_gap"] = (v - a.ours) / (a.published - a.ours)
    print(f"\n  frame AUC                 {res['auc_frame']:.4f}")
    print(f"  video AUC (mean, theirs)  {v:.4f}")
    print(f"  video AUC (max)           {res['auc_video_max']:.4f}")
    print("\n  per method (mean agg):")
    for k, val in per.items():
        print(f"    {k:<16}{val:.4f}")
    print(f"\n  our encoder, same crops   {a.ours:.4f}")
    print(f"  their reported            {a.published:.4f}")
    print(f"  -> recovers {100 * res['recovered_fraction_of_gap']:.0f}% of the "
          f"encoder gap on OUR crops")
    if v >= 0.97:
        print("  VERDICT: crop conventions are compatible — adopting their "
              "encoder is a re-extraction, nothing more.")
    elif v >= a.ours + 0.05:
        print("  VERDICT: clearly better than ours but short of published. "
              "Their crop geometry matters; port `crop_face` before extracting.")
    else:
        print("  VERDICT: no better than ours on these crops. The gap is in the "
              "preprocessing, not the weights — port their face pipeline first.")

    dest = root / "results" / "official_sbi_eval.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2, default=float))
    print(f"[official] wrote {dest}")


if __name__ == "__main__":
    main()
