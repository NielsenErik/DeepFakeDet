"""
The gap decomposition: where, exactly, is the distance to a SotA detector lost?

Every stage is measured on the SAME crops — the script reads the `path` field
of the feature index and scores the encoder on exactly the images whose
projected features the probe and the circuit see, so the four numbers are
differences in METHOD and not in protocol, sampling or frame budget.  The
project's `0.860` for the encoder had no artefact behind it; this produces one.

    published SBI on FF++ c23        the target
      | encoder training
    our encoder, end to end          <- measured here
      | PCA projection to grid x out_dim
    supervised linear probe          <- pcdf probe
      | one-class density scoring
    circuit, NLL                     <- pcdf evaluate
      | likelihood-ratio scoring
    circuit, exact log-ratio         <- pcdf fit-ratio

Reading the result: whichever arrow is longest is where the project's remaining
effort belongs, and the arrows below the encoder are the only ones that are
about probabilistic circuits at all.
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
from pcdf.stages import _feat_dir, _tag  # noqa: E402


@torch.no_grad()
def encoder_scores(cfg, index, checkpoint: Path, batch: int = 64) -> np.ndarray:
    """Fake-probability from the SBI encoder for every crop in the index."""
    import cv2
    import timm

    from pcdf.models.supervised import SbiConfig

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scfg = SbiConfig(**{k: v for k, v in blob.get("cfg", {}).items()
                        if k in SbiConfig.__dataclass_fields__})
    dev = cfg["device"]
    net = timm.create_model(scfg.arch, pretrained=False, num_classes=1)
    net.load_state_dict(blob["model"])
    net.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    paths = list(index["path"])
    out, buf = [], []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            buf.append(torch.zeros(3, scfg.image_size, scfg.image_size))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (scfg.image_size,) * 2,
                             interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div_(255.)
            buf.append((x - mean) / std)
        if len(buf) == batch or i == len(paths) - 1:
            xb = torch.stack(buf).to(dev)
            with torch.amp.autocast("cuda", enabled=dev.startswith("cuda")):
                out.append(torch.sigmoid(net(xb).squeeze(1)).float().cpu().numpy())
            buf = []
            if (i + 1) % 5000 < batch:
                print(f"[waterfall] encoder {i + 1}/{len(paths)}", flush=True)
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--set", "-s", action="append", dest="overrides", default=[])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--published-sbi", type=float, default=0.9964,
                    help="published SBI FF++ c23 video AUC, the target")
    a = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    from pcdf.eval.metrics import video_keys, video_level

    cfg = load_config(a.config, a.overrides)
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root = Path(cfg["root"])
    tag = _tag(cfg)
    fd = _feat_dir(cfg)

    index = json.loads((fd / "ffpp_test_index.json").read_text())
    index = {k: np.array(v) for k, v in index.items()}
    y = index["label"].astype(int)
    vkey = video_keys(index)
    print(f"[waterfall] {len(y)} test crops, {len(set(vkey.tolist()))} videos",
          flush=True)

    ckpt = Path(a.checkpoint or (cfg["features"]["backbone_kwargs"]["checkpoint"]))
    s_enc = encoder_scores(cfg, index, ckpt)
    vs, vy = video_level(s_enc, vkey, y)
    enc = {"auc_video": float(roc_auc_score(vy, vs)),
           "auc_frame": float(roc_auc_score(y, s_enc)),
           "checkpoint": str(ckpt),
           "best_val_auc_video": float(torch.load(
               ckpt, map_location="cpu",
               weights_only=False).get("best_val_auc_video", float("nan")))}
    per_method = {}
    for meth in sorted(set(index["method"][y == 1].tolist())):
        m = (y == 0) | (index["method"] == meth)
        v, ly = video_level(s_enc[m], vkey[m], y[m])
        per_method[meth] = float(roc_auc_score(ly, v))
    enc["per_method"] = per_method
    print(json.dumps(enc, indent=2), flush=True)

    # pull the downstream stages from their own artefacts
    res_dir = root / "results" / tag
    def _read(name):
        p = res_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    probe, scores, ratio = _read("probe.json"), _read("scores.json"), _read("ratio.json")
    stages = [("published SBI (reported)", a.published_sbi),
              ("our encoder, end to end", enc["auc_video"])]
    if probe:
        stages.append((f"linear probe on projected features "
                       f"({probe['best']})",
                       probe["probes"][probe["best"]]["auc_video"]))
    if scores:
        stages.append(("circuit, one-class NLL",
                       scores["summary"]["PC"]["ffpp"]["auc_video"]))
    if ratio:
        stages.append(("circuit, exact log-ratio", ratio["best"]["auc_video"]))

    out = {"tag": tag, "encoder": enc, "waterfall": []}
    print(f"\n{'stage':<52}{'video AUC':>10}{'lost':>9}")
    print("-" * 71)
    prev = None
    for name, v in stages:
        drop = "" if prev is None else f"{prev - v:+.4f}"
        out["waterfall"].append({"stage": name, "auc_video": v,
                                 "lost_vs_previous": None if prev is None
                                 else prev - v})
        print(f"{name:<52}{v:>10.4f}{drop:>9}")
        prev = v
    total = stages[0][1] - stages[-1][1]
    out["total_gap"] = total
    out["encoder_share"] = (stages[0][1] - stages[1][1]) / total if total else None
    print("-" * 71)
    print(f"{'TOTAL':<52}{total:>10.4f}")
    print(f"\nencoder accounts for {100 * out['encoder_share']:.0f}% of the gap")

    p = res_dir / "gap_waterfall.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"[waterfall] wrote {p}")


if __name__ == "__main__":
    main()
