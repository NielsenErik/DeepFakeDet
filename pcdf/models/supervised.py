"""
The SBI detector: EfficientNet-B4 trained on FF++ REAL frames plus their
self-blends (Shiohara & Yamasaki, CVPR 2022).

It plays two roles in this study and they must not be confused:

  BASELINE   the strongest real-only competitor (published cross-dataset AUC
             ≈ 93 on Celeb-DF-v2).  If the circuit cannot approach this, the
             project's "strong performance" goal fails on the detection axis
             regardless of how elegant the exact queries are.
  ENCODER    its penultimate feature map is an artifact-tuned representation.
             The circuit fitted on that space (real frames only) is the
             configuration with a genuine chance at both SotA detection AND
             exact localization — the POC's negative result was that frozen
             semantic features carry no artifact signal at all.

Protocol note, stated because it is the thing reviewers check first: no real
forgery is used for training.  Real FF++ forgeries are used only for model
selection on the validation split — the same concession SBI itself makes — and
that is recorded in the checkpoint metadata.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..data.sbi import blend_ratio, self_blend
from ..device import autocast_kwargs, resolve_device


@dataclass
class SbiConfig:
    arch: str = "tf_efficientnet_b4_ns"
    image_size: int = 380
    batch_size: int = 24
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 8
    amp: bool = True
    seed: int = 0
    device: str = "auto"
    max_frames_per_video: int = 8


class SbiFrameDataset(Dataset):
    """
    Real crops in, (image, label) out: half the samples are self-blended into
    pseudo-fakes on the fly.  Blending happens per-epoch and per-sample, so the
    network never sees the same forgery twice — that is what makes SBI
    generalise across generators instead of memorising one.
    """

    def __init__(self, items: Sequence[Tuple[str, np.ndarray]], size: int,
                 train: bool = True, seed: int = 0):
        self.items = list(items)
        self.size = size
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, img: np.ndarray, rng) -> np.ndarray:
        import cv2

        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.3:
            q = int(rng.integers(40, 100))
            ok, enc = cv2.imencode(".jpg", img[:, :, ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
        if rng.random() < 0.3:
            f = rng.uniform(0.5, 1.0)
            h, w = img.shape[:2]
            img = cv2.resize(cv2.resize(img, (int(w * f), int(h * f))), (w, h))
        return np.ascontiguousarray(img)

    def __getitem__(self, i: int):
        import cv2

        path, lmk = self.items[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e3) % 100000))
        label = 0.0
        if self.train and rng.random() < 0.5 and lmk is not None:
            blended, mask = self_blend(img, lmk, rng)
            if 0.02 < blend_ratio(mask) < 0.9:      # skip degenerate blends
                img, label = blended, 1.0
        if self.train:
            img = self._augment(img, rng)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div_(255.)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std, torch.tensor(label)


def collect_real_items(cfg: Dict, split: str = "train",
                       max_per_video: int = 8) -> List[Tuple[str, np.ndarray]]:
    """(crop path, landmarks) for real videos of one split."""
    from ..data.manifest import read_manifest

    root = Path(cfg["root"])
    recs = [r for r in read_manifest(root / "manifests" / "ffpp_ingested.csv")
            if r.split == split and r.label == 0]
    items: List[Tuple[str, np.ndarray]] = []
    for r in recs:
        d = root / "crops" / r.dataset / r.method / Path(r.video).stem
        lmk_path = d / "landmarks.npy"
        lmks = np.load(lmk_path) if lmk_path.exists() else None
        jpgs = sorted(d.glob("[0-9]*.jpg"))[:max_per_video]
        for j, p in enumerate(jpgs):
            items.append((str(p), lmks[j] if lmks is not None and j < len(lmks) else None))
    return items


def collect_labeled_items(cfg: Dict, split: str, max_per_video: int = 4
                          ) -> Tuple[List[Tuple[str, None]], np.ndarray, np.ndarray]:
    """Real + fake crops of a split, for VALIDATION / testing only."""
    from ..data.manifest import read_manifest

    root = Path(cfg["root"])
    recs = [r for r in read_manifest(root / "manifests" / "ffpp_ingested.csv")
            if r.split == split]
    items, labels, videos = [], [], []
    for r in recs:
        d = root / "crops" / r.dataset / r.method / Path(r.video).stem
        for p in sorted(d.glob("[0-9]*.jpg"))[:max_per_video]:
            items.append((str(p), None))
            labels.append(r.label)
            videos.append(f"{r.method}/{Path(r.video).stem}")
    return items, np.array(labels), np.array(videos)


def train_sbi(cfg: Dict, scfg: Optional[SbiConfig] = None) -> Path:
    import timm
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    scfg = scfg or SbiConfig(device=cfg["device"], seed=cfg["seed"])
    scfg.device = resolve_device(scfg.device)
    torch.manual_seed(scfg.seed)
    root = Path(cfg["root"])

    train_items = collect_real_items(cfg, "train", scfg.max_frames_per_video)
    val_items, val_y, val_v = collect_labeled_items(cfg, "val", 4)
    print(f"[sbi] {len(train_items)} real training crops "
          f"(self-blended on the fly), {len(val_items)} validation crops")

    train_ds = SbiFrameDataset(train_items, scfg.image_size, True, scfg.seed)
    val_ds = SbiFrameDataset(val_items, scfg.image_size, False, scfg.seed)
    train_dl = DataLoader(train_ds, batch_size=scfg.batch_size, shuffle=True,
                          num_workers=scfg.workers, pin_memory=True, drop_last=True,
                          persistent_workers=scfg.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=scfg.batch_size * 2, shuffle=False,
                        num_workers=scfg.workers, pin_memory=True)

    net = timm.create_model(scfg.arch, pretrained=True, num_classes=1).to(scfg.device)
    opt = torch.optim.AdamW(net.parameters(), lr=scfg.lr, weight_decay=scfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=scfg.epochs)
    amp = autocast_kwargs(scfg.device, scfg.amp)
    scaler = torch.amp.GradScaler(amp["device_type"], enabled=amp["enabled"])
    out = root / "models" / "sbi_effnetb4.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    best = -1.0
    history = []
    for ep in range(scfg.epochs):
        net.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in train_dl:
            x, y = x.to(scfg.device, non_blocking=True), y.to(scfg.device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp["device_type"], enabled=amp["enabled"]):
                logit = net(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(x)
            n += len(x)
        sched.step()

        net.eval()
        preds = []
        with torch.no_grad(), torch.amp.autocast(amp["device_type"],
                                                 enabled=amp["enabled"]):
            for x, _ in val_dl:
                preds.append(torch.sigmoid(net(x.to(scfg.device)).squeeze(1)).float().cpu())
        p = torch.cat(preds).numpy()
        frame_auc = float(roc_auc_score(val_y, p))
        vs = np.array([p[val_v == v].mean() for v in np.unique(val_v)])
        vy = np.array([val_y[val_v == v][0] for v in np.unique(val_v)])
        video_auc = float(roc_auc_score(vy, vs))
        history.append({"epoch": ep, "loss": tot / max(n, 1),
                        "val_auc_frame": frame_auc, "val_auc_video": video_auc,
                        "sec": time.time() - t0})
        print(f"[sbi] epoch {ep:3d} loss {tot / max(n, 1):.4f} "
              f"val AUC frame {frame_auc:.4f} video {video_auc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if video_auc > best:
            best = video_auc
            torch.save({"model": net.state_dict(), "cfg": scfg.__dict__,
                        "best_val_auc_video": best, "history": history,
                        "protocol": "real frames + self-blends only; FF++ val "
                                    "forgeries used for model selection"}, out)
    (root / "models" / "sbi_history.json").write_text(json.dumps(history, indent=2))
    print(f"[sbi] best val video AUC {best:.4f} -> {out}")
    return out


@torch.no_grad()
def score_sbi(cfg: Dict, checkpoint: str | Path, items: Sequence[Tuple[str, None]],
              scfg: Optional[SbiConfig] = None) -> np.ndarray:
    """Frame-level fake probabilities from a trained SBI detector."""
    import timm
    from torch.utils.data import DataLoader

    scfg = scfg or SbiConfig(device=cfg["device"])
    scfg.device = resolve_device(scfg.device)
    amp = autocast_kwargs(scfg.device, scfg.amp)
    net = timm.create_model(scfg.arch, pretrained=False, num_classes=1)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["model"])
    net.eval().to(scfg.device)
    dl = DataLoader(SbiFrameDataset(items, scfg.image_size, False),
                    batch_size=scfg.batch_size * 2, num_workers=scfg.workers)
    out = []
    with torch.amp.autocast(amp["device_type"], enabled=amp["enabled"]):
        for x, _ in dl:
            out.append(torch.sigmoid(net(x.to(scfg.device)).squeeze(1)).float().cpu())
    return torch.cat(out).numpy()
