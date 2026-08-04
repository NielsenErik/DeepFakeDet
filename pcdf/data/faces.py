"""
Video -> aligned face crops (+ landmarks, + pseudo ground-truth masks).

Design constraints that shaped this file:

* DISK.  The workstation has ~100 GB free and FF++ c23 alone is 18 GB of video.
  Crops are the only thing worth keeping (32 frames/video at 256² JPEG ≈ 3 GB
  for all of FF++), so the ingestion is crop-then-purge: `pcdf data ingest`
  can delete the source videos of a method once its crops are written.
* LANDMARKS ARE NOT OPTIONAL.  Self-blending (the SBI protocol that shapes the
  representation) needs a face-shaped mask, and the localization evaluation
  needs to know where the face is.  They are extracted once and cached.
* LOCALIZATION GROUND TRUTH.  This FF++ distribution ships no mask videos, but
  the four classic manipulations are frame-aligned re-renderings of their
  source video, so |fake_t − real_t| localizes the manipulation directly.  The
  masks produced here are therefore *derived*, and named `pseudo_mask` in every
  output so no result can silently claim official masks.

The detector backend is whichever of insightface (SCRFD, GPU) / mediapipe is
importable; both give a box and landmarks.  The choice is recorded in the shard
metadata because crop geometry is a real confound when comparing to published
numbers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── detector backends ───────────────────────────────────────────────────────

class FaceDetector:
    """Uniform interface: image (H,W,3) RGB uint8 -> (bbox, landmarks) or None."""

    def __init__(self, backend: str = "auto", device: str = "cuda", det_size: int = 640):
        self.backend = backend
        self.device = device
        self._impl = None
        if backend in ("auto", "insightface"):
            try:
                self._init_insightface(det_size)
                self.backend = "insightface"
                return
            except Exception as exc:                       # noqa: BLE001
                if backend == "insightface":
                    raise
                self._insight_error = str(exc)
        if backend in ("auto", "mediapipe"):
            self._init_mediapipe()
            self.backend = "mediapipe"
            return
        raise ValueError(f"unknown detector backend {backend!r}")

    def _init_insightface(self, det_size: int) -> None:
        from insightface.app import FaceAnalysis

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self.device == "cuda" else ["CPUExecutionProvider"])
        app = FaceAnalysis(name="buffalo_l", providers=providers,
                           allowed_modules=["detection", "landmark_2d_106"])
        app.prepare(ctx_id=0 if self.device == "cuda" else -1,
                    det_size=(det_size, det_size))
        self._impl = app

    MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                    "face_landmarker/float16/1/face_landmarker.task")

    def _init_mediapipe(self) -> None:
        """
        MediaPipe FaceLandmarker (478 points).  Cheap enough on one CPU core
        (~10 ms/frame) that ingestion parallelises cleanly across workers, and
        its dense landmarks are what the self-blending masks are built from.
        """
        import urllib.request

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        cache = Path.home() / ".cache" / "pcdf"
        cache.mkdir(parents=True, exist_ok=True)
        model = cache / "face_landmarker.task"
        if not model.exists():
            urllib.request.urlretrieve(self.MP_MODEL_URL, model)
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1)
        self._impl = vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp

    def __call__(self, img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.backend == "insightface":
            faces = self._impl.get(img[:, :, ::-1])        # insightface wants BGR
            if not faces:
                return None
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            lmk = getattr(f, "landmark_2d_106", None)
            if lmk is None:
                lmk = f.kps
            return np.asarray(f.bbox, np.float32), np.asarray(lmk, np.float32)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(img))
        res = self._impl.detect(mp_img)
        if not res.face_landmarks:
            return None
        h, w = img.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
        bbox = np.array([pts[:, 0].min(), pts[:, 1].min(),
                         pts[:, 0].max(), pts[:, 1].max()], np.float32)
        return bbox, pts


# ── cropping ────────────────────────────────────────────────────────────────

def crop_box(bbox: np.ndarray, shape: Tuple[int, int], margin: float = 1.3
             ) -> Tuple[int, int, int, int]:
    """
    Square crop around the detection, enlarged by `margin`.

    The margin is the single most consequential preprocessing choice in this
    field: too tight and the blending boundary — the artifact every FF++
    manipulation leaves — is cropped away; too loose and background dominates
    the feature.  1.3 follows the SBI / DeepfakeBench convention so numbers stay
    comparable.
    """
    h, w = shape
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * margin
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = int(round(x0 + side)), int(round(y0 + side))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def sample_frame_indices(n_total: int, n_want: int) -> List[int]:
    """Evenly spaced frames — never the first n, which are near-duplicates."""
    if n_total <= 0:
        return []
    if n_total <= n_want:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_want).round().astype(int).tolist()


@dataclass
class ExtractConfig:
    n_frames: int = 32
    size: int = 256
    margin: float = 1.3
    jpeg_quality: int = 95
    mask_size: int = 64
    mask_threshold: float = 0.06     # on [0,1] intensity difference
    detector: str = "auto"
    device: str = "cuda"


def extract_video(
    video: str | Path,
    out_dir: str | Path,
    cfg: ExtractConfig,
    detector: FaceDetector,
    reference_video: Optional[str | Path] = None,
) -> Dict:
    """
    Decode `video`, crop faces from `cfg.n_frames` evenly spaced frames, and
    write `<out_dir>/<idx>.jpg` + `landmarks.npy` (+ `mask_<idx>.png` when a
    frame-aligned `reference_video` real counterpart is given).

    Returns a per-video record with the number of frames actually written; a
    video where the detector fails on every frame yields 0 and is dropped from
    the manifest rather than silently contributing nothing.
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sample_frame_indices(n_total, cfg.n_frames)

    # SEEK to the sampled frames instead of decoding the whole file: the 32
    # wanted frames are spread over ~500, and decoding all of them (twice, when
    # a reference video is needed for the mask) was the ingestion bottleneck.
    rcap = None
    if reference_video is not None and Path(reference_video).exists():
        rcap = cv2.VideoCapture(str(reference_video))

    landmarks: List[np.ndarray] = []
    boxes: List[np.ndarray] = []
    kept: List[int] = []
    for i in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = detector(rgb)
        if det is None:
            continue
        bbox, lmk = det
        x0, y0, x1, y1 = crop_box(bbox, rgb.shape[:2], cfg.margin)
        if x1 - x0 < 32 or y1 - y0 < 32:
            continue
        crop = cv2.resize(rgb[y0:y1, x0:x1], (cfg.size, cfg.size),
                          interpolation=cv2.INTER_AREA)
        idx = len(kept)
        cv2.imwrite(str(out_dir / f"{idx:03d}.jpg"), crop[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        scale = cfg.size / max(x1 - x0, 1)          # landmarks -> crop frame
        landmarks.append((lmk - [x0, y0]) * scale)
        boxes.append(np.array([x0, y0, x1, y1], np.float32))
        kept.append(i)

        if rcap is not None:
            rcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok_r, ref = rcap.read()
            if ok_r and ref.shape == frame.shape:
                diff = cv2.absdiff(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                diff = diff.astype(np.float32) / 255.0
                diff = cv2.GaussianBlur(diff[y0:y1, x0:x1], (9, 9), 0)
                m = (diff > cfg.mask_threshold).astype(np.uint8) * 255
                m = cv2.resize(m, (cfg.mask_size, cfg.mask_size),
                               interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_dir / f"mask_{idx:03d}.png"), m)
    cap.release()
    if rcap is not None:
        rcap.release()
    ref_frames = {0: None} if rcap is not None else {}

    if landmarks:
        np.save(out_dir / "landmarks.npy", np.stack(landmarks).astype(np.float32))
        np.save(out_dir / "boxes.npy", np.stack(boxes).astype(np.float32))
    meta = {"video": str(video), "n_frames": len(kept), "frame_indices": kept,
            "detector": detector.backend, "size": cfg.size, "margin": cfg.margin,
            "has_pseudo_mask": bool(ref_frames)}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def crop_dir_for(root: str | Path, rec) -> Path:
    """Deterministic output directory for a manifest record."""
    return Path(root) / rec.dataset / rec.method / Path(rec.video).stem
