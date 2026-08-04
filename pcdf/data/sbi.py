"""
Self-Blended Images (Shiohara & Yamasaki, CVPR 2022) — reimplemented.

Why this is in a probabilistic-circuit project at all: SBI is the strongest
*real-only* deepfake detector line (no real forgery is ever seen in training,
which is exactly the protocol the PC needs), and its trick is representational,
not architectural.  It manufactures the two artifacts every face-swap shares —
a blending boundary and a statistical mismatch between the swapped interior and
its context — from a single real face.  That gives:

  1. a strong supervised baseline to be measured against, and
  2. an artifact-sensitive encoder whose features the circuit can model, which
     is the answer to the POC's core negative result (frozen semantic features
     put every density model at chance, POC.md §6b).

The pipeline per real image, following the paper:

    source, target  <- two differently perturbed copies of the SAME face
                       (color / frequency / resolution jitter; the paper's
                       "source-target generator")
    mask            <- convex hull of the landmarks, deformed, eroded/dilated,
                       blurred with a random kernel
    blended         <- mask * affine(source) + (1 - mask) * target

Both the image and its mask are returned: the mask is the training signal for
any localization head and the ground truth for the self-consistency checks in
`pcdf.eval.localization`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ── mask construction ───────────────────────────────────────────────────────

def landmark_hull_mask(landmarks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Convex hull of the face landmarks as a float mask in [0, 1]."""
    import cv2

    h, w = shape
    mask = np.zeros((h, w), np.float32)
    pts = landmarks.astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask


def deform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomize the mask so the classifier cannot memorise one boundary shape:
    elastic-ish warp by a coarse random displacement field, random
    erosion/dilation, random blur, random global alpha.
    """
    import cv2

    h, w = mask.shape
    # coarse random displacement, upsampled -> smooth elastic deformation
    grid = 8
    dx = cv2.resize(rng.normal(0, 1, (grid, grid)).astype(np.float32), (w, h)) * (w * 0.02)
    dy = cv2.resize(rng.normal(0, 1, (grid, grid)).astype(np.float32), (w, h)) * (h * 0.02)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mask = cv2.remap(mask, xx + dx, yy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    k = int(rng.integers(1, max(2, int(min(h, w) * 0.05))))
    kernel = np.ones((k, k), np.uint8)
    if rng.random() < 0.5:
        mask = cv2.erode(mask, kernel)
    else:
        mask = cv2.dilate(mask, kernel)

    blur = int(rng.integers(3, max(4, int(min(h, w) * 0.12)))) | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask * rng.uniform(0.25, 1.0), 0.0, 1.0)


# ── source / target perturbations ───────────────────────────────────────────

def _color_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2

    out = img.astype(np.float32)
    out = out * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12)          # brightness
    out = (out - out.mean()) * rng.uniform(0.9, 1.1) + out.mean()     # contrast
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-6, 7))) % 180      # hue
    hsv[..., 1] = np.clip(hsv[..., 1] + int(rng.integers(-20, 21)), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def _resolution_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Down-up resample: the generator/warping fingerprint of every face swap."""
    import cv2

    h, w = img.shape[:2]
    f = rng.uniform(0.4, 0.9)
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _jpeg_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2

    q = int(rng.integers(50, 100))
    ok, enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]


def source_target_pair(img: np.ndarray, rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two perturbed copies of one real face.  The source gets the artifacts a
    generator would leave (resolution loss, color shift, compression); the
    target stays closer to the original, so their blend has exactly the
    inconsistency a swap has.
    """
    src, tgt = img.copy(), img.copy()
    if rng.random() < 0.9:
        src = _color_jitter(src, rng)
    if rng.random() < 0.8:
        src = _resolution_jitter(src, rng)
    if rng.random() < 0.5:
        src = _jpeg_jitter(src, rng)
    if rng.random() < 0.3:
        tgt = _color_jitter(tgt, rng)
    if rng.random() < 0.5:
        src, tgt = tgt, src
    return src, tgt


def _random_affine(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Small misalignment of the donor face — the geometric half of the cue."""
    import cv2

    h, w = img.shape[:2]
    tx, ty = rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h
    scale = rng.uniform(0.95, 1.05)
    ang = rng.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return (cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT),
            cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0))


# ── the operator ────────────────────────────────────────────────────────────

def self_blend(
    img: np.ndarray,
    landmarks: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-blended pseudo-fake from a single real face.

    img:       (H, W, 3) uint8 RGB
    landmarks: (L, 2) in image coordinates
    returns:   (blended uint8, mask float32 in [0,1])

    The mask is the *blending* mask, i.e. the ground-truth manipulated region —
    which makes it directly usable as the target of a localization evaluation.
    """
    rng = rng or np.random.default_rng()
    src, tgt = source_target_pair(img, rng)
    mask = deform_mask(landmark_hull_mask(landmarks, img.shape[:2]), rng)
    src, mask = _random_affine(src, mask, rng)
    m = mask[..., None]
    blended = (m * src.astype(np.float32) + (1.0 - m) * tgt.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8), mask


def blend_ratio(mask: np.ndarray) -> float:
    """Fraction of the image the manipulation covers — used to drop degenerate
    blends (empty or whole-image masks teach the classifier nothing)."""
    return float((mask > 0.1).mean())
