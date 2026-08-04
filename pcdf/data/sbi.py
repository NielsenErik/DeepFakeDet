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

def match_source_pipeline(img: np.ndarray, rng: np.random.Generator,
                          quality_range: Tuple[int, int] = (88, 96)) -> np.ndarray:
    """
    Re-encode a freshly blended image the way the REAL crops were produced.

    This is not cosmetic.  Measured on FF++ c23 spectral-residual features: the
    exact two-circuit ratio separates real faces from our own self-blends at
    AUC 0.953, but only 0.554 from actual FF++ forgeries.  The construction is
    therefore sound and the remaining gap is a DOMAIN GAP between the
    pseudo-fakes and the real ones — and its most obvious cause is compression
    history.  An FF++ manipulation is rendered and then the whole video is
    re-encoded (c23); our blend was composited in float and used raw, so it
    carries fresh, un-quantized resampling traces that no real forgery has.
    In a residual/spectral representation that difference dominates.

    Passing the blend back through the same JPEG quantization the stored crops
    went through removes the asymmetry, so p_blend models "a blend that has
    been through the pipeline" rather than "a blend".
    """
    import cv2

    q = int(rng.integers(quality_range[0], quality_range[1] + 1))
    ok, enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]


def self_blend(
    img: np.ndarray,
    landmarks: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    post_compress: bool = True,
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
    out = np.clip(blended, 0, 255).astype(np.uint8)
    if post_compress:
        out = match_source_pipeline(out, rng)
    return out, mask


# ── multi-family pseudo-forgeries ───────────────────────────────────────────
#
# Measured problem this exists to solve.  A single self-blend family makes
# forgeries that deviate from real in ONE direction — rougher, because the
# recipe adds resampling and compression noise.  On FF++ that works for the
# neural manipulations (Deepfakes 0.68, FaceShifter 0.67) and INVERTS on the
# graphics-rendered ones (Face2Face 0.45, FaceSwap 0.36), which are *smoother*
# than real because a 3D render has no sensor noise at all.
#
# Training p_blend harder does not help: the discriminative loss reaches 0.0000
# and real-vs-own-blend AUC 0.9996 while FF++ stays at 0.827.  The pseudo-task
# is saturated; the gap is the pseudo-fake DISTRIBUTION.  (Independently argued
# in "The Alpha Blending Hypothesis", 2026: blend-trained detectors learn a
# compositing shortcut rather than forgery evidence.)
#
# The fix is diversity in both directions, and a circuit is the right model for
# it: p_blend becomes a MIXTURE over forgery families, which it can represent
# natively, and the mixture posterior is itself an interpretable output —
# "this looks like a render, not a neural swap".

def _render_like(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Graphics-render appearance: SMOOTHER than the camera, not rougher.
    Bilateral filtering keeps edges while removing the sensor-noise floor,
    which is what a rasterised 3D face looks like.
    """
    import cv2

    out = cv2.bilateralFilter(img, d=int(rng.integers(5, 10)),
                              sigmaColor=float(rng.uniform(40, 90)),
                              sigmaSpace=float(rng.uniform(40, 90)))
    if rng.random() < 0.5:                      # mild texture flattening
        out = cv2.GaussianBlur(out, (3, 3), float(rng.uniform(0.3, 0.8)))
    return out


def _overshoot(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Upsampling overshoot: the ringing a generator's decoder leaves."""
    import cv2

    h, w = img.shape[:2]
    f = rng.uniform(0.3, 0.6)
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return np.clip(up.astype(np.float32) * 1.05 - 0.05 * img.astype(np.float32),
                   0, 255).astype(np.uint8)


def _statistical_only(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Colour/contrast mismatch with NO geometric blending — isolates the
    'donor statistics differ from context' cue from the boundary cue."""
    return _color_jitter(img, rng)


PSEUDO_FAMILIES = {
    "blend": None,                 # the original SBI recipe (rougher)
    "render": _render_like,        # smoother — the graphics-forgery direction
    "overshoot": _overshoot,       # decoder ringing
    "statistical": _statistical_only,
}


def multi_family_blend(
    img: np.ndarray,
    landmarks: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    family: Optional[str] = None,
    post_compress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    One pseudo-forgery from a randomly chosen family.  Returns
    (image, mask, family_name) so the family label can be kept for analysis.

    Every family still blends through a landmark mask — what differs is how the
    donor region is transformed, and crucially in which DIRECTION it deviates
    from real.
    """
    import cv2

    rng = rng or np.random.default_rng()
    names = list(PSEUDO_FAMILIES)
    fam = family or names[int(rng.integers(len(names)))]

    if fam == "blend":
        out, mask = self_blend(img, landmarks, rng, post_compress=post_compress)
        return out, mask, fam

    src = PSEUDO_FAMILIES[fam](img.copy(), rng)
    mask = deform_mask(landmark_hull_mask(landmarks, img.shape[:2]), rng)
    src, mask = _random_affine(src, mask, rng)
    m = mask[..., None]
    out = np.clip(m * src.astype(np.float32) + (1.0 - m) * img.astype(np.float32),
                  0, 255).astype(np.uint8)
    if post_compress:
        out = match_source_pipeline(out, rng)
    return out, mask, fam


def blend_ratio(mask: np.ndarray) -> float:
    """Fraction of the image the manipulation covers — used to drop degenerate
    blends (empty or whole-image masks teach the classifier nothing)."""
    return float((mask > 0.1).mean())
