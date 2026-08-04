"""
Post-processing perturbations for the robustness protocol.

Spectral and noise-residual cues — the ones every low-level forensic method
relies on — are notoriously fragile to recompression and rescaling, and a
detector that only survives on pristine crops is useless in deployment.  These
transforms are applied to the CROPS at feature-extraction time (never to the
training data), so the models are frozen and only the input distribution moves.

The sweep is deliberately mild-to-severe: c40-like JPEG, half-resolution
resampling, mild blur, and additive noise all sit within what a social platform
does to an uploaded video.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def _jpeg(quality: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        import cv2

        ok, enc = cv2.imencode(".jpg", img[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1] if ok else img
    return f


def _resize(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        small = cv2.resize(img, (max(8, int(w * factor)), max(8, int(h * factor))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return f


def _blur(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(img: np.ndarray) -> np.ndarray:
        import cv2

        k = int(2 * round(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (k, k), sigma)
    return f


def _noise(sd: float) -> Callable[[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)

    def f(img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32) + rng.normal(0, sd, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    return f


PERTURBATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clean": lambda img: img,
    "jpeg70": _jpeg(70),
    "jpeg50": _jpeg(50),
    "jpeg30": _jpeg(30),
    "resize0.5": _resize(0.5),
    "resize0.25": _resize(0.25),
    "blur": _blur(1.0),
    "blur2": _blur(2.0),
    "noise5": _noise(5.0),
}


def get_perturbation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; have {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[name]
