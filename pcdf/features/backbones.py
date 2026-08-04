"""
Frozen (and one trained) patch-level feature extractors.

The POC's decisive negative result was representational, not model-related:
globally pooled features put the PC, Mahalanobis and a full-covariance GMM at
*identical* chance accuracy on real face swaps, because an in-context swap
matches the reals' global statistics by construction (POC.md §6b).  The signal
is local — a blending boundary and an interior whose low-level statistics do
not match their surroundings — so every extractor here returns PATCH tokens,
shape (N, P, C), never a pooled vector.

Available streams:

  clip      CLIP ViT-L/14 patch tokens (the UnivFD substrate: CLIP's space
            demonstrably retains generator traces that ImageNet features drop)
  dinov2    DINOv2 ViT-L/14 patch tokens — self-supervised, texture-sensitive,
            the natural adversary for CLIP on this task
  sbi       patch grid from an EfficientNet-B4 trained under the SBI protocol
            (real frames + self-blends only); an *artifact-tuned* space, which
            is the representation-side answer to the POC failure
  spectral  per-patch noise-residual SPECTRUM and autocorrelation, after Corvi
            et al. (CVPRW 2023) — the domain in which synthetic content is
            genuinely atypical rather than over-typical; the corrective for the
            central negative result (see STATUS.md)
  srm       hand-built local forensic descriptor over a patch grid: SRM
            high-pass residual moments, radial DCT band energies and local
            color statistics — no learned component, so it isolates how much of
            any result is the representation versus the density model
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FeatureSpec:
    name: str
    grid: int          # patches per side
    dim: int           # channels per patch
    input_size: int


class PatchExtractor(nn.Module):
    """Base: (B,3,H,W) float in [0,1] -> (B, P, C) patch tokens."""

    spec: FeatureSpec

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


class ClipPatchExtractor(PatchExtractor):
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14",
                 layer: int = -2, input_size: int = 224):
        super().__init__()
        from transformers import CLIPVisionModel

        self.net = CLIPVisionModel.from_pretrained(model_id).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.layer = layer
        hid = self.net.config.hidden_size
        patch = self.net.config.patch_size
        grid = input_size // patch
        self.spec = FeatureSpec("clip", grid, hid, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.spec.input_size,) * 2, mode="bicubic",
                          align_corners=False)
        out = self.net(pixel_values=_normalize(x, CLIP_MEAN, CLIP_STD),
                       output_hidden_states=True)
        # the penultimate block keeps more low-level structure than the last,
        # which is heavily specialised for the contrastive objective
        h = out.hidden_states[self.layer]
        return h[:, 1:, :]                      # drop CLS -> (B, P, C)


class DinoV2PatchExtractor(PatchExtractor):
    def __init__(self, model_id: str = "facebook/dinov2-large", input_size: int = 224):
        super().__init__()
        from transformers import AutoModel

        self.net = AutoModel.from_pretrained(model_id).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        hid = self.net.config.hidden_size
        grid = input_size // self.net.config.patch_size
        self.spec = FeatureSpec("dinov2", grid, hid, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.spec.input_size,) * 2, mode="bicubic",
                          align_corners=False)
        out = self.net(pixel_values=_normalize(x, IMAGENET_MEAN, IMAGENET_STD))
        return out.last_hidden_state[:, 1:, :]


class SbiEncoderExtractor(PatchExtractor):
    """
    Patch grid from a detector trained under the SBI protocol.  The spatial
    feature map of the last block is average-pooled onto a `grid × grid` layout
    so the circuit's region graph can stay the same across representations.
    """

    def __init__(self, checkpoint: str, arch: str = "tf_efficientnet_b4.ns_jft_in1k",
                 grid: int = 8, input_size: int = 380):
        super().__init__()
        import timm

        self.net = timm.create_model(arch, pretrained=False, num_classes=1)
        state = torch.load(checkpoint, map_location="cpu")
        self.net.load_state_dict(state.get("model", state), strict=False)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        dim = self.net.num_features
        self.spec = FeatureSpec("sbi", grid, dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.spec.input_size,) * 2, mode="bicubic",
                          align_corners=False)
        fmap = self.net.forward_features(_normalize(x, IMAGENET_MEAN, IMAGENET_STD))
        if fmap.dim() == 3:                              # ViT-style (B, N, C)
            n = int(fmap.shape[1] ** 0.5)
            fmap = fmap.transpose(1, 2).reshape(fmap.shape[0], -1, n, n)
        fmap = F.adaptive_avg_pool2d(fmap, self.spec.grid)
        return fmap.flatten(2).transpose(1, 2)           # (B, P, C)


# ── hand-built local forensics (no learning at all) ────────────────────────

_SRM_KERNELS = torch.tensor([
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
], dtype=torch.float32) / torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1)


class SrmPatchExtractor(PatchExtractor):
    """
    Per-patch forensic descriptor: SRM residual moments (std / mean-abs /
    kurtosis per residual filter per channel), radial DCT band energies, and
    local color statistics.  Everything is differentiable-free and fixed, so a
    result obtained here cannot be attributed to a learned representation.
    """

    def __init__(self, grid: int = 8, input_size: int = 256, n_dct_bands: int = 6):
        super().__init__()
        self.register_buffer("srm", _SRM_KERNELS.unsqueeze(1).repeat(1, 1, 1, 1))
        self.n_dct_bands = n_dct_bands
        dim = 3 * 3 * 3 + n_dct_bands + 6        # residual moments + DCT + color
        self.spec = FeatureSpec("srm", grid, dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        g, S = self.spec.grid, self.spec.input_size
        x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        ps = S // g

        # SRM residuals per colour channel
        res = []
        for c in range(3):
            res.append(F.conv2d(x[:, c:c + 1], self.srm.to(x.dtype), padding=2))
        R = torch.cat(res, dim=1)                                  # (B, 9, S, S)
        Rp = R.unfold(2, ps, ps).unfold(3, ps, ps)                 # (B,9,g,g,ps,ps)
        Rp = Rp.reshape(B, 9, g * g, ps * ps)
        mu = Rp.mean(-1, keepdim=True)
        sd = Rp.std(-1)
        mabs = Rp.abs().mean(-1)
        kurt = (((Rp - mu) ** 4).mean(-1) / (sd ** 4 + 1e-8)).clamp(0, 100)
        moments = torch.cat([sd, mabs, kurt], dim=1)               # (B, 27, P)

        # radial DCT band energies per patch (spectral generator fingerprint)
        gray = x.mean(1, keepdim=True)
        gp = gray.unfold(2, ps, ps).unfold(3, ps, ps).reshape(B, g * g, ps, ps)
        spec = torch.fft.rfft2(gp, norm="ortho").abs().add(1e-6).log()
        fy = torch.arange(spec.shape[-2], device=x.device).view(-1, 1).float()
        fx = torch.arange(spec.shape[-1], device=x.device).view(1, -1).float()
        rad = (fy ** 2 + fx ** 2).sqrt()
        edges = torch.linspace(0, float(rad.max()) + 1e-6, self.n_dct_bands + 1,
                               device=x.device)
        bands = []
        for i in range(self.n_dct_bands):
            m = ((rad >= edges[i]) & (rad < edges[i + 1])).float()
            bands.append((spec * m).sum((-2, -1)) / m.sum().clamp_min(1.0))
        bands_t = torch.stack(bands, dim=1)                        # (B, nb, P)

        # local colour statistics
        xp = x.unfold(2, ps, ps).unfold(3, ps, ps).reshape(B, 3, g * g, ps * ps)
        color = torch.cat([xp.mean(-1), xp.std(-1)], dim=1)        # (B, 6, P)

        feats = torch.cat([moments, bands_t, color], dim=1)        # (B, C, P)
        return feats.transpose(1, 2).contiguous()                  # (B, P, C)


class SpectralResidualExtractor(PatchExtractor):
    """
    Per-patch noise-residual spectrum, following the analysis in "Intriguing
    properties of synthetic images: from GANs to diffusion models"
    (Corvi et al., CVPRW 2023).

    Why this exists: the CLIP arm established that in a semantic space fakes are
    MORE typical than reals, which breaks the low-density assumption the whole
    method rests on.  Corvi et al. give the domain where the opposite holds —
    the noise residual.  A real camera image carries a sensor-noise floor with
    broadly flat, aperiodic high-frequency content; synthetic content lacks it
    and instead carries PERIODIC traces of the generator's upsampling.  In that
    domain a forgery should be genuinely out-of-distribution, i.e. low-density,
    which is exactly what a density model needs.

    Three design choices follow from the paper, and each is a correction of the
    `srm` extractor in this same file:

    1. NO RADIAL AVERAGING.  Generator fingerprints are discrete peaks at
       specific (fx, fy) — a radial profile averages them away.  Here the 2D
       log-spectrum is pooled onto a coarse 2D grid that preserves peak
       LOCATION.
    2. AUTOCORRELATION AT SMALL LAGS.  Periodic upsampling traces appear as
       off-centre autocorrelation peaks of the residual; a few lags capture
       them compactly.
    3. NO RESAMPLING INSIDE THE EXTRACTOR.  Resizing attenuates precisely these
       cues, so the crop is used at its stored resolution when it already
       matches (`input_size`), and the high-frequency energy ratio — the
       paper's "synthetic images are too smooth" statistic — is measured
       explicitly.
    """

    def __init__(self, grid: int = 8, input_size: int = 256,
                 spec_cells: int = 6, max_lag: int = 3):
        super().__init__()
        self.register_buffer("srm", _SRM_KERNELS.unsqueeze(1))
        self.spec_cells = spec_cells
        self.max_lag = max_lag
        n_lags = (2 * max_lag + 1) ** 2 - 1          # autocorrelation, minus lag 0
        dim = spec_cells * spec_cells + n_lags + 3 * 3 + 2
        self.spec = FeatureSpec("spectral", grid, dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        g, S = self.spec.grid, self.spec.input_size
        if x.shape[-1] != S:                          # only if truly necessary
            x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        ps = S // g

        # ── noise residual: high-pass, the domain the paper works in ──────
        gray = x.mean(1, keepdim=True)
        res = F.conv2d(gray, self.srm.to(x.dtype), padding=2)      # (B,3,S,S)
        r = res[:, :1]                                             # main residual
        rp = r.unfold(2, ps, ps).unfold(3, ps, ps).reshape(B, g * g, ps, ps)

        # ── 1. 2D log-spectrum pooled on a grid (peaks keep their place) ──
        spec = torch.fft.fft2(rp, norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(-2, -1)).abs().add(1e-8).log()
        # normalise per patch so absolute contrast does not dominate
        spec = spec - spec.mean(dim=(-2, -1), keepdim=True)
        cells = F.adaptive_avg_pool2d(spec, self.spec_cells)       # (B,P,c,c)
        cells = cells.flatten(2)                                   # (B,P,c*c)

        # ── 2. autocorrelation of the residual at small lags ──────────────
        power = torch.fft.fft2(rp, norm="ortho").abs() ** 2
        ac = torch.fft.ifft2(power, norm="ortho").real
        ac = torch.fft.fftshift(ac, dim=(-2, -1))
        mid, L = ps // 2, self.max_lag
        ac = ac[..., mid - L:mid + L + 1, mid - L:mid + L + 1]
        ac = ac / ac.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        ac = ac.flatten(2)
        centre = (2 * L + 1) ** 2 // 2
        ac = torch.cat([ac[..., :centre], ac[..., centre + 1:]], dim=-1)

        # ── 3. residual moments + the high-frequency deficit statistic ────
        allres = res.unfold(2, ps, ps).unfold(3, ps, ps).reshape(B, 3, g * g, ps * ps)
        sd = allres.std(-1)
        mabs = allres.abs().mean(-1)
        kurt = (((allres - allres.mean(-1, keepdim=True)) ** 4).mean(-1)
                / (sd ** 4 + 1e-8)).clamp(0, 100)
        moments = torch.cat([sd, mabs, kurt], dim=1).transpose(1, 2)   # (B,P,9)

        # energy above vs below half-Nyquist: "synthetic images are too smooth"
        n = spec.shape[-1]
        q = n // 4
        hi = spec[..., :q, :].mean((-2, -1)) + spec[..., -q:, :].mean((-2, -1))
        lo = spec[..., q:-q, :].mean((-2, -1))
        ratio = torch.stack([hi, hi - lo], dim=-1)                     # (B,P,2)

        return torch.cat([cells, ac, moments, ratio], dim=-1)          # (B,P,C)


def build_extractor(name: str, device: str = "cuda", **kwargs) -> PatchExtractor:
    ex: PatchExtractor
    if name == "clip":
        ex = ClipPatchExtractor(**kwargs)
    elif name == "dinov2":
        ex = DinoV2PatchExtractor(**kwargs)
    elif name == "sbi":
        ex = SbiEncoderExtractor(**kwargs)
    elif name == "srm":
        ex = SrmPatchExtractor(**kwargs)
    elif name == "spectral":
        ex = SpectralResidualExtractor(**kwargs)
    else:
        raise KeyError(f"unknown extractor {name!r}")
    return ex.to(device).eval()
