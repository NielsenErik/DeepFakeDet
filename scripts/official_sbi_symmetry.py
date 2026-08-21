"""
Does the leak of Finding 1 exist in the OFFICIAL SBI implementation?

F1 says: classifying real vs self-blend using only 8x8 blocks entirely outside
the dilated blending mask reaches AUC 0.937 under our recipe, while the same
features on real FF++ forgeries score 0.477 (chance).  Everything about that is
a statement about `pcdf.data.sbi.self_blend` — OUR reimplementation.  Before F1
can be a claim about the SBI-derived line (SBI, FSBI, BlenD, ...) it has to be
checked against Shiohara & Yamasaki's released code.

    git clone https://github.com/mapooon/SelfBlendedImages

THE TWO PLACES A GLOBAL ASYMMETRY COULD ENTER, in SBI_Dataset.__getitem__:

  1. self_blending():  img_blended = mask*source + (1-mask)*target, and the
     array returned as the REAL is `target` ITSELF.  Both branches of the
     p=0.5 coin perturb `img` BEFORE blending and return that same perturbed
     `img` as the real, so the composite shares the real's background exactly.
     Ours instead composites onto `tgt`, a SEPARATELY perturbed copy, while
     training compares against the raw `img` -- deviation 1.

  2. self.transforms: alb.Compose(..., additional_targets={'image1':'image'})
     called as transforms(image=img_f, image1=img_r).  Albumentations replays
     one sampled parameter set across all targets, so ImageCompression q40-100
     hits the fake and the real at the SAME quality.  Ours applies
     match_source_pipeline (JPEG q88-96) to the blend only -- deviation 2.

So the prediction is that the official recipe is periphery-symmetric and would
score the null, 0.500, in `shortcut_audit.py --tests T1`.  This script measures
it rather than asserting it.

  A  parameter sharing   -- identical inputs to both targets must give
                            identical outputs, or claim 2 is wrong
  B  periphery symmetry  -- with a blend in the centre, do the fake and the
                            real differ outside the mask?
  C  spatial extent      -- if they do, is it a global cue or one JPEG block
                            at the boundary?  periphery_blocks() dilates by
                            24 px, so anything inside 8 px is already excluded
                            from the audit.

Needs albumentations, which is NOT in the project env (it is an SBI dependency,
not ours):  python -m venv /tmp/albenv && /tmp/albenv/bin/pip install albumentations

Output: results/official_sbi_symmetry.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as alb
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(f"{e}\nSee the docstring: this needs albumentations.")

H = W = 256
BOX = slice(80, 176)          # stand-in for the face hull, 8-px aligned
N_TRIALS = 200
DILATE_PX = 24                # must match periphery_blocks() in shortcut_audit.py


def official_transforms():
    """Verbatim SBI_Dataset.get_transforms(), modulo the albumentations 2.x
    rename of quality_lower/quality_upper -> quality_range."""
    try:
        comp = alb.ImageCompression(quality_range=(40, 100), p=0.5)
    except TypeError:                                    # albumentations 1.x
        comp = alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5)
    return alb.Compose([
        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3),
                               sat_shift_limit=(-0.3, 0.3),
                               val_shift_limit=(-0.3, 0.3), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3),
                                     contrast_limit=(-0.3, 0.3), p=0.3),
        comp,
    ], additional_targets={"image1": "image"}, p=1.0)


def _pair(rng):
    """One (real, official-fake, mask) triple.

    `img_f` is built the way dynamic_blend builds it -- compositing onto `img`
    itself -- which is the whole point: the real returned by SBI is that same
    `img`, so step 1 cannot introduce a periphery difference.
    """
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    source = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W, 1), np.float32)
    mask[BOX, BOX] = 1.0
    img_f = (mask * source + (1 - mask) * img).astype(np.uint8)
    return img, img_f, mask


def test_A_parameter_sharing(rng) -> dict:
    """Feed the SAME image as both targets; any difference means the sampled
    parameters are not shared and deviation 2 does not hold."""
    T = official_transforms()
    worst = 0
    for _ in range(N_TRIALS):
        img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        out = T(image=img, image1=img.copy())
        worst = max(worst, int(np.abs(out["image"].astype(int)
                                      - out["image1"].astype(int)).max()))
    return {"max_abs_diff": worst,
            "shared": worst == 0,
            "note": ("Identical inputs to `image` and `image1`. 0 means "
                     "albumentations replays one parameter set across targets, "
                     "so the JPEG quality is the same for fake and real.")}


def test_B_periphery(rng) -> dict:
    """Official recipe end to end: is the periphery bit-identical?"""
    T = official_transforms()
    worst_raw = worst_final = 0
    fired = 0
    for _ in range(N_TRIALS):
        img, img_f, mask = _pair(rng)
        periphery = mask[..., 0] == 0
        worst_raw = max(worst_raw, int(np.abs(img_f[periphery].astype(int)
                                              - img[periphery].astype(int)).max()))
        out = T(image=img_f, image1=img)
        f2, r2 = out["image"], out["image1"]
        if not np.array_equal(f2, img_f):
            fired += 1
        worst_final = max(worst_final, int(np.abs(f2[periphery].astype(int)
                                                  - r2[periphery].astype(int)).max()))
    return {"max_periphery_diff_after_blend": worst_raw,
            "max_periphery_diff_after_transforms": worst_final,
            "trials_where_a_transform_fired": fired,
            "trials": N_TRIALS,
            "note": ("after_blend=0 confirms dynamic_blend composites onto the "
                     "array returned as the real. A nonzero value after the "
                     "shared transforms is JPEG non-locality, not asymmetry -- "
                     "test C localizes it.")}


def test_C_spatial_extent(rng) -> dict:
    """Where does any residual difference live, relative to the mask?"""
    try:
        comp = alb.ImageCompression(quality_range=(40, 100), p=1)
    except TypeError:
        comp = alb.ImageCompression(quality_lower=40, quality_upper=100, p=1)
    T = alb.Compose([comp], additional_targets={"image1": "image"}, p=1.0)

    bands = [(0, 8), (8, 16), (16, 24), (24, 48), (48, 10**9)]
    acc = {b: [] for b in bands}
    for _ in range(N_TRIALS):
        img, img_f, mask = _pair(rng)
        m0 = (mask[..., 0] > 0).astype(np.uint8)
        dist = cv2.distanceTransform(1 - m0, cv2.DIST_L2, 5)
        out = T(image=img_f, image1=img)
        d = np.abs(out["image"].astype(int) - out["image1"].astype(int)).max(axis=2)
        for lo, hi in bands:
            sel = (dist >= lo) & (dist < hi) & (m0 == 0)
            if sel.any():
                acc[(lo, hi)].append((d[sel].max(), (d[sel] > 0).mean()))

    out = {}
    for (lo, hi), v in acc.items():
        a = np.array(v)
        lab = f"{lo}-{'inf' if hi > 10**8 else hi}px"
        out[lab] = {"max_abs_diff": float(a[:, 0].max()),
                    "frac_pixels_changed": float(a[:, 1].mean())}
    out["note"] = (f"Compression forced on (p=1) to bound the worst case. "
                   f"periphery_blocks() dilates the mask by {DILATE_PX} px, so "
                   f"every band below that is excluded from the T1 audit anyway.")
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    res = {
        "what": "Periphery symmetry of the official SBI recipe (mapooon/SelfBlendedImages)",
        "A_parameter_sharing": test_A_parameter_sharing(rng),
        "B_periphery_symmetry": test_B_periphery(rng),
        "C_spatial_extent": test_C_spatial_extent(rng),
        "albumentations_version": alb.__version__,
    }

    a, b, c = res["A_parameter_sharing"], res["B_periphery_symmetry"], res["C_spatial_extent"]
    beyond = max(v["max_abs_diff"] for k, v in c.items()
                 if k != "note" and int(k.split("-")[0]) >= 8)
    res["verdict"] = (
        "NO LEAK in the official recipe: parameters are shared across targets, "
        "the blend composites onto the array returned as the real, and any "
        f"residual difference dies within one 8x8 JPEG block (max {beyond} "
        f"beyond 8 px) -- inside the {DILATE_PX}px dilation the audit already "
        "discards. The official pipeline would score the null 0.500 in T1. "
        "Finding 1 is therefore a defect in OUR self_blend, not a property of "
        "SBI: we composite onto a separately-perturbed copy (deviation 1) and "
        "re-encode the blend only (deviation 2)."
        if a["shared"] and b["max_periphery_diff_after_blend"] == 0 and beyond == 0
        else "LEAK PRESENT in the official recipe -- F1 generalises. Re-read the tests."
    )

    dest = Path(__file__).resolve().parents[1] / "results" / "official_sbi_symmetry.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
