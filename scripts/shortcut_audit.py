"""
Is the pseudo-fake task partly solvable WITHOUT looking at the forgery?

MOTIVATION (from the code, not from a hunch).  `pcdf.data.sbi.self_blend` ends
with `match_source_pipeline`, a JPEG re-encode at q88-96, and it is on by
default.  The real crops it is compared against are read from disk and used as
they are.  So every pseudo-fake carries one more JPEG generation than every
real image, everywhere in the frame — including the ~84% of pixels the blend
never touched.

That re-encode was added for a good reason (matching the compression history of
FF++ forgeries, see the docstring), but it is applied ASYMMETRICALLY: to the
blend only.  Double-JPEG detection is a solved forensics problem, so if the
asymmetry is measurable the network can reach a low training loss without ever
learning what a blending boundary looks like — which is exactly the saturation
we observe (disc loss 0.0000, real-vs-blend AUC 0.9996, FF++ 0.827).

THREE TESTS
-----------
T1 LEAKAGE     Classify real vs self-blend using ONLY 8x8 blocks that lie
               entirely OUTSIDE the blending mask.  Those pixels differ from
               the real image by nothing except the extra JPEG generation, so
               any AUC above chance is pure leakage.  Run with the re-encode on
               and off; the difference is the size of the artefact.

T2 RELIANCE    Score real / blend(+reencode) / blend(no reencode) with the
               TRAINED SBI encoder.  If its fake-probability collapses when the
               re-encode is removed, the encoder is leaning on the leak.

T3 TRANSFER    Run the same periphery classifier on REAL FF++ forgeries vs real
               faces.  A cue that does not exist in real forgeries is capacity
               spent on nothing — and it predicts precisely the pattern we see.

Output: results/shortcut_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdf.data.sbi import blend_ratio, landmark_hull_mask, self_blend  # noqa: E402


# ── compression features (the classic double-JPEG signature) ────────────────

def _dct_blocks(gray: np.ndarray) -> np.ndarray:
    """(n_blocks, 8, 8) DCT of every JPEG-grid-aligned 8x8 block."""
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    g = gray[:h8, :w8].astype(np.float32) - 128.0
    blocks = (g.reshape(h8 // 8, 8, w8 // 8, 8)
               .transpose(0, 2, 1, 3)
               .reshape(-1, 8, 8))
    return np.stack([cv2.dct(b) for b in blocks])


# the first AC frequencies in zig-zag order: where double quantization shows up
_ZIGZAG = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)]
_BINS = np.arange(-8.5, 9.5, 1.0)          # 18 bins over [-8, 8]


def compression_features(img_rgb: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """
    Histogram of DCT coefficients over the VALID 8x8 blocks, per AC frequency.

    Double compression leaves periodic gaps and peaks in these histograms; a
    linear model on them is the standard detector.  `valid` is a boolean mask
    at block resolution (H/8, W/8) selecting blocks to use.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    bh, bw = h // 8, w // 8
    v = valid[:bh, :bw].reshape(-1)
    if v.sum() < 40:                       # too little periphery to measure
        return None
    D = _dct_blocks(gray)[v]
    feats = []
    for (u, k) in _ZIGZAG:
        c = D[:, u, k]
        hist, _ = np.histogram(c, bins=_BINS)
        feats.append(hist / max(len(c), 1))

    # Blockiness: discontinuity ACROSS block borders relative to inside them.
    # Every quantity here is restricted to valid blocks — an earlier version
    # averaged over the whole frame, which let the manipulated region leak into
    # a "periphery-only" measurement and made the test unsound in both
    # directions.
    g = gray.astype(np.float32)
    vb = valid[:bh, :bw]
    B = g[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)  # (bh,bw,8,8)

    pair_v = vb[:, :-1] & vb[:, 1:]           # block and its right neighbour
    pair_h = vb[:-1, :] & vb[1:, :]           # block and its lower neighbour
    if pair_v.sum() < 10 or pair_h.sum() < 10:
        return None
    across_v = np.abs(B[:, :-1, :, 7] - B[:, 1:, :, 0])[pair_v].mean()
    inside_v = np.abs(B[:, :, :, 3] - B[:, :, :, 4])[vb].mean()
    across_h = np.abs(B[:-1, :, 7, :] - B[1:, :, 0, :])[pair_h].mean()
    inside_h = np.abs(B[:, :, 3, :] - B[:, :, 4, :])[vb].mean()
    feats.append(np.array([across_v / (inside_v + 1e-6),
                           across_h / (inside_h + 1e-6)]))
    return np.concatenate(feats).astype(np.float32)


def periphery_blocks(mask: np.ndarray, shape, dilate_px: int = 24) -> np.ndarray:
    """
    Block-resolution mask of 8x8 blocks entirely outside the manipulated region.

    The mask is dilated generously first: a blending boundary is soft, and we
    want zero chance that a block touched by the blend counts as periphery.
    """
    h, w = shape
    m = (mask > 0.02).astype(np.uint8)
    k = np.ones((dilate_px, dilate_px), np.uint8)
    m = cv2.dilate(m, k)
    # a block is valid only if NO pixel in it is manipulated
    bh, bw = h // 8, w // 8
    mb = m[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).max(axis=(1, 3))
    return mb == 0


# ── data ────────────────────────────────────────────────────────────────────

def collect(root: Path, split: str, label: int, max_videos: int, per_video: int):
    """(crop path, landmarks) pairs from the ingested manifest."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcdf.data.manifest import read_manifest

    recs = [r for r in read_manifest(root / "manifests" / "ffpp_ingested.csv")
            if r.split == split and r.label == label]
    rng = np.random.default_rng(0)
    rng.shuffle(recs)
    out = []
    for r in recs[:max_videos]:
        d = root / "crops" / r.dataset / r.method / Path(r.video).stem
        lmk_p = d / "landmarks.npy"
        if not lmk_p.exists():
            continue
        lmks = np.load(lmk_p)
        for j, p in enumerate(sorted(d.glob("[0-9]*.jpg"))[:per_video]):
            if j < len(lmks):
                out.append((str(p), lmks[j], r.method))
    return out


def auc_lr(X0: np.ndarray, X1: np.ndarray, seed: int = 0,
           g0: np.ndarray | None = None, g1: np.ndarray | None = None) -> dict:
    """
    Honest AUC: fit logistic regression on half the GROUPS, score the other half.

    Splitting by group matters more than it looks.  In T1 each real image and
    its blend are the same underlying photo, and where the recipe leaks nothing
    their periphery features are bit-identical; a random row split then puts one
    twin in train and the other in test, the model memorises the vector with the
    wrong label, and a no-signal condition reports AUC well BELOW 0.5 instead of
    at it.  Grouping by image makes "no signal" read as 0.5, which is what a
    reader will assume it means.  In T3 the group is the video, which also stops
    frames of one video appearing on both sides.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    g0 = np.arange(len(X0)) if g0 is None else np.asarray(g0)
    g1 = (g0 if (g1 is None and len(X1) == len(X0)) else
          (np.arange(len(X1)) + 10 ** 9 if g1 is None else np.asarray(g1)))
    groups = np.unique(np.concatenate([g0, g1]))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_g = set(groups[:len(groups) // 2].tolist())

    X = np.concatenate([X0, X1])
    y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    g = np.concatenate([g0, g1])
    tr = np.array([gi in train_g for gi in g])
    te = ~tr
    if tr.sum() < 20 or te.sum() < 20 or len(set(y[te])) < 2:
        return {"auc": None, "n": int(len(X)), "error": "degenerate split"}
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    s = clf.decision_function(sc.transform(X[te]))
    return {"auc": float(roc_auc_score(y[te], s)), "n": int(len(X)),
            "n_test": int(te.sum())}


# ── T1 + T3 ─────────────────────────────────────────────────────────────────

# Blend recipes to compare.  Each entry is
#   (post_compress, pristine_background, re-encode the REAL image too)
# so the leak can be attributed to each mechanism separately.
BLEND_CONFIGS = {
    "current":                  (True,  False, False),
    "no_reencode":              (False, False, False),
    "pristine_bg":              (True,  True,  False),
    "pristine_bg_no_reencode":  (False, True,  False),
    "pristine_bg_symmetric":    (True,  True,  True),
}


def test_leakage(root: Path, n_videos: int, per_video: int) -> dict:
    """
    How much of real-vs-self-blend is decidable from pixels the blend never
    touched, under each recipe?  Every recipe uses the SAME seed per image, so
    the mask and the donor geometry are identical across conditions and only
    the mechanism under test changes.
    """
    from pcdf.data.sbi import match_source_pipeline

    items = collect(root, "train", 0, n_videos, per_video)
    print(f"[T1] {len(items)} real training crops, "
          f"{len(BLEND_CONFIGS)} recipes", flush=True)
    rng = np.random.default_rng(0)

    F_real = {k: [] for k in BLEND_CONFIGS}
    F_blend = {k: [] for k in BLEND_CONFIGS}
    for i, (path, lmk, _) in enumerate(items):
        raw = cv2.imread(path)
        if raw is None:
            continue
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        seed = int(rng.integers(1 << 30))
        made = {}
        for name, (pc, pristine, _sym) in BLEND_CONFIGS.items():
            b, m = self_blend(img, lmk, np.random.default_rng(seed),
                              post_compress=pc, pristine_background=pristine)
            made[name] = (b, m)
        if not all(0.02 < blend_ratio(m) < 0.9 for _, m in made.values()):
            continue
        # one periphery for all recipes: the union of every mask, dilated
        union = np.maximum.reduce([m for _, m in made.values()])
        valid = periphery_blocks(union, img.shape[:2])
        for name, (b, _m) in made.items():
            _pc, _pr, sym = BLEND_CONFIGS[name]
            ref = (match_source_pipeline(img, np.random.default_rng(seed + 7))
                   if sym else img)
            f_r = compression_features(ref, valid)
            f_b = compression_features(b, valid)
            if f_r is not None and f_b is not None:
                F_real[name].append(f_r)
                F_blend[name].append(f_b)
        if (i + 1) % 500 == 0:
            print(f"[T1] {i + 1}/{len(items)}", flush=True)

    out = {"note": ("AUC over periphery blocks only — pixels the blend never "
                    "touched. Chance = 0.5; anything above it is a cue the "
                    "network can use without looking at the forgery.")}
    for name in BLEND_CONFIGS:
        if F_real[name]:
            out[name] = auc_lr(np.stack(F_real[name]), np.stack(F_blend[name]))
    return out


def test_transfer(root: Path, n_videos: int, per_video: int) -> dict:
    """Is the same compression cue present in REAL FF++ forgeries?"""
    reals = collect(root, "test", 0, n_videos, per_video)
    fakes = collect(root, "test", 1, n_videos * 2, per_video)
    print(f"[T3] {len(reals)} real / {len(fakes)} forged test crops", flush=True)

    def feats(items):
        by_method = {}
        for path, lmk, method in items:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hull = landmark_hull_mask(lmk, img.shape[:2])
            valid = periphery_blocks(hull, img.shape[:2])
            f = compression_features(img, valid)
            if f is not None:
                # group by SOURCE identity, so a video and its forgeries never
                # straddle the split
                by_method.setdefault(method, ([], []))
                by_method[method][0].append(f)
                by_method[method][1].append(Path(path).parent.name.split("_")[0])
        return {k: (np.stack(v[0]), np.array(v[1])) for k, v in by_method.items()}

    Fr, Ff = feats(reals), feats(fakes)
    R = np.concatenate([v[0] for v in Fr.values()])
    Rg = np.concatenate([v[1] for v in Fr.values()])
    out = {"per_method": {}}
    for meth, (X, G) in Ff.items():
        out["per_method"][meth] = auc_lr(R, X, g0=Rg, g1=G)
    allf = np.concatenate([v[0] for v in Ff.values()])
    allg = np.concatenate([v[1] for v in Ff.values()])
    out["pooled"] = auc_lr(R, allf, g0=Rg, g1=allg)
    out["note"] = ("Same periphery/compression features, real FF++ forgeries vs "
                   "real faces. Near 0.5 means the cue the pseudo-task offers "
                   "does not exist in the forgeries we must actually detect.")
    return out


# ── T2 ──────────────────────────────────────────────────────────────────────

def test_reliance(root: Path, ckpt: Path, n_videos: int, per_video: int,
                  device: str) -> dict:
    import timm
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcdf.models.supervised import SbiConfig

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    scfg = SbiConfig(**{k: v for k, v in blob["cfg"].items()
                        if k in SbiConfig.__dataclass_fields__})
    net = timm.create_model(scfg.arch, pretrained=False, num_classes=1)
    net.load_state_dict(blob["model"])
    net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def prep(img):
        im = cv2.resize(img, (scfg.image_size, scfg.image_size),
                        interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(im.copy()).permute(2, 0, 1).float().div_(255.)
        return (x - mean) / std

    items = collect(root, "val", 0, n_videos, per_video)
    print(f"[T2] {len(items)} val real crops, checkpoint {ckpt.name}", flush=True)
    rng = np.random.default_rng(1)
    buckets = {"real": [], "blend_reencode": [], "blend_plain": []}
    batch = []

    def flush():
        if not batch:
            return
        keys, xs = zip(*batch)
        with torch.no_grad():
            p = torch.sigmoid(net(torch.stack(xs).to(device)).squeeze(1)).cpu().numpy()
        for k, v in zip(keys, p):
            buckets[k].append(float(v))
        batch.clear()

    for i, (path, lmk, _) in enumerate(items):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seed = int(rng.integers(1 << 30))
        b_pc, m1 = self_blend(img, lmk, np.random.default_rng(seed), post_compress=True)
        b_np, m2 = self_blend(img, lmk, np.random.default_rng(seed), post_compress=False)
        if not (0.02 < blend_ratio(m1) < 0.9):
            continue
        batch.append(("real", prep(img)))
        batch.append(("blend_reencode", prep(b_pc)))
        batch.append(("blend_plain", prep(b_np)))
        if len(batch) >= 48:
            flush()
        if (i + 1) % 400 == 0:
            print(f"[T2] {i + 1}/{len(items)}", flush=True)
    flush()

    from sklearn.metrics import roc_auc_score

    out = {k: {"mean_p_fake": float(np.mean(v)), "n": len(v)}
           for k, v in buckets.items()}
    r = np.array(buckets["real"])
    for k in ("blend_reencode", "blend_plain"):
        f = np.array(buckets[k])
        n = min(len(r), len(f))
        y = np.r_[np.zeros(n), np.ones(n)]
        out[k]["auc_vs_real"] = float(roc_auc_score(y, np.r_[r[:n], f[:n]]))
    out["note"] = ("Identical blend geometry in both conditions (same seed); the "
                   "only difference is the extra JPEG generation.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "deepfake_data"))
    ap.add_argument("--videos", type=int, default=200)
    ap.add_argument("--per-video", type=int, default=4)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tests", nargs="*", default=["T1", "T2", "T3"])
    a = ap.parse_args()

    root = Path(a.root)
    res = {}
    if "T1" in a.tests:
        res["T1_leakage"] = test_leakage(root, a.videos, a.per_video)
        print(json.dumps(res["T1_leakage"], indent=2), flush=True)
    if "T3" in a.tests:
        res["T3_transfer"] = test_transfer(root, a.videos, a.per_video)
        print(json.dumps(res["T3_transfer"], indent=2), flush=True)
    if "T2" in a.tests:
        ck = Path(a.checkpoint or (root / "models" / "sbi_effnetb4.pt"))
        if ck.exists():
            res["T2_reliance"] = test_reliance(root, ck, a.videos, a.per_video,
                                               a.device)
            print(json.dumps(res["T2_reliance"], indent=2), flush=True)
        else:
            print(f"[T2] no checkpoint at {ck}, skipping")

    out = root / "results" / "shortcut_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[audit] wrote {out}")


if __name__ == "__main__":
    main()
