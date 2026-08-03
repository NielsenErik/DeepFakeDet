"""
POC: deepfake detection as density-based anomaly detection with a
probabilistic circuit — the PCNET recipe (arXiv:2605.05953) transferred from
LLM residual streams to vision-encoder embeddings.

Pipeline
--------
1. Real faces: LFW (identity-disjoint train/test split).
2. Fakes: self-blended images (SBI-style pseudo face-swaps, Shiohara &
   Yamasaki CVPR 2022) built ONLY from test identities — never seen in any
   form at training time. A second, easier artifact type (bicubic down-up
   resampling, the classic generator fingerprint) is evaluated too.
3. Embedding: frozen ImageNet ResNet-18 penultimate features -> PCA -> z-score
   (all fit on real train only).
4. Density: DensityPC from src/bak/allinone_probabilistic_circuits.py —
   structured-decomposable, smooth, exactly normalized. Trained by exact NLL
   on REAL faces only. Score = -log p(z).
5. Property audit: smoothness, decomposability, structured decomposability,
   partition function == 1, exact-marginal consistency.
6. Baselines: Mahalanobis (Ledoit-Wolf) and full-covariance GMM in the same
   feature space.

Everything the PC needs (smoothness + decomposability) is validated loudly at
run time; nothing in this pipeline is allowed to break normalization.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

SRC_BAK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bak")
sys.path.insert(0, SRC_BAK)

from allinone_probabilistic_circuits import (  # noqa: E402
    DensityPC,
    GaussianMixtureLeaf,
    chow_liu_vtree,
    random_balanced_vtree,
    vtree_depth,
)

SEED = 0
# "lfw": real LFW faces + generated pseudo-fakes (self-blend / down-up).
# "openforensics": real deepfake data — HF Hemg/deepfake-and-real-images,
#   a mirror of the OpenForensics-derived face crops (GAN face swaps).
DATA_SET = os.environ.get("POC_DATA", "lfw")
OPENFORENSICS_PARQUET = os.environ.get(
    "POC_OF_PARQUET",
    "/private/tmp/claude-501/-Users-eriknielsen-Documents-UNITN-PHD-MAIN-Project-DeepFakeDet"
    "/0eceb845-293a-4ec9-8ba7-d32b9399d7d7/scratchpad/df_shard0.parquet",
)
FEATURE_SET = os.environ.get("POC_FEATURES", "resnet")  # resnet | forensic | clip
N_PCA = 24
LEAF_COMPONENTS = 4
SUM_COMPONENTS = 2
EPOCHS = int(os.environ.get("POC_EPOCHS", "300"))
LR = 5e-3
JITTER = 0.2
CACHE = os.environ.get(
    "POC_CACHE",
    "/private/tmp/claude-501/-Users-eriknielsen-Documents-UNITN-PHD-MAIN-Project-DeepFakeDet"
    "/0eceb845-293a-4ec9-8ba7-d32b9399d7d7/scratchpad/"
    f"poc_features_{DATA_SET}_{FEATURE_SET}.npz",
)

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════
# 1+2. Data: real faces and pseudo-fakes
# ═══════════════════════════════════════════════════════════════════════════

def load_lfw():
    from sklearn.datasets import fetch_lfw_people

    d = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=20)
    X = d.images.astype(np.float32)  # (N, H, W, 3)
    if X.max() > 1.5:
        X /= 255.0
    return X, d.target


def load_openforensics(n_real: int = 4500, n_fake: int = 1500):
    """
    Real deepfake data: Hemg/deepfake-and-real-images (HF mirror of the
    OpenForensics-derived 'Deepfake and real images' face crops; label 0=Fake
    GAN face swaps, 1=Real). Streams one parquet shard, decodes the first
    n_real reals and n_fake fakes, resized to 128x128 RGB in [0,1].
    """
    import cv2
    import pyarrow.parquet as pq

    reals, fakes = [], []
    pf = pq.ParquetFile(OPENFORENSICS_PARQUET)
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        imgs = batch.column("image")
        labs = batch.column("label")
        for i in range(len(labs)):
            lab = labs[i].as_py()
            bucket, cap = (reals, n_real) if lab == 1 else (fakes, n_fake)
            if len(bucket) >= cap:
                continue
            raw = np.frombuffer(imgs[i]["bytes"].as_py(), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            bucket.append(img.astype(np.float32) / 255.0)
        if len(reals) >= n_real and len(fakes) >= n_fake:
            break
    return np.stack(reals), np.stack(fakes)


def _soft_face_mask(h: int, w: int, rng) -> np.ndarray:
    """Random elliptical face-region mask with blurred boundary."""
    import cv2

    mask = np.zeros((h, w), np.float32)
    cy = h * (0.52 + 0.06 * rng.uniform(-1, 1))
    cx = w * (0.50 + 0.05 * rng.uniform(-1, 1))
    ay = h * (0.34 + 0.06 * rng.uniform(-1, 1))
    ax = w * (0.30 + 0.05 * rng.uniform(-1, 1))
    cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)),
                rng.uniform(-12, 12), 0, 360, 1.0, -1)
    k = 2 * int(min(h, w) * 0.08) + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask[..., None]


def self_blend(target: np.ndarray, source: np.ndarray, rng,
               return_mask: bool = False):
    """
    SBI-style pseudo face swap: color-match the source face to the target,
    push it through a down-up resample (the generator/warping fingerprint),
    and blend it into the target under a soft elliptical mask.
    """
    import cv2

    h, w = target.shape[:2]
    mask = _soft_face_mask(h, w, rng)

    # per-channel color transfer inside the mask region
    src = source.copy()
    m = mask[..., 0] > 0.5
    for c in range(3):
        mu_t, sd_t = target[..., c][m].mean(), target[..., c][m].std() + 1e-6
        mu_s, sd_s = src[..., c][m].mean(), src[..., c][m].std() + 1e-6
        src[..., c] = (src[..., c] - mu_s) / sd_s * sd_t + mu_t
    src = np.clip(src, 0, 1)

    # resampling artifact on the donor face
    f = rng.uniform(0.5, 0.75)
    small = cv2.resize(src, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_LINEAR)
    src = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    blended = (mask * src + (1.0 - mask) * target).astype(np.float32)
    return (blended, mask[..., 0]) if return_mask else blended


def down_up(img: np.ndarray, rng) -> np.ndarray:
    """Whole-image bicubic down-up resample (easy frequency artifact)."""
    import cv2

    h, w = img.shape[:2]
    f = rng.uniform(0.45, 0.6)
    small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Frozen embedding
# ═══════════════════════════════════════════════════════════════════════════

def resnet_features(images: np.ndarray, batch: int = 64) -> np.ndarray:
    """Penultimate (512-d) ImageNet ResNet-18 features for (N,H,W,3) in [0,1]."""
    import torch.nn as nn
    import torchvision
    from torchvision.models import ResNet18_Weights

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval().to(dev)

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2).to(dev)
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False)
            feats.append(net((x - mean) / std).cpu().numpy())
    return np.concatenate(feats, 0)


def forensic_features(images: np.ndarray) -> np.ndarray:
    """
    Artifact-sensitive forensic descriptor (~34-d) per image:
      * 16 radial log-DCT band energies (Frank et al., ICML 2020 — resampling
        and generator fingerprints live in the spectrum),
      * noise-residual moments per channel (std / skew / kurtosis) and
        cross-channel residual correlations (blending disturbs sensor noise),
      * inner-ellipse vs outer-ring color statistics deltas (self-blend color
        transfer breaks face/background consistency).
    """
    import cv2
    from scipy.stats import kurtosis, skew

    h0 = w0 = 128
    fy, fx = np.mgrid[0:h0, 0:w0]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    bands = np.linspace(0, rad.max() + 1e-6, 17)
    band_idx = [(rad >= bands[i]) & (rad < bands[i + 1]) for i in range(16)]

    mask_inner = np.zeros((h0, w0), np.uint8)
    cv2.ellipse(mask_inner, (w0 // 2, int(h0 * 0.52)),
                (int(w0 * 0.28), int(h0 * 0.32)), 0, 0, 360, 1, -1)
    inner = mask_inner.astype(bool)
    outer = ~inner

    out = []
    for img in images:
        img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0

        d = cv2.dct(gray)
        logmag = np.log(np.abs(d) + 1e-8)
        spec = [logmag[m].mean() for m in band_idx]

        res = img - cv2.GaussianBlur(img, (5, 5), 0)
        moments = []
        for c in range(3):
            rc = res[..., c].ravel()
            moments += [rc.std(), float(skew(rc)), float(kurtosis(rc))]
        cc = [np.corrcoef(res[..., a].ravel(), res[..., b].ravel())[0, 1]
              for a, b in ((0, 1), (0, 2), (1, 2))]

        color = []
        for c in range(3):
            color += [img[..., c][inner].mean() - img[..., c][outer].mean(),
                      img[..., c][inner].std() - img[..., c][outer].std()]

        out.append(np.array(spec + moments + cc + color, dtype=np.float32))
    return np.stack(out)


def clip_features(images: np.ndarray, batch: int = 32) -> np.ndarray:
    """
    Pooled CLIP ViT-L/14 vision features (1024-d) — the UnivFD substrate
    (Ojha et al., CVPR 2023): CLIP's feature space retains generator traces
    that ImageNet-classification features discard. Requires `transformers`
    (env expllm_env, not cvad_venv).
    """
    from transformers import CLIPVisionModel

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14").eval().to(dev)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=dev).view(1, 3, 1, 1)

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2).to(dev)
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bicubic", align_corners=False)
            out = net(pixel_values=(x - mean) / std)
            feats.append(out.pooler_output.float().cpu().numpy())
    return np.concatenate(feats, 0)


_EXTRACTORS = {
    "forensic": forensic_features,
    "resnet": resnet_features,
    "clip": clip_features,
}


def build_features():
    if os.path.exists(CACHE):
        print(f"[data] loading cached features from {CACHE}")
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}

    t0 = time.time()
    if DATA_SET == "openforensics":
        R, F = load_openforensics()
        print(f"[data] OpenForensics crops: {len(R)} real, {len(F)} fake "
              f"({time.time() - t0:.1f}s)")
        perm = rng.permutation(len(R))     # no identity labels: random split
        cut = len(R) - len(F)              # test real count = fake count
        Xtr, Xte = R[perm[:cut]], R[perm[cut:]]
        images = {"train_real": Xtr, "test_real": Xte, "fake_swap": F}
    else:
        X, ident = load_lfw()
        print(f"[data] LFW: {X.shape}, {len(set(ident))} identities "
              f"({time.time() - t0:.1f}s)")
        # identity-disjoint split
        ids = np.array(sorted(set(ident)))
        rng.shuffle(ids)
        cut = int(0.7 * len(ids))
        train_mask = np.isin(ident, ids[:cut])
        Xtr, Xte = X[train_mask], X[~train_mask]
        ident_te = ident[~train_mask]

        # pseudo-fakes from TEST identities only
        fakes_sb, fakes_du = [], []
        idx = np.arange(len(Xte))
        for i in idx:
            others = idx[ident_te != ident_te[i]]
            j = rng.choice(others)
            fakes_sb.append(self_blend(Xte[i], Xte[j], rng))
            fakes_du.append(down_up(Xte[i], rng))
        images = {"train_real": Xtr, "test_real": Xte,
                  "fake_selfblend": np.stack(fakes_sb),
                  "fake_downup": np.stack(fakes_du)}
    print(f"[data] train real {len(images['train_real'])}, "
          f"test real {len(images['test_real'])}")

    t0 = time.time()
    extract = _EXTRACTORS[FEATURE_SET]
    out = {k: extract(v) for k, v in images.items()}
    print(f"[feat] {FEATURE_SET} features done ({time.time() - t0:.1f}s)")
    np.savez_compressed(CACHE, **out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4+5. PC density on real faces, with the full property audit
# ═══════════════════════════════════════════════════════════════════════════

class SymBrokenGMLeaf(GaussianMixtureLeaf):
    """
    GaussianMixtureLeaf whose data fit adds location noise. fit_leaves()'s own
    jitter only targets scalar-mu leaves (`mu`/`log_sigma` attributes), so the
    K sibling copies of a GaussianMixtureLeaf start exactly identical and
    gradient symmetry then keeps them identical forever — the circuit
    silently degenerates to a product of marginals and the vtree cannot
    matter. Symmetry must be broken here, at the leaf's own fit().
    """

    def fit(self, X) -> None:
        super().fit(X)
        with torch.no_grad():
            self.mus.add_(torch.randn_like(self.mus) * JITTER * self.sigmas)
            self.logits.add_(0.1 * torch.randn_like(self.logits))


def train_pc(Ztr: torch.Tensor, vtree, tag: str) -> DensityPC:
    pc = DensityPC(
        vtree, n_sum_components=SUM_COMPONENTS,
        leaf_factory=lambda i: SymBrokenGMLeaf(i, LEAF_COMPONENTS),
    )
    pc.fit_leaves(Ztr, jitter=JITTER)
    n_par = sum(p.numel() for p in pc.parameters())
    print(f"[pc:{tag}] vtree depth {vtree_depth(vtree)}, {n_par} params")

    opt = torch.optim.Adam(pc.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        opt.zero_grad()
        nll = -pc.log_prob(Ztr).mean()
        nll.backward()
        opt.step()
        if ep % 50 == 0 or ep == EPOCHS - 1:
            print(f"[pc:{tag}] epoch {ep:4d}  train NLL {nll.item():8.3f} "
                  f"({time.time() - t0:.0f}s)")
    return pc


def audit_pc(pc: DensityPC, Z: torch.Tensor) -> None:
    pc.validate()  # smoothness + decomposability + structured decomposability
    logZ = float(pc.log_partition())
    assert abs(logZ) < 1e-4, f"partition function broke: log Z = {logZ}"
    # exact-marginal consistency: marginalizing nothing == log_prob,
    # marginalizing everything == log 1 = 0
    with torch.no_grad():
        lp = pc.log_prob(Z[:8])
        m0 = pc.log_marginal(Z[:8], [])
        mall = pc.log_marginal(Z[:8], list(range(Z.shape[1])))
    assert torch.allclose(lp, m0, atol=1e-5)
    assert mall.abs().max() < 1e-4
    # a real partial marginal must be a valid log-prob (finite)
    with torch.no_grad():
        mhalf = pc.log_marginal(Z[:8], list(range(Z.shape[1] // 2)))
    assert torch.isfinite(mhalf).all()
    print(f"[audit] smooth ✓  decomposable ✓  structured-decomposable ✓  "
          f"log Z = {logZ:+.2e} ✓  exact marginals ✓")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def auroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.zeros(len(scores_neg)), np.ones(len(scores_pos))]
    return roc_auc_score(y, np.r_[scores_neg, scores_pos])


def main() -> None:
    data = build_features()

    # projection fit on REAL TRAIN only. resnet: PCA + z-score (512d is too
    # high for scalar-parameter PCs). forensic: z-score only — KEEP the raw
    # correlated coordinates so the learned vtree has structure to exploit.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if FEATURE_SET == "forensic":
        if os.environ.get("POC_WHITEN") == "1":
            # full-dim whitening: a fixed invertible linear map fit on real
            # train only. The PC still models a valid normalized density (in
            # rotated coordinates); linear correlations are absorbed by the
            # transform so circuit capacity goes to non-Gaussian structure.
            pca = PCA(whiten=True, random_state=SEED).fit(data["train_real"])
            scaler = StandardScaler().fit(pca.transform(data["train_real"]))
            proj = lambda A: scaler.transform(pca.transform(A)).astype(np.float32)
        else:
            scaler = StandardScaler().fit(data["train_real"])
            proj = lambda A: scaler.transform(A).astype(np.float32)
        n_dim = data["train_real"].shape[1]
    else:
        pca = PCA(n_components=N_PCA, random_state=SEED,
                  whiten=os.environ.get("POC_WHITEN") == "1",
                  ).fit(data["train_real"])
        scaler = StandardScaler().fit(pca.transform(data["train_real"]))
        proj = lambda A: scaler.transform(pca.transform(A)).astype(np.float32)
        n_dim = N_PCA

    Ztr = torch.from_numpy(proj(data["train_real"]))
    Zte = torch.from_numpy(proj(data["test_real"]))
    fake_keys = sorted(k for k in data if k.startswith("fake_"))
    Zfk = {k: torch.from_numpy(proj(data[k])) for k in fake_keys}
    if FEATURE_SET == "forensic":
        print(f"[feat] forensic features: {n_dim}d (z-scored, no PCA)")
    else:
        evr = pca.explained_variance_ratio_.sum()
        print(f"[feat] PCA {N_PCA}d keeps {evr:.1%} variance")

    results = {}
    raw = {}  # model -> (train scores, test-real scores, {fake: scores})

    # ── PC with learned (Chow-Liu) vtree, and random vtree control ────────
    vtree_tags = os.environ.get("POC_VTREES", "chow_liu,random").split(",")
    vtree_makers = {
        "chow_liu": lambda: chow_liu_vtree(Ztr.numpy()),
        "random": lambda: random_balanced_vtree(list(range(n_dim)), seed=SEED),
    }
    for tag in vtree_tags:
        pc = train_pc(Ztr, vtree_makers[tag](), tag)
        audit_pc(pc, Ztr)
        with torch.no_grad():
            raw[f"PC({tag})"] = (
                pc.anomaly_score(Ztr).numpy(),
                pc.anomaly_score(Zte).numpy(),
                {k: pc.anomaly_score(Zfk[k]).numpy() for k in fake_keys},
            )
            s_te = raw[f"PC({tag})"][1]
            results[f"PC({tag})"] = tuple(
                auroc(s_te, raw[f"PC({tag})"][2][k]) for k in fake_keys)

    # ── baselines in the identical feature space ──────────────────────────
    from sklearn.covariance import LedoitWolf
    from sklearn.mixture import GaussianMixture

    lw = LedoitWolf().fit(Ztr.numpy())
    maha = lambda A: lw.mahalanobis(A)
    raw["Mahalanobis"] = (maha(Ztr.numpy()), maha(Zte.numpy()),
                          {k: maha(Zfk[k].numpy()) for k in fake_keys})

    gmm = GaussianMixture(4, covariance_type="full", random_state=SEED,
                          reg_covar=1e-4).fit(Ztr.numpy())
    gs = lambda A: -gmm.score_samples(A)
    raw["GMM-full(4)"] = (gs(Ztr.numpy()), gs(Zte.numpy()),
                          {k: gs(Zfk[k].numpy()) for k in fake_keys})

    for name in ("Mahalanobis", "GMM-full(4)"):
        _, s_te, s_fk = raw[name]
        results[name] = tuple(auroc(s_te, s_fk[k]) for k in fake_keys)

    cols = [k.removeprefix("fake_") for k in fake_keys]
    print("\n════════ AUROC (real-test vs fakes, higher = better) ════════")
    print(f"{'model':<16} " + " ".join(f"{c:>12}" for c in cols))
    for k, vals in results.items():
        print(f"{k:<16} " + " ".join(f"{v:>12.3f}" for v in vals))

    # two-sided typicality: |score − median(train score)| flags samples that
    # are TOO typical as well as too atypical (smooth GAN faces sit closer to
    # the mode than held-out reals — the classic likelihood-OOD pathology)
    print("\n════ two-sided typicality AUROC |NLL − train median| ════")
    print(f"{'model':<16} " + " ".join(f"{c:>12}" for c in cols))
    for name, (s_tr, s_te, s_fk) in raw.items():
        med = float(np.median(s_tr))
        vals = [auroc(np.abs(s_te - med), np.abs(s_fk[k] - med))
                for k in fake_keys]
        print(f"{name:<16} " + " ".join(f"{v:>12.3f}" for v in vals))


if __name__ == "__main__":
    main()
