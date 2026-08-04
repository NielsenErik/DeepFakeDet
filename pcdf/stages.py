"""
Pipeline stages after ingestion: features, circuit fitting, baselines,
evaluation, explanation, reporting, benchmarking.

Every stage reads and writes files under `cfg["root"]` and records the resolved
config next to its output, so results are reproducible from disk alone.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── shared IO ───────────────────────────────────────────────────────────────

def _feat_dir(cfg: Dict) -> Path:
    return Path(cfg["root"]) / "features" / cfg["features"]["backbone"]


def _feature_dims(cfg: Dict) -> Tuple[int, int]:
    """
    (grid, channels) of the projected features.

    The channel count must come from the fitted projector, not the config:
    `features.out_dim: 0` means "no PCA, keep every coordinate", so the true value
    is only known after fitting.  Reading it from disk keeps every stage
    consistent with the features that actually exist.
    """
    grid = int(cfg["features"]["grid"])
    conf = int(cfg["features"]["out_dim"])
    proj = _feat_dir(cfg) / "projector.npz"
    if conf <= 0 and proj.exists():
        import numpy as _np

        return grid, int(_np.load(proj)["out_dim"])
    return grid, conf


def _tag(cfg: Dict) -> str:
    f, p = cfg["features"], cfg["pc"]
    _, c = _feature_dims(cfg)
    return (f"{f['backbone']}_g{f['grid']}c{c}_"
            f"{p['patch_method']}-{p['channel_method']}_K{p['n_sum_components']}")


def _load_split(cfg: Dict, split: str, dataset: str = "ffpp",
                label: Optional[int] = None, method: Optional[str] = None,
                perturb: str = "clean"
                ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load projected features + index for one (dataset, split) selection."""
    fd = _feat_dir(cfg)
    sfx = "" if perturb in (None, "clean") else f"_{perturb}"
    Z = np.load(fd / f"{dataset}_{split}{sfx}.npy", mmap_mode="r")
    idx = json.loads((fd / f"{dataset}_{split}{sfx}_index.json").read_text())
    keys = {k: np.array(v) for k, v in idx.items()}
    mask = np.ones(len(Z), bool)
    if label is not None:
        mask &= keys["label"] == label
    if method is not None:
        mask &= keys["method"] == method
    Zs = np.asarray(Z[mask], dtype=np.float32)
    return Zs, {k: v[mask] for k, v in keys.items()}


# ── stage: features ─────────────────────────────────────────────────────────

def cmd_features(cfg: Dict, args) -> None:
    import torch

    from .data.manifest import read_manifest
    from .features.backbones import build_extractor
    from .features.projector import PatchProjector

    root = Path(cfg["root"])
    fcfg = cfg["features"]
    out_dir = _feat_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset

    recs = read_manifest(root / "manifests" / f"{dataset}_ingested.csv")
    if args.methods:
        recs = [r for r in recs if r.method in args.methods]
    if getattr(args, "limit_videos", None):
        # cap videos PER (split, method) so a subset run keeps the protocol's
        # shape — capping globally would silently drop whole methods
        keep, seen = [], {}
        for r in recs:
            k = (r.split, r.method)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= args.limit_videos:
                keep.append(r)
        recs = keep
        print(f"[features] subset: {len(recs)} videos "
              f"(≤{args.limit_videos} per split/method)")
    ex = build_extractor(fcfg["backbone"], device=cfg["device"],
                         **(fcfg.get("backbone_kwargs") or {}))
    grid_src = ex.spec.grid
    grid = int(fcfg["grid"])
    assert grid_src % grid == 0, f"backbone grid {grid_src} not divisible by {grid}"
    P, C = grid * grid, ex.spec.dim

    perturb = None
    if getattr(args, "perturb", "clean") not in (None, "clean"):
        from .data.transforms import get_perturbation

        perturb = get_perturbation(args.perturb)
        print(f"[features] applying perturbation {args.perturb!r} to every crop "
              f"(models stay frozen; only the input distribution moves)")

    pseudo = bool(getattr(args, "pseudo", False))
    families = getattr(args, "pseudo_families", False)
    if pseudo:
        from .data.sbi import blend_ratio, multi_family_blend, self_blend

        print("[features] SELF-BLENDING every crop: these become the training "
              "set of p_blend for the likelihood-ratio detector. No real "
              "forgery is involved — the blends are made from these same reals.")

    def iter_crops(records, batch: int):
        """Yield (images uint8 batch, meta rows)."""
        import cv2

        rng = np.random.default_rng(cfg["seed"])
        buf, meta = [], []
        for r in records:
            d = root / "crops" / r.dataset / r.method / Path(r.video).stem
            lmk_path = d / "landmarks.npy"
            lmks = np.load(lmk_path) if (pseudo and lmk_path.exists()) else None
            for jpg in sorted(d.glob("[0-9]*.jpg")):
                img = cv2.imread(str(jpg))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if pseudo:
                    j = int(jpg.stem)
                    if lmks is None or j >= len(lmks):
                        continue
                    if families:
                        blended, mask, _fam = multi_family_blend(rgb, lmks[j], rng)
                    else:
                        blended, mask = self_blend(rgb, lmks[j], rng)
                    if not (0.02 < blend_ratio(mask) < 0.9):
                        continue          # degenerate blend teaches nothing
                    rgb = blended
                buf.append(perturb(rgb) if perturb is not None else rgb)
                meta.append({"video": Path(r.video).stem, "method": r.method,
                             "label": r.label, "split": r.split,
                             "dataset": r.dataset, "frame": int(jpg.stem),
                             "path": str(jpg)})
                if len(buf) == batch:
                    yield np.stack(buf), meta
                    buf, meta = [], []
        if buf:
            yield np.stack(buf), meta

    @torch.no_grad()
    def tokens(batch_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().div_(255.)
        x = x.to(cfg["device"], non_blocking=True)
        t = ex.encode(x)                                        # (B, P_src, C)
        if grid_src != grid:
            B = t.shape[0]
            t = t.transpose(1, 2).reshape(B, C, grid_src, grid_src)
            t = torch.nn.functional.adaptive_avg_pool2d(t, grid)
            t = t.flatten(2).transpose(1, 2)
        return t.float().cpu().numpy()

    # ── pass 1: fit the projector on REAL TRAIN only (streaming moments) ──
    proj_path = out_dir / "projector.npz"
    train_reals = [r for r in recs if r.split == "train" and r.label == 0]
    if not proj_path.exists():
        assert train_reals, "no real training videos — cannot fit the projector"
        n_acc, s1, s2 = 0, np.zeros(C, np.float64), np.zeros((C, C), np.float64)
        seen_imgs = 0
        t0 = time.time()
        for imgs, meta in iter_crops(train_reals, fcfg["batch"]):
            T = tokens(imgs).reshape(-1, C).astype(np.float64)
            s1 += T.sum(0)
            s2 += T.T @ T
            n_acc += len(T)
            seen_imgs += len(imgs)
            if seen_imgs >= fcfg["max_projector_images"]:
                break
        proj = PatchProjector.fit_from_moments(n_acc, s1, s2, n_patches=P,
                                               out_dim=fcfg["out_dim"],
                                               whiten=fcfg["whiten"])
        # per-patch statistics need projected values: one short second pass
        chunks, seen = [], 0
        for imgs, meta in iter_crops(train_reals, fcfg["batch"]):
            chunks.append(proj.transform(tokens(imgs), standardize=False))
            seen += len(imgs)
            if seen >= min(8000, fcfg["max_projector_images"]):
                break
        proj.set_patch_stats(np.concatenate(chunks))
        proj.save(proj_path)
        print(f"[features] projector fitted on {seen_imgs} real train crops "
              f"({proj.explained:.1%} variance in {fcfg['out_dim']} dims, "
              f"{time.time() - t0:.0f}s)")
    proj = PatchProjector.load(proj_path)

    # ── pass 2: project and store every split ────────────────────────────
    by_split: Dict[str, List] = {}
    for r in recs:
        by_split.setdefault(r.split, []).append(r)

    suffix = "" if getattr(args, "perturb", "clean") in (None, "clean") else f"_{args.perturb}"
    if pseudo:
        suffix += "_blend"
    for split, split_recs in by_split.items():
        if suffix.endswith("_blend") and split not in ("train", "val"):
            continue          # p_blend is fitted on train, monitored on val
        if suffix and not suffix.endswith("_blend") and split != "test":
            continue          # robustness is a TEST-time question only
        out_npy = out_dir / f"{dataset}_{split}{suffix}.npy"
        if out_npy.exists() and not args.overwrite:
            print(f"[features] {out_npy.name} exists, skipping")
            continue
        Zs, index = [], {"video": [], "method": [], "label": [], "split": [],
                         "dataset": [], "frame": [], "path": []}
        t0, n = time.time(), 0
        for imgs, meta in iter_crops(split_recs, fcfg["batch"]):
            Z = proj.transform(tokens(imgs), standardize=True)   # (B,P,c)
            Zs.append(Z.astype(np.float16))
            for m in meta:
                for k in index:
                    index[k].append(m[k])
            n += len(imgs)
            if n % 5000 < fcfg["batch"]:
                print(f"[features] {dataset}/{split}: {n} crops "
                      f"({n / max(time.time() - t0, 1e-9):.0f}/s)", flush=True)
        if not Zs:
            continue
        Zall = np.concatenate(Zs)
        np.save(out_npy, Zall)
        (out_dir / f"{dataset}_{split}{suffix}_index.json").write_text(json.dumps(index))
        print(f"[features] {dataset}/{split}: {Zall.shape} -> {out_npy} "
              f"({Zall.nbytes / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} min)")

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


# ── stage: fit the circuit ──────────────────────────────────────────────────

def cmd_fit_pc(cfg: Dict, args) -> None:
    from .models.density_pc import PCConfig, PCDetector

    root = Path(cfg["root"])
    grid, c = _feature_dims(cfg)
    Ztr, _ = _load_split(cfg, "train", label=0)
    Zval, _ = _load_split(cfg, "val", label=0)
    n_val = min(len(Zval), 4000)
    Zval = Zval[:n_val]
    print(f"[fit-pc] train {Ztr.shape}, val {Zval.shape} "
          f"(real faces only — no forgery is seen at any point)")

    pcc = PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"])
    det = PCDetector(grid, grid, c, pcc)
    struct_cache = root / "models" / f"structure_{_tag(cfg)}.pkl"
    det.fit(Ztr.reshape(len(Ztr), -1), Zval.reshape(len(Zval), -1),
            structure_cache=struct_cache)
    det.calibrate(Zval.reshape(len(Zval), -1))
    audit = det.audit(Ztr.reshape(len(Ztr), -1))
    print("[fit-pc] property audit:", json.dumps(audit, indent=2, default=float))

    out = root / "models" / f"pc_{_tag(cfg)}.pt"
    det.save(out)
    (root / "models" / f"pc_{_tag(cfg)}_audit.json").write_text(
        json.dumps(audit, indent=2, default=float))
    print(f"[fit-pc] saved {out}")


# ── stage: combine feature sets ─────────────────────────────────────────────

def cmd_combine_features(cfg: Dict, args) -> None:
    """
    Concatenate two already-extracted feature sets along the CHANNEL axis.

    The arms measured so far fail and succeed for different reasons: the
    SBI-shaped encoder captures blending geometry (it was trained on blends),
    while the spectral-residual features capture generator noise traces and are
    far more uniform across manipulation types.  Neither subsumes the other, and
    concatenating them costs no new extraction — the circuit simply gets a wider
    per-patch vector, and its region graph can still learn which coordinates
    belong together.

    Patch grids must match; channels are standardized within each source first
    so one source cannot dominate purely by scale.
    """
    root = Path(cfg["root"])
    out_name = args.out or "+".join(args.sources)
    out_dir = root / "features" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        parts, index = [], None
        for src in args.sources:
            d = root / "features" / src
            f = d / f"{args.dataset}_{split}.npy"
            if not f.exists():
                print(f"[combine] missing {f}, skipping split {split}")
                parts = []
                break
            Z = np.load(f, mmap_mode="r")
            Z = np.asarray(Z, dtype=np.float32)
            mu, sd = Z.mean((0, 1), keepdims=True), Z.std((0, 1), keepdims=True) + 1e-6
            parts.append((Z - mu) / sd)
            if index is None:
                index = json.loads((d / f"{args.dataset}_{split}_index.json").read_text())
        if not parts:
            continue
        assert len({p.shape[1] for p in parts}) == 1, "patch grids differ"
        Zc = np.concatenate(parts, axis=2).astype(np.float16)
        np.save(out_dir / f"{args.dataset}_{split}.npy", Zc)
        (out_dir / f"{args.dataset}_{split}_index.json").write_text(json.dumps(index))
        print(f"[combine] {split}: {Zc.shape} -> {out_dir}")

    # a projector stub so _feature_dims and the stages agree on the width
    from .features.projector import PatchProjector

    C = int(np.load(out_dir / f"{args.dataset}_test.npy", mmap_mode="r").shape[2])
    P = int(np.load(out_dir / f"{args.dataset}_test.npy", mmap_mode="r").shape[1])
    PatchProjector(mean=np.zeros(C, np.float32),
                   components=np.eye(C, dtype=np.float32),
                   scale=np.ones(C, np.float32), explained=1.0,
                   n_patches=P, out_dim=C, whiten=False).save(out_dir / "projector.npz")
    print(f"[combine] wrote projector stub (C={C})")


# ── stage: train the SBI encoder / baseline ─────────────────────────────────

def cmd_train_sbi(cfg: Dict, args) -> None:
    from .models.supervised import SbiConfig, train_sbi

    scfg = SbiConfig(device=cfg["device"], seed=cfg["seed"],
                     epochs=args.epochs, batch_size=args.batch_size,
                     max_frames_per_video=args.frames_per_video,
                     workers=args.workers)
    train_sbi(cfg, scfg)


# ── stage: structure ablation ───────────────────────────────────────────────

def cmd_ablate_structure(cfg: Dict, args) -> None:
    """
    Does the region graph matter, and does curvature beat the alternatives?

    Every variant trains the SAME circuit (identical K, leaves, budget, epochs)
    on the SAME features and differs only in how the scope is partitioned:

      random/random   the collapse control.  If a learned graph cannot beat
                      this, structure learning is decoration here.
      kd/*            spatial prior over patches — the hand-built adversary.
      chow_liu        MI max-spanning-tree cuts (the incumbent).
      orc             Ollivier-Ricci curvature bottleneck cuts (exact W₁).
      forman          the closed-form curvature proxy.
      spectral        recursive normalized cut — the honest adversary that
                      tells you whether transport geometry is doing anything
                      that plain spectral clustering does not.

    Reported per variant: validation NLL (the structure-quality measure that
    does not depend on any downstream choice), fit time, and detection AUC.
    """
    from sklearn.metrics import roc_auc_score

    from .models.density_pc import PCConfig, PCDetector

    root = Path(cfg["root"])
    grid, c = _feature_dims(cfg)
    Ztr, _ = _load_split(cfg, "train", label=0)
    Zval, _ = _load_split(cfg, "val", label=0)
    Zval = Zval[:4000]
    Zte, idx_te = _load_split(cfg, "test")
    if args.limit_test and len(Zte) > args.limit_test:
        sel = np.sort(np.random.default_rng(0).choice(len(Zte), args.limit_test, False))
        Zte, idx_te = Zte[sel], {k: v[sel] for k, v in idx_te.items()}

    variants = args.variants or ["random/random", "kd/random", "kd/chow_liu",
                                 "kd/orc", "kd/forman", "chow_liu/orc", "orc/orc"]
    out: Dict[str, Dict] = {}
    for spec in variants:
        pm, cm = spec.split("/")
        pcfg = dict(cfg["pc"])
        pcfg.update({"patch_method": pm, "channel_method": cm,
                     "epochs": args.epochs or cfg["pc"]["epochs"]})
        det = PCDetector(grid, grid, c, PCConfig(device=cfg["device"],
                                                 seed=cfg["seed"], **pcfg))
        t0 = time.time()
        try:
            det.fit(Ztr.reshape(len(Ztr), -1), Zval.reshape(len(Zval), -1),
                    structure_cache=root / "models" / f"structure_ab_{pm}-{cm}.pkl",
                    verbose=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"[ablate] {spec} failed: {exc}")
            out[spec] = {"error": str(exc)}
            continue
        fit_s = time.time() - t0
        det.calibrate(Zval.reshape(len(Zval), -1))
        s = det.score(Zte.reshape(len(Zte), -1))
        aucs = {k: float(roc_auc_score(idx_te["label"], v))
                for k, v in s.items() if not k.startswith("_")}
        best = max(aucs, key=aucs.get)
        val_nll = min(h["val_nll"] for h in det.history)
        out[spec] = {"val_nll": val_nll, "fit_seconds": fit_s,
                     "auc_video": aucs[best], "best_score": best,
                     "all_aucs": aucs, "structure": det.structure_info,
                     "epochs_run": len(det.history)}
        print(f"[ablate] {spec:>18}  val NLL {val_nll:10.2f}  "
              f"AUC {aucs[best]:.4f} ({best})  {fit_s:.0f}s", flush=True)

    p = root / "results" / f"structure_ablation_{_tag(cfg)}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"[ablate] wrote {p}")


# ── stage: likelihood-ratio detector ────────────────────────────────────────

def cmd_fit_ratio(cfg: Dict, args) -> None:
    """
    Fit p_real and p_blend over one shared region graph and score by their
    exact log-ratio.  This is the answer to the measured failure of raw
    likelihood (dilution + inversion, see `pcdf/eval/diagnose.py`): the ratio
    cancels the background statistics that both classes share.
    """
    from sklearn.metrics import roc_auc_score

    from .eval.metrics import video_keys, video_level
    from .models.density_pc import PCConfig
    from .models.ratio import PCRatioDetector

    root = Path(cfg["root"])
    grid, c = _feature_dims(cfg)
    Ztr, _ = _load_split(cfg, "train", label=0)
    Zval, _ = _load_split(cfg, "val", label=0)
    Zbl, _ = _load_split(cfg, "train", label=0, perturb="blend")
    Zval_bl, _ = _load_split(cfg, "val", label=0, perturb="blend")
    n = min(len(Ztr), len(Zbl))
    print(f"[ratio] {n} real / {len(Zbl)} self-blended training crops, d={grid*grid*c}")

    pcc = PCConfig(device=cfg["device"], seed=cfg["seed"], **cfg["pc"])
    det = PCRatioDetector(grid, grid, c, pcc)
    det.fit(Ztr.reshape(len(Ztr), -1), Zbl.reshape(len(Zbl), -1),
            Zval[:3000].reshape(-1, grid * grid * c),
            Zval_bl[:3000].reshape(-1, grid * grid * c),
            structure_cache=root / "models" / f"structure_{_tag(cfg)}.pkl")
    # persist before scoring: the fit is the expensive part and scoring is the
    # memory-hungry part, so a failure there must not cost the circuits
    det.save(root / "models" / f"ratio_{_tag(cfg)}.pt")
    # 1500 real frames is ample for per-patch mean/sd, and the ratio's
    # per-patch pass expands to (frames x 64 masks x 2 circuits)
    det.calibrate(Zval[:1500].reshape(-1, grid * grid * c))

    Zte, ite = _load_split(cfg, "test")
    if args.limit_test and len(Zte) > args.limit_test:
        sel = np.sort(np.random.default_rng(0).choice(len(Zte), args.limit_test, False))
        Zte, ite = Zte[sel], {k: v[sel] for k, v in ite.items()}
    s = det.score(Zte.reshape(len(Zte), -1))
    y, vkey = ite["label"].astype(int), video_keys(ite)

    out = {"tag": _tag(cfg), "n_test": int(len(Zte)), "scores": {}}
    for name, val in s.items():
        if name.startswith("_"):
            continue
        vs, vy = video_level(val, vkey, y)
        out["scores"][name] = {"auc_video": float(roc_auc_score(vy, vs)),
                               "auc_frame": float(roc_auc_score(y, val))}
    best = max(out["scores"], key=lambda k: out["scores"][k]["auc_video"])
    out["best"] = {"score": best, **out["scores"][best]}
    # per-method breakdown of the winner
    per_method = {}
    for meth in sorted(set(ite["method"][y == 1].tolist())):
        m = (y == 0) | (ite["method"] == meth)
        vs, vy = video_level(s[best][m], vkey[m], y[m])
        per_method[meth] = float(roc_auc_score(vy, vs))
    out["per_method"] = per_method

    det.save(root / "models" / f"ratio_{_tag(cfg)}.pt")
    p = root / "results" / _tag(cfg) / "ratio.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


# ── stage: baselines ────────────────────────────────────────────────────────

def cmd_baselines(cfg: Dict, args) -> None:
    from .models.baselines import build_baseline

    root = Path(cfg["root"])
    Ztr, _ = _load_split(cfg, "train", label=0)
    Zval, _ = _load_split(cfg, "val", label=0)
    Zval = Zval[:4000]
    out = root / "models" / f"baselines_{_tag(cfg)}.pkl"
    models = {}
    for name in (args.only or cfg["baselines"]):
        t0 = time.time()
        kwargs = {"device": cfg["device"]} if name in ("patchcore", "flow") else {}
        m = build_baseline(name, **kwargs).fit(Ztr)
        m.calibrate(Zval)
        models[name] = m
        print(f"[baselines] {name} fitted in {time.time() - t0:.1f}s")
    with open(out, "wb") as fh:
        pickle.dump(models, fh)
    print(f"[baselines] saved {out}")


# ── stage: evaluate ─────────────────────────────────────────────────────────

def cmd_evaluate(cfg: Dict, args) -> None:
    from .eval.metrics import evaluate_all, save_results

    root = Path(cfg["root"])
    tag = _tag(cfg)
    results = evaluate_all(cfg, tag, datasets=args.datasets or ["ffpp"],
                           robustness=args.robustness)
    save_results(results, root / "results" / tag / "scores.json")
    print(json.dumps(results["summary"], indent=2, default=float))


# ── stage: explain ──────────────────────────────────────────────────────────

def cmd_explain(cfg: Dict, args) -> None:
    from .explain.localization import run_localization

    out = run_localization(cfg, _tag(cfg), n_images=args.n_images,
                           make_figures=not args.no_figures)
    print(json.dumps(out["summary"], indent=2, default=float))


# ── stage: report ───────────────────────────────────────────────────────────

def cmd_report(cfg: Dict, args) -> None:
    from .eval.report import build_report

    path = build_report(cfg, _tag(cfg))
    print(f"[report] wrote {path}")


# ── stage: bench ────────────────────────────────────────────────────────────

def cmd_bench(cfg: Dict, args) -> None:
    """Throughput of the tensorized circuit vs the reference object graph."""
    import torch

    from .circuits.einsum_pc import EinsumPC
    from .circuits.structure import build_image_structure
    from .device import synchronize
    from .pclib import GaussianMixtureLeaf, RegionGraphPC, eval_log_prob

    dev = cfg["device"]
    out = []
    for grid, c, K in [(4, 8, 4), (8, 16, 8), (16, 16, 8)]:
        d = grid * grid * c
        Z = np.random.randn(1024, d).astype(np.float32)
        t0 = time.time()
        rg, _ = build_image_structure(Z, grid, grid, c, patch_method="kd",
                                      channel_method="chow_liu")
        t_struct = time.time() - t0
        pc = EinsumPC(rg, n_sum_components=K, n_input_components=K,
                      leaf_components=4).to(dev)
        X = torch.from_numpy(Z[:256]).to(dev)
        pc.fit_leaves(torch.from_numpy(Z))
        for _ in range(3):
            loss = -pc.log_prob(X).mean()
            loss.backward()
        synchronize(dev)
        t0 = time.time()
        for _ in range(10):
            loss = -pc.log_prob(X).mean()
            loss.backward()
        synchronize(dev)
        einsum_s = (time.time() - t0) / 10

        ref_s = None
        if args.with_reference and d <= 256:
            ref = RegionGraphPC(rg, n_sum_components=K, n_input_components=K,
                                leaf_factory=lambda i: GaussianMixtureLeaf(i, 4))
            Xc = torch.from_numpy(Z[:256])
            t0 = time.time()
            for _ in range(3):
                (-eval_log_prob(ref.root, Xc).mean()).backward()
            ref_s = (time.time() - t0) / 3

        rec = {"grid": grid, "channels": c, "d": d, "K": K,
               "structure_sec": t_struct, "einsum_sec_per_step": einsum_s,
               "reference_sec_per_step": ref_s,
               "speedup": (ref_s / einsum_s) if ref_s else None,
               "size": pc.size(), "log_partition": float(pc.log_partition())}
        out.append(rec)
        print(json.dumps(rec, indent=2, default=float), flush=True)

    p = Path(cfg["root"]) / "results" / "bench.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))


# ── registration ────────────────────────────────────────────────────────────

def register_stages(sub) -> None:
    f = sub.add_parser("features", help="crops -> projected patch features")
    f.add_argument("--dataset", default="ffpp")
    f.add_argument("--methods", nargs="*", default=None)
    f.add_argument("--overwrite", action="store_true")
    f.add_argument("--pseudo", action="store_true",
                   help="self-blend every crop -> training data for p_blend")
    f.add_argument("--pseudo-families", action="store_true",
                   help="draw from ALL pseudo-forgery families (blend / render / "
                        "overshoot / statistical) so p_blend covers both the "
                        "rougher-than-real and smoother-than-real directions")
    f.add_argument("--limit-videos", type=int, default=None,
                   help="cap videos per (split, method) — for quick subset runs")
    f.add_argument("--perturb", default="clean",
                   help="robustness transform applied to test crops "
                        "(clean|jpeg70|jpeg50|jpeg30|resize0.5|blur|noise5)")
    f.set_defaults(fn=cmd_features)

    p = sub.add_parser("fit-pc", help="train the circuit on real faces only")
    p.set_defaults(fn=cmd_fit_pc)

    cf = sub.add_parser("combine-features", help="concatenate feature sets channel-wise")
    cf.add_argument("--sources", nargs="+", required=True)
    cf.add_argument("--out", default=None)
    cf.add_argument("--dataset", default="ffpp")
    cf.set_defaults(fn=cmd_combine_features)

    t = sub.add_parser("train-sbi", help="train the SBI encoder/baseline on real frames")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=24)
    t.add_argument("--frames-per-video", type=int, default=8)
    t.add_argument("--workers", type=int, default=8)
    t.set_defaults(fn=cmd_train_sbi)

    a = sub.add_parser("ablate-structure", help="region-graph ablation (ORC vs Chow-Liu vs …)")
    a.add_argument("--variants", nargs="*", default=None,
                   help="patch/channel method pairs, e.g. kd/orc chow_liu/orc")
    a.add_argument("--epochs", type=int, default=None)
    a.add_argument("--limit-test", type=int, default=20000)
    a.set_defaults(fn=cmd_ablate_structure)

    dg = sub.add_parser("diagnose", help="why does the probe beat the density model?")
    dg.add_argument("--n-eval", type=int, default=8000)
    dg.set_defaults(fn=lambda cfg, args: __import__(
        "pcdf.eval.diagnose", fromlist=["cmd_diagnose"]).cmd_diagnose(cfg, args))

    pr = sub.add_parser("probe", help="supervised linear probe: is the signal in the features at all?")
    pr.add_argument("--max-train", type=int, default=40000)
    pr.set_defaults(fn=lambda cfg, args: __import__(
        "pcdf.eval.probe", fromlist=["cmd_probe"]).cmd_probe(cfg, args))

    fr = sub.add_parser("fit-ratio", help="two circuits scored by exact log-ratio")
    fr.add_argument("--limit-test", type=int, default=None)
    fr.set_defaults(fn=cmd_fit_ratio)

    b = sub.add_parser("baselines", help="fit the one-class baselines")
    b.add_argument("--only", nargs="*", default=None)
    b.set_defaults(fn=cmd_baselines)

    e = sub.add_parser("evaluate", help="score every model on every test set")
    e.add_argument("--datasets", nargs="*", default=None)
    e.add_argument("--robustness", action="store_true")
    e.set_defaults(fn=cmd_evaluate)

    x = sub.add_parser("explain", help="exact per-patch localization")
    x.add_argument("--n-images", type=int, default=1000)
    x.add_argument("--no-figures", action="store_true")
    x.set_defaults(fn=cmd_explain)

    r = sub.add_parser("report", help="assemble results + decision rubric")
    r.set_defaults(fn=cmd_report)

    bn = sub.add_parser("bench", help="circuit throughput vs the reference")
    bn.add_argument("--with-reference", action="store_true")
    bn.set_defaults(fn=cmd_bench)
