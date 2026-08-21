"""
The detector: a probabilistic circuit fitted to REAL faces only, scored by
exact likelihood queries.

Score family (all exact, all one-pass, none available to a VAE/flow/kNN in the
same form):

  nll               −log p(z).  The PCNET score.  Known failure mode on this
                    task: a swapped face can be *more* typical than a real one
                    (smooth generated skin), which is exactly what the POC
                    measured — every global density model landed slightly BELOW
                    chance.  Kept as the honest baseline of the family.

  patch_cond        max / top-q mean over patches of the standardized
                    conditional surprisal
                        s_p = −log p(z_p | z_{−p}) = log p(z_{−p}) − log p(z)
                    A face swap is not globally atypical; it is locally
                    inconsistent WITH ITS OWN CONTEXT.  The conditional is the
                    quantity that asks precisely that, and a smooth +
                    decomposable circuit answers it exactly with two passes.

  patch_marg        same aggregation over −log p(z_p) (no context).  The
                    control that shows whether conditioning is what matters.

  incoherence       spread (std) of s_p across patches: a blended image has a
                    few strongly surprising patches among normal ones, a real
                    image is uniformly unremarkable.

Standardization matters: s_p is compared against the mean/sd of that SAME patch
position over real training faces, so "surprising" means surprising for an eye
patch, not surprising compared to the background.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..circuits.einsum_pc import EinsumPC
from ..circuits.structure import build_image_structure, build_structure, structure_report

SCORES = ("nll", "patch_cond_max", "patch_cond_topq", "patch_marg_max",
          "incoherence", "typicality")


@dataclass
class PCConfig:
    n_sum_components: int = 8
    n_input_components: int = 8
    leaf_components: int = 4
    patch_method: str = "kd"          # kd | orc | chow_liu | spectral | forman
    channel_method: str = "orc"
    max_arity: int = 4
    epochs: int = 60
    batch_size: int = 256
    lr: float = 5e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    patience: int = 8
    leaf_jitter: float = 0.2
    weight_jitter: float = 0.5
    topq: float = 0.05
    seed: int = 0
    device: str = "auto"


class PCDetector:
    """
    Trains `EinsumPC` on real patch features and exposes the exact score family.

    grid_h/grid_w/n_channels describe the layout of the flattened features:
    feature index = patch * n_channels + channel.
    """

    def __init__(self, grid_h: int, grid_w: int, n_channels: int,
                 cfg: Optional[PCConfig] = None):
        self.gh, self.gw, self.C = grid_h, grid_w, n_channels
        self.P = grid_h * grid_w
        self.d = self.P * n_channels
        self.cfg = cfg or PCConfig()
        self.pc: Optional[EinsumPC] = None
        self.region_graph = None
        self.patch_regions: Dict[int, object] = {}
        self.history: List[Dict[str, float]] = []
        self.calib: Dict[str, np.ndarray] = {}
        self.structure_info: Dict[str, float] = {}

    # ── structure + training ────────────────────────────────────────────

    def build(self, Ztr: np.ndarray, structure_cache: Optional[str | Path] = None) -> None:
        from ..circuits.structure import load_structure, save_structure

        if structure_cache and Path(structure_cache).exists():
            self.region_graph = load_structure(structure_cache)
            # patch regions are recovered by scope, not rebuilt
            self.patch_regions = self._patch_regions_by_scope(self.region_graph)
        else:
            self.region_graph, self.patch_regions = build_image_structure(
                Ztr, self.gh, self.gw, self.C,
                patch_method=self.cfg.patch_method,
                channel_method=self.cfg.channel_method,
                max_arity=self.cfg.max_arity, seed=self.cfg.seed)
            if structure_cache:
                save_structure(self.region_graph, structure_cache)
        self.structure_info = structure_report(self.region_graph)
        self.pc = EinsumPC(
            self.region_graph,
            n_sum_components=self.cfg.n_sum_components,
            n_input_components=self.cfg.n_input_components,
            leaf_components=self.cfg.leaf_components,
            weight_jitter=self.cfg.weight_jitter,
            seed=self.cfg.seed,
        ).to(self.cfg.device)

    def _patch_regions_by_scope(self, rg) -> Dict[int, object]:
        from ..pclib import region_nodes

        want = {p: frozenset(range(p * self.C, (p + 1) * self.C)) for p in range(self.P)}
        by_scope = {r.scope: r for r in region_nodes(rg)}
        return {p: by_scope[s] for p, s in want.items() if s in by_scope}

    def fit(self, Ztr: np.ndarray, Zval: Optional[np.ndarray] = None,
            structure_cache: Optional[str | Path] = None, verbose: bool = True) -> None:
        cfg = self.cfg
        if self.pc is None:
            self.build(Ztr, structure_cache=structure_cache)
        dev = cfg.device
        Xtr = torch.from_numpy(np.ascontiguousarray(Ztr, dtype=np.float32))
        self.pc.fit_leaves(Xtr[:min(len(Xtr), 20000)], jitter=cfg.leaf_jitter,
                           seed=cfg.seed)
        Xval = (torch.from_numpy(np.ascontiguousarray(Zval, dtype=np.float32)).to(dev)
                if Zval is not None else None)

        opt = torch.optim.Adam(self.pc.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        best, best_state, bad = float("inf"), None, 0
        batch = cfg.batch_size
        n = len(Xtr)
        g = torch.Generator().manual_seed(cfg.seed)

        for ep in range(cfg.epochs):
            self.pc.train()
            perm = torch.randperm(n, generator=g)
            tot, seen, t0 = 0.0, 0, time.time()
            i = 0
            while i < n:
                xb = Xtr[perm[i:i + batch]].to(dev, non_blocking=True)
                try:
                    opt.zero_grad(set_to_none=True)
                    nll = -self.pc.log_prob(xb).mean()
                    nll.backward()
                except torch.OutOfMemoryError:
                    # Back off rather than lose the run: circuit memory scales
                    # linearly in the batch, so halving always makes progress.
                    opt.zero_grad(set_to_none=True)
                    del xb
                    torch.cuda.empty_cache()
                    if batch <= 8:
                        raise
                    batch = max(8, batch // 2)
                    print(f"[pc] CUDA OOM -> batch size {batch}", flush=True)
                    continue
                torch.nn.utils.clip_grad_norm_(self.pc.parameters(), cfg.grad_clip)
                opt.step()
                tot += float(nll) * len(xb)
                seen += len(xb)
                i += batch
            sched.step()
            train_nll = tot / max(seen, 1)

            val_nll = float("nan")
            if Xval is not None:
                self.pc.eval()
                with torch.no_grad():
                    val_nll = float(-self.pc.log_prob(Xval, chunk=batch).mean())
            self.history.append({"epoch": ep, "train_nll": train_nll,
                                 "val_nll": val_nll, "sec": time.time() - t0})
            if verbose and (ep % 5 == 0 or ep == cfg.epochs - 1):
                print(f"[pc] epoch {ep:3d}  train NLL {train_nll:10.3f}  "
                      f"val NLL {val_nll:10.3f}  ({time.time() - t0:.1f}s)", flush=True)

            # early stopping on VALIDATION NLL: the POC showed longer training
            # keeps improving train NLL while test AUROC degrades
            monitor = val_nll if Xval is not None else train_nll
            if monitor < best - 1e-4:
                best, bad = monitor, 0
                best_state = {k: v.detach().clone() for k, v in self.pc.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    if verbose:
                        print(f"[pc] early stop at epoch {ep} (best {best:.3f})")
                    break
        if best_state is not None:
            self.pc.load_state_dict(best_state)
        self.pc.eval()

    # ── exact queries ───────────────────────────────────────────────────

    def _patch_masks(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(P, d) masks: `keep_patch` observes only patch p; `drop_patch`
        observes everything except patch p."""
        eye = torch.zeros(self.P, self.d, dtype=torch.bool, device=device)
        for p in range(self.P):
            eye[p, p * self.C:(p + 1) * self.C] = True
        return eye, ~eye

    def _auto_chunk_rows(self, budget_bytes: int = 1 << 30) -> int:
        """
        Rows per circuit pass that keep the leaf layer inside `budget_bytes`.

        `region_log_marginals` expands (B samples, Q masks) into B*Q rows before
        the leaf layer sees them, and the leaf layer materialises (rows, L, M)
        with L = d * n_input_components.  With P = 64 patch masks and B = 64
        that is 4096 rows, so a fixed chunk of 8192 never chunks at all and the
        arm's memory use is quadratic in nothing anyone looks at until it OOMs:
        d = 6080 (spectral) fits in 16 GB, d = 7104 (combined) does not, and the
        failure lands in `calibrate` AFTER training has finished.

        Scaling the chunk with d instead makes that a speed cost rather than a
        crash.  Chunking is exact, so no result changes.
        """
        L, M = self.pc.leaves.mus.shape
        per_row = L * M * 4 * 3          # ~3 live float32 intermediates
        return int(max(64, min(8192, budget_bytes // max(per_row, 1))))

    @torch.no_grad()
    def patch_surprisal(self, Z: np.ndarray, batch: int = 64,
                        chunk_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Exact per-patch quantities for every sample.

        returns {"cond": (N,P), "marg": (N,P), "logp": (N,)} where
            cond[n,p] = −log p(z_p | z_{−p})   (two exact passes)
            marg[n,p] = −log p(z_p)            (one exact pass)
        """
        dev = self.cfg.device
        if chunk_rows is None:
            chunk_rows = self._auto_chunk_rows()
        keep, drop = self._patch_masks(dev)
        conds, margs, logps = [], [], []
        for i in range(0, len(Z), batch):
            x = torch.from_numpy(np.ascontiguousarray(Z[i:i + batch], np.float32)).to(dev)
            logp = self.pc.log_prob(x)                                  # (B,)
            m_drop = self.pc.region_log_marginals(x, drop, chunk_rows)   # (B,P)
            m_keep = self.pc.region_log_marginals(x, keep, chunk_rows)   # (B,P)
            conds.append((m_drop - logp[:, None]).cpu().numpy())
            margs.append((-m_keep).cpu().numpy())
            logps.append(logp.cpu().numpy())
        return {"cond": np.concatenate(conds), "marg": np.concatenate(margs),
                "logp": np.concatenate(logps)}

    def calibrate(self, Zref: np.ndarray, batch: int = 64) -> None:
        """
        Per-patch reference statistics from REAL data (train or val).  Every
        localized score is standardized against these, so a score of +4 means
        "four sigma above what this patch position normally is on real faces".
        """
        q = self.patch_surprisal(Zref, batch=batch)
        self.calib = {
            "cond_mean": q["cond"].mean(0), "cond_std": q["cond"].std(0) + 1e-6,
            "marg_mean": q["marg"].mean(0), "marg_std": q["marg"].std(0) + 1e-6,
            "logp_median": np.array([np.median(q["logp"])]),
            "nll_mean": np.array([(-q["logp"]).mean()]),
            "nll_std": np.array([(-q["logp"]).std() + 1e-6]),
        }

    # ── scores ──────────────────────────────────────────────────────────

    def score(self, Z: np.ndarray, batch: int = 64) -> Dict[str, np.ndarray]:
        """All members of the score family plus the per-patch maps."""
        q = self.patch_surprisal(Z, batch=batch)
        if not self.calib:
            raise RuntimeError("call calibrate() on real data before scoring")
        zc = (q["cond"] - self.calib["cond_mean"]) / self.calib["cond_std"]
        zm = (q["marg"] - self.calib["marg_mean"]) / self.calib["marg_std"]
        k = max(1, int(round(self.cfg.topq * self.P)))
        topq = np.sort(zc, axis=1)[:, -k:].mean(1)
        # TWO-SIDED patch scores.  Measured on FF++: Deepfakes and FaceShifter
        # score BELOW chance under a one-sided "high surprisal" rule, i.e. the
        # swapped region is *more* typical than real tissue — generated skin is
        # smoother than the camera's. |z| and the low tail catch that; keeping
        # both directions separate keeps the diagnosis visible instead of
        # hiding it inside an absolute value.
        absz = np.abs(zc)
        return {
            "nll": -q["logp"],
            "patch_cond_max": zc.max(1),
            "patch_cond_topq": topq,
            "patch_cond_absmax": absz.max(1),
            "patch_cond_abstopq": np.sort(absz, axis=1)[:, -k:].mean(1),
            "patch_cond_lowmax": (-zc).max(1),
            "patch_marg_max": zm.max(1),
            "patch_marg_absmax": np.abs(zm).max(1),
            "incoherence": zc.std(1),
            "typicality": np.abs(-q["logp"] - (-self.calib["logp_median"][0])),
            "_patch_cond_z": zc,
            "_patch_marg_z": zm,
        }

    # ── audit + persistence ─────────────────────────────────────────────

    def audit(self, Z: Optional[np.ndarray] = None) -> Dict[str, float]:
        x = (torch.from_numpy(np.ascontiguousarray(Z[:8], np.float32)).to(self.cfg.device)
             if Z is not None else None)
        rep = self.pc.validate(x)
        rep.update({f"structure_{k}": v for k, v in self.structure_info.items()})
        rep.update({f"size_{k}": v for k, v in self.pc.size().items()})
        return rep

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.pc.state_dict(),
            "cfg": self.cfg.__dict__,
            "layout": {"gh": self.gh, "gw": self.gw, "C": self.C},
            "calib": {k: v for k, v in self.calib.items()},
            "history": self.history,
            "structure_info": self.structure_info,
        }, path)

    def load(self, path: str | Path, Ztr_for_structure: np.ndarray,
             structure_cache: Optional[str | Path] = None) -> "PCDetector":
        blob = torch.load(path, map_location=self.cfg.device, weights_only=False)
        if self.pc is None:
            self.build(Ztr_for_structure, structure_cache=structure_cache)
        self.pc.load_state_dict(blob["state_dict"])
        self.calib = {k: np.asarray(v) for k, v in blob["calib"].items()}
        self.history = blob.get("history", [])
        self.structure_info = blob.get("structure_info", {})
        self.pc.eval()
        return self
