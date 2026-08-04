"""
Likelihood-RATIO detection with two exact circuits.

WHY THE RAW LIKELIHOOD CANNOT WORK, MEASURED RATHER THAN ASSUMED
----------------------------------------------------------------
`pcdf/eval/diagnose.py` established two things on FF++ c23 spectral features:

  * DILUTION — restricting the exact marginal to the most discriminative
    coordinates gains +0.093 AUC over the full joint. The shift is real but
    spread thin: log p(z) sums surprisal over thousands of coordinates, and the
    handful that carry forgery evidence are outvoted by the rest.
  * INVERSION — 55% of forgeries have HIGHER likelihood than the median real
    face. Generated skin is smoother and more average than camera output, so a
    forgery is not an outlier in any direction a one-sided NLL can use.

Both are the textbook failure of likelihood-based OOD detection (Ren et al.,
NeurIPS 2019): p(x) is dominated by background statistics shared by both
classes, and the cure is a RATIO against a background model, which cancels them.

THE CONSTRUCTION
----------------
Two circuits over the SAME region graph:

    p_real   fitted on real training faces
    p_blend  fitted on SELF-BLENDED versions of those same training faces

and the score is the exact log-ratio

    s(z) = log p_blend(z) − log p_real(z)

No real forgery is ever seen — the self-blends are manufactured from the
training reals by `pcdf.data.sbi`, the SBI protocol.  What changes is only the
question being asked: not "is this face unusual" (which the data says it is
not) but "is this face better explained by the blended process than by the
camera process", which is what a forensic detector should ask.

WHAT THIS PRESERVES
-------------------
Everything the project is about.  Both circuits are smooth, decomposable and
structured-decomposable, both are exactly normalized, so the ratio is an exact
log-likelihood ratio, not a heuristic score.  And because both support exact
marginals, the PER-PATCH ratio

    s_p(z) = log p_blend(z_p | z_−p) − log p_real(z_p | z_−p)

is also exact — a localization map with a probabilistic meaning ("this region
is k nats better explained by blending"), which no memory bank or flow provides.
It is a generative classifier: Bayes-optimal when both densities are correct.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .density_pc import PCConfig, PCDetector


class PCRatioDetector:
    """
    Two `PCDetector`s over one shared region graph, scored by their exact
    log-ratio.  The structure is learned once (on real data) and reused, so the
    ratio compares like with like: any difference in score comes from the
    parameters, never from a different scope decomposition.
    """

    def __init__(self, grid_h: int, grid_w: int, n_channels: int,
                 cfg: Optional[PCConfig] = None):
        self.gh, self.gw, self.C = grid_h, grid_w, n_channels
        self.P = grid_h * grid_w
        self.cfg = cfg or PCConfig()
        self.real = PCDetector(grid_h, grid_w, n_channels, self.cfg)
        self.blend = PCDetector(grid_h, grid_w, n_channels, self.cfg)
        self.calib: Dict[str, np.ndarray] = {}

    # ── fitting ──────────────────────────────────────────────────────────

    def fit(self, Zreal: np.ndarray, Zblend: np.ndarray,
            Zval_real: Optional[np.ndarray] = None,
            Zval_blend: Optional[np.ndarray] = None,
            structure_cache: Optional[str | Path] = None,
            verbose: bool = True) -> None:
        if verbose:
            print(f"[ratio] p_real on {len(Zreal)} real crops")
        self.real.fit(Zreal, Zval_real, structure_cache=structure_cache,
                      verbose=verbose)
        # share the structure: same regions, same scopes, same budget
        self.blend.region_graph = self.real.region_graph
        self.blend.patch_regions = self.real.patch_regions
        self.blend.structure_info = self.real.structure_info
        from ..circuits.einsum_pc import EinsumPC

        self.blend.pc = EinsumPC(
            self.real.region_graph,
            n_sum_components=self.cfg.n_sum_components,
            n_input_components=self.cfg.n_input_components,
            leaf_components=self.cfg.leaf_components,
            weight_jitter=self.cfg.weight_jitter,
            seed=self.cfg.seed + 1,
        ).to(self.cfg.device)
        if verbose:
            print(f"[ratio] p_blend on {len(Zblend)} self-blended crops "
                  f"(manufactured from the SAME real training frames)")
        self.blend.fit(Zblend, Zval_blend, structure_cache=None, verbose=verbose)

    def fit_hybrid(self, Zreal: np.ndarray, Zblend: np.ndarray,
                   Zval_real: Optional[np.ndarray] = None,
                   Zval_blend: Optional[np.ndarray] = None,
                   structure_cache: Optional[str | Path] = None,
                   lam: float = 0.5, epochs: Optional[int] = None,
                   verbose: bool = True) -> Dict[str, list]:
        """
        Train BOTH circuits against one hybrid objective:

            L = λ · [ NLL_real(real) + NLL_blend(blend) ]        generative anchor
              + (1−λ) · BCE( σ( log p_blend(x) − log p_real(x) ), y )

        λ = 1 reproduces two independent maximum-likelihood fits (what we did
        before); λ = 0 is a purely discriminative circuit pair.

        WHY.  Fitting each density by likelihood spends capacity on everything,
        including the thousands of coordinates that encode identity, pose and
        lighting — which are identical in both classes.  The ratio then has to
        cancel them at scoring time.  Measured consequence: the circuit ties a
        plain GMM (0.828 vs 0.830), because the extra capacity went somewhere
        useless.  A discriminative term instead puts capacity where the two
        processes actually differ.  This is the pair-of-generative-circuits
        version of discriminative SPN learning (Gens & Domingos, NeurIPS 2012),
        whose tractable class is BROADER than the generative one — so nothing
        structural is lost.

        Everything stays exactly normalized (softmax sum weights, normalized
        leaves), so log p, marginals, conditionals and box queries remain exact
        for both circuits and the ratio remains a true log-likelihood ratio.
        """
        import torch.nn.functional as F

        cfg = self.cfg
        epochs = epochs or cfg.epochs
        dev = cfg.device
        if self.real.pc is None:
            self.real.build(Zreal, structure_cache=structure_cache)
        from ..circuits.einsum_pc import EinsumPC

        if self.blend.pc is None:
            self.blend.region_graph = self.real.region_graph
            self.blend.patch_regions = self.real.patch_regions
            self.blend.structure_info = self.real.structure_info
            self.blend.pc = EinsumPC(
                self.real.region_graph,
                n_sum_components=cfg.n_sum_components,
                n_input_components=cfg.n_input_components,
                leaf_components=cfg.leaf_components,
                weight_jitter=cfg.weight_jitter, seed=cfg.seed + 1).to(dev)

        Xr = torch.from_numpy(np.ascontiguousarray(Zreal, np.float32))
        Xb = torch.from_numpy(np.ascontiguousarray(Zblend, np.float32))
        self.real.pc.fit_leaves(Xr[:20000], jitter=cfg.leaf_jitter, seed=cfg.seed)
        self.blend.pc.fit_leaves(Xb[:20000], jitter=cfg.leaf_jitter, seed=cfg.seed + 1)

        params = list(self.real.pc.parameters()) + list(self.blend.pc.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = min(len(Xr), len(Xb))
        half = max(8, cfg.batch_size // 2)
        g = torch.Generator().manual_seed(cfg.seed)
        history: Dict[str, list] = {"gen": [], "disc": [], "val_auc": []}

        for ep in range(epochs):
            self.real.pc.train()
            self.blend.pc.train()
            pr, pb = torch.randperm(len(Xr), generator=g), torch.randperm(len(Xb), generator=g)
            tot_g = tot_d = seen = 0.0
            for i in range(0, n, half):
                xr = Xr[pr[i:i + half]].to(dev)
                xb = Xb[pb[i:i + half]].to(dev)
                opt.zero_grad(set_to_none=True)

                lr_r, lb_r = self.real.pc.log_prob(xr), self.blend.pc.log_prob(xr)
                lr_b, lb_b = self.real.pc.log_prob(xb), self.blend.pc.log_prob(xb)

                gen = -(lr_r.mean() + lb_b.mean())
                logits = torch.cat([lb_r - lr_r, lb_b - lr_b])
                y = torch.cat([torch.zeros(len(xr), device=dev),
                               torch.ones(len(xb), device=dev)])
                disc = F.binary_cross_entropy_with_logits(logits, y)

                (lam * gen + (1.0 - lam) * disc).backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                tot_g += float(gen) * len(xr)
                tot_d += float(disc) * len(xr)
                seen += len(xr)
            sched.step()
            history["gen"].append(tot_g / max(seen, 1))
            history["disc"].append(tot_d / max(seen, 1))

            if Zval_real is not None and Zval_blend is not None and (ep % 5 == 0
                                                                     or ep == epochs - 1):
                auc = self._val_auc(Zval_real, Zval_blend)
                history["val_auc"].append({"epoch": ep, "auc": auc})
                if verbose:
                    print(f"[hybrid λ={lam:.2f}] epoch {ep:3d}  gen {history['gen'][-1]:9.2f}"
                          f"  disc {history['disc'][-1]:.4f}  val(real vs blend) AUC {auc:.4f}",
                          flush=True)
        self.real.pc.eval()
        self.blend.pc.eval()
        return history

    @torch.no_grad()
    def _val_auc(self, Zval_real: np.ndarray, Zval_blend: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score

        lr_r, lb_r = self._log_probs(Zval_real[:1500])
        lr_b, lb_b = self._log_probs(Zval_blend[:1500])
        s = np.r_[lb_r - lr_r, lb_b - lr_b]
        y = np.r_[np.zeros(len(lr_r)), np.ones(len(lr_b))]
        return float(roc_auc_score(y, s))

    # ── scoring ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _log_probs(self, Z: np.ndarray, batch: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        dev = self.cfg.device
        X = torch.from_numpy(np.ascontiguousarray(Z, np.float32))
        lr, lb = [], []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(dev)
            lr.append(self.real.pc.log_prob(xb).cpu().numpy())
            lb.append(self.blend.pc.log_prob(xb).cpu().numpy())
        return np.concatenate(lr), np.concatenate(lb)

    @torch.no_grad()
    def patch_ratio(self, Z: np.ndarray, batch: int = 32,
                    chunk_rows: int = 1024) -> Dict[str, np.ndarray]:
        """
        Exact per-patch conditional log-ratio, plus the joint ratio.

        Memory note: this expands to (images x masks) rows, and the leaf layer
        materialises (rows, d*I, M).  At d = 6080 that is gigabytes per chunk,
        so `chunk_rows` is deliberately small and the loop backs off further on
        OOM rather than dying after the circuits are already trained.
        """
        dev = self.cfg.device
        keep, drop = self.real._patch_masks(dev)
        X = torch.from_numpy(np.ascontiguousarray(Z, np.float32))

        # probe ONCE on the first batch to find a size that fits, then run
        while batch > 4:
            try:
                xb = X[:batch].to(dev)
                self.real.pc.region_log_marginals(xb, drop, chunk_rows)
                del xb
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch = max(4, batch // 2)
                chunk_rows = max(256, chunk_rows // 2)
                print(f"[ratio] OOM -> batch {batch}, chunk_rows {chunk_rows}",
                      flush=True)

        joint, cond = [], []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(dev)
            lp_r = self.real.pc.log_prob(xb)
            lp_b = self.blend.pc.log_prob(xb)
            md_r = self.real.pc.region_log_marginals(xb, drop, chunk_rows)
            md_b = self.blend.pc.region_log_marginals(xb, drop, chunk_rows)
            # log p(z_p | z_-p) = log p(z) - log p(z_-p), per model
            c_r = lp_r[:, None] - md_r
            c_b = lp_b[:, None] - md_b
            cond.append((c_b - c_r).cpu().numpy())
            joint.append((lp_b - lp_r).cpu().numpy())
        return {"joint": np.concatenate(joint), "cond": np.concatenate(cond)}

    def calibrate(self, Zref: np.ndarray, batch: int = 64) -> None:
        q = self.patch_ratio(Zref, batch=batch)
        self.calib = {
            "cond_mean": q["cond"].mean(0), "cond_std": q["cond"].std(0) + 1e-6,
            "joint_mean": np.array([q["joint"].mean()]),
            "joint_std": np.array([q["joint"].std() + 1e-6]),
        }

    def score(self, Z: np.ndarray, batch: int = 64) -> Dict[str, np.ndarray]:
        q = self.patch_ratio(Z, batch=batch)
        if not self.calib:
            raise RuntimeError("calibrate() on real data before scoring")
        zc = (q["cond"] - self.calib["cond_mean"]) / self.calib["cond_std"]
        k = max(1, int(round(self.cfg.topq * self.P)))
        return {
            "ratio_joint": q["joint"],
            "ratio_patch_max": zc.max(1),
            "ratio_patch_topq": np.sort(zc, axis=1)[:, -k:].mean(1),
            "ratio_patch_mean": zc.mean(1),
            "_patch_ratio_z": zc,
        }

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"real": self.real.pc.state_dict(),
                    "blend": self.blend.pc.state_dict(),
                    "cfg": self.cfg.__dict__,
                    "calib": {k: v for k, v in self.calib.items()},
                    "layout": {"gh": self.gh, "gw": self.gw, "C": self.C}}, path)

    def load(self, path: str | Path, Zref: np.ndarray,
             structure_cache: Optional[str | Path] = None) -> "PCRatioDetector":
        blob = torch.load(path, map_location=self.cfg.device, weights_only=False)
        if self.real.pc is None:
            self.real.build(Zref, structure_cache=structure_cache)
        self.real.pc.load_state_dict(blob["real"])
        from ..circuits.einsum_pc import EinsumPC

        self.blend.region_graph = self.real.region_graph
        self.blend.patch_regions = self.real.patch_regions
        self.blend.pc = EinsumPC(
            self.real.region_graph,
            n_sum_components=self.cfg.n_sum_components,
            n_input_components=self.cfg.n_input_components,
            leaf_components=self.cfg.leaf_components,
            weight_jitter=self.cfg.weight_jitter, seed=self.cfg.seed + 1,
        ).to(self.cfg.device)
        self.blend.pc.load_state_dict(blob["blend"])
        self.calib = {k: np.asarray(v) for k, v in blob["calib"].items()}
        self.real.pc.eval()
        self.blend.pc.eval()
        return self
