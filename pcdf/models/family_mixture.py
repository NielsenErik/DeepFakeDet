"""
An exact mixture of pseudo-forgery densities, and the queries it makes possible.

WHY A MIXTURE, AND WHY IT HAS TO BE A CIRCUIT
---------------------------------------------
The measured bottleneck of this project is the pseudo-fake distribution, not
the model:

  * the discriminative objective SATURATES on self-blends (loss 0.0000,
    real-vs-blend AUC 0.9996) while real-forgery AUC sits at 0.827, so there is
    no gradient left to extract;
  * a single self-blend family deviates from real in ONE direction — rougher —
    which is why the graphics-rendered manipulations invert (Face2Face 0.45,
    FaceSwap 0.36) while the neural ones are detected;
  * and `scripts/shortcut_audit.py` showed the single family also leaks a global
    compression cue that real forgeries do not have.

The response is to make p_blend a MIXTURE over forgery mechanisms

    p_mix(z) = Σ_f π_f · p_f(z)

with one circuit per family over a SHARED region graph.  Three things then hold
that hold for no other model in the comparison:

  1. `p_mix` is itself a valid circuit — a sum node over components with
     identical scope — so it stays smooth, decomposable, structured
     decomposable and exactly normalized.  The mixture is not an ensemble
     heuristic bolted on top; it is one circuit.
  2. The score stays an exact log-likelihood ratio,
     `s(z) = log p_mix(z) − log p_real(z)`, so it is a log-odds and can be
     calibrated rather than merely ranked.
  3. The MIXTURE POSTERIOR is available in closed form,

         P(f | z) = π_f p_f(z) / Σ_g π_g p_g(z)

     which answers a question no baseline in this study can even express: given
     a face the model believes is forged, WHICH MECHANISM does it look like?
     A flow gives no marginals and no components; a memory bank gives no
     probabilities at all.  Because every p_f is exactly normalized over the
     same scope, the posterior is a genuine probability rather than a softmax
     over incomparable scores.

The per-region version of the same quantity localizes the mechanism:

    P(f | z_R) ∝ π_f p_f(z_R)

with each p_f(z_R) an exact marginal of the region, which is what makes the
statement "the cheek is explained by rendering, the jawline by blending"
well defined.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .density_pc import PCConfig, PCDetector


class FamilyMixtureRatio:
    """`log Σ_f π_f p_f(z) − log p_real(z)`, every term exact."""

    def __init__(self, grid_h: int, grid_w: int, n_channels: int,
                 families: Sequence[str], cfg: Optional[PCConfig] = None):
        self.gh, self.gw, self.C = grid_h, grid_w, n_channels
        self.P = grid_h * grid_w
        self.d = grid_h * grid_w * n_channels
        self.families = list(families)
        self.cfg = cfg or PCConfig()
        self.real = PCDetector(grid_h, grid_w, n_channels, self.cfg)
        self.comp: Dict[str, PCDetector] = {}
        self.log_pi = np.log(np.ones(len(families)) / len(families))
        self.calib: Dict[str, np.ndarray] = {}

    # ── fitting ──────────────────────────────────────────────────────────

    def fit(self, Zreal: np.ndarray, Zfam: Dict[str, np.ndarray],
            Zval_real: Optional[np.ndarray] = None,
            structure_cache: Optional[str | Path] = None,
            verbose: bool = True) -> None:
        """
        One region graph, learned once on real data, reused by every component.

        Sharing the structure is what makes the components comparable: any
        difference between `p_f(z)` and `p_g(z)` comes from parameters, never
        from a different scope decomposition, so the posterior over families is
        a statement about the data and not about how each model happened to
        factorise.
        """
        from ..circuits.einsum_pc import EinsumPC

        if verbose:
            print(f"[mix] p_real on {len(Zreal)} real crops", flush=True)
        self.real.fit(Zreal, Zval_real, structure_cache=structure_cache,
                      verbose=verbose)

        for i, fam in enumerate(self.families):
            Z = Zfam[fam]
            det = PCDetector(self.gh, self.gw, self.C, self.cfg)
            det.region_graph = self.real.region_graph
            det.patch_regions = self.real.patch_regions
            det.structure_info = self.real.structure_info
            det.pc = EinsumPC(
                self.real.region_graph,
                n_sum_components=self.cfg.n_sum_components,
                n_input_components=self.cfg.n_input_components,
                leaf_components=self.cfg.leaf_components,
                weight_jitter=self.cfg.weight_jitter,
                seed=self.cfg.seed + 1 + i).to(self.cfg.device)
            if verbose:
                print(f"[mix] p_{fam} on {len(Z)} crops", flush=True)
            det.fit(Z, None, structure_cache=None, verbose=verbose)
            self.comp[fam] = det

        # mixture weights proportional to how much data each family contributed;
        # with equal extraction budgets this is uniform, and stays explicit so a
        # deliberately unbalanced prior can be set later
        n = np.array([len(Zfam[f]) for f in self.families], dtype=np.float64)
        self.log_pi = np.log(n / n.sum())

    # ── scoring ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def component_log_probs(self, Z: np.ndarray, batch: int = 256) -> np.ndarray:
        """(n, n_families + 1): each family's log p_f, then log p_real last."""
        X = torch.from_numpy(np.ascontiguousarray(Z, np.float32))
        out = []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(self.cfg.device)
            cols = [self.comp[f].pc.log_prob(xb) for f in self.families]
            cols.append(self.real.pc.log_prob(xb))
            out.append(torch.stack(cols, 1).cpu().numpy())
        return np.concatenate(out)

    def score(self, Z: np.ndarray, batch: int = 256) -> Dict[str, np.ndarray]:
        L = self.component_log_probs(Z, batch)
        lf, lr = L[:, :-1], L[:, -1]
        w = lf + self.log_pi[None, :]
        mix = torch.logsumexp(torch.from_numpy(w), dim=1).numpy()
        post = np.exp(w - mix[:, None])
        return {
            "ratio_mixture": mix - lr,                    # the detector
            "ratio_best_family": (w.max(1) - lr),         # max instead of sum
            "family_posterior": post,                     # (n, n_families)
            "log_p_real": lr,
            "log_p_mixture": mix,
        }

    # ── the query no baseline can answer ─────────────────────────────────

    @torch.no_grad()
    def region_family_posterior(self, Z: np.ndarray, batch: int = 32,
                                chunk_rows: int = 512) -> np.ndarray:
        """
        (n, P, n_families) — exact posterior over forgery mechanisms per REGION.

        `p_f(z_R)` is an exact marginal: the region is a node of the shared
        region graph, so marginalising the rest is one upward pass with the
        other leaves set to 1.  Normalising across families at each region gives
        a per-region mechanism attribution whose values are probabilities.
        """
        keep, drop = self.real._patch_masks(self.cfg.device)
        X = torch.from_numpy(np.ascontiguousarray(Z, np.float32))
        out = []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(self.cfg.device)
            # log p_f(z_R): marginal over the region's OWN scope, i.e. drop
            # everything else -> `keep` selects the region
            cols = [self.comp[f].pc.region_log_marginals(xb, keep, chunk_rows)
                    for f in self.families]
            M = torch.stack(cols, -1)                     # (b, P, F)
            M = M + torch.from_numpy(self.log_pi).to(M).view(1, 1, -1)
            out.append(torch.softmax(M, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def calibrate(self, Zref: np.ndarray) -> None:
        s = self.score(Zref)["ratio_mixture"]
        self.calib = {"mean": np.array([s.mean()]),
                      "std": np.array([s.std() + 1e-6])}

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"real": self.real.pc.state_dict(),
                    "components": {f: d.pc.state_dict()
                                   for f, d in self.comp.items()},
                    "families": self.families, "log_pi": self.log_pi,
                    "cfg": self.cfg.__dict__, "calib": self.calib,
                    "layout": {"gh": self.gh, "gw": self.gw, "C": self.C}}, path)


# ── calibration metrics (contribution C5) ───────────────────────────────────

def expected_calibration_error(p: np.ndarray, y: np.ndarray,
                               n_bins: int = 15) -> Dict[str, float]:
    """
    ECE and MCE over equal-width probability bins.

    Reported because the ratio is a LOG-ODDS, not an arbitrary score: if both
    densities were correct, `sigmoid(s)` would be the posterior probability of
    forgery.  Whether it actually is, is testable, and it is the difference
    between a detector that ranks and one whose threshold means something.
    """
    p = np.clip(p, 1e-7, 1 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = mce = 0.0
    bins = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        gap = abs(conf - acc)
        ece += m.mean() * gap
        mce = max(mce, gap)
        bins.append({"bin": b, "n": int(m.sum()), "confidence": conf,
                     "accuracy": acc})
    return {"ece": float(ece), "mce": float(mce), "bins": bins}


def risk_coverage(score: np.ndarray, y: np.ndarray,
                  n_points: int = 20) -> List[Dict[str, float]]:
    """
    Selective prediction: abstain on the least confident fraction and report the
    error on what remains.

    Confidence here is |s| — distance from the decision boundary in nats — which
    is meaningful precisely because s is an exact log-ratio.  A forensic user
    cares about this curve far more than about a single AUC: it says what
    accuracy is available if the system is allowed to say "I don't know".
    """
    conf = np.abs(score)
    order = np.argsort(-conf)
    pred = (score > 0).astype(int)
    correct = (pred == y).astype(float)[order]
    out = []
    for cov in np.linspace(1.0, 0.1, n_points):
        k = max(1, int(round(cov * len(correct))))
        out.append({"coverage": float(k / len(correct)),
                    "risk": float(1.0 - correct[:k].mean())})
    return out
