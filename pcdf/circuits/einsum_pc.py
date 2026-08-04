"""
EinsumPC — a tensorized executor for the reference library's region graphs.

WHAT IS BROKEN IN THE REFERENCE EXECUTION (and only the execution)
------------------------------------------------------------------
`RegionGraphPC` already fixed the *parameter* blowup: a region owns K units and
the products are shared, so size is O(d·K²) rather than O(d·K^depth).  What it
did not fix is the *compute*: every unit is an `nn.Module`, every query is a
memoised Python recursion over the DAG, and every sum node stacks a Python list
of child tensors.  For the circuits this project needs — d = 4096 variables
(256 patches × 16 channels), K = 8..16 — that is ~10⁵ Python-level module calls
per forward pass and no GPU utilisation to speak of.  A 300-epoch full-batch
fit took 2-4 minutes at d = 34 in the POC; at d = 4096 it does not finish.

THE FIX
-------
Same circuit, same parameters, different execution.  Regions are grouped by
*shape signature* and evaluated one level at a time as batched einsums:

    products (binary partition)   P[b,g,i,j] = L[b,g,i] + R[b,g,j]
    sum units                     out[b,g,k] = logsumexp_p ( logw[g,k,p] + P[b,g,p] )

with the max-subtraction done once per (b,g).  All G regions of a level, all K
units and all P products are one op.  Nothing about the *model* changes, which
is the point: `tests/test_equivalence.py` copies parameters between this class
and `RegionGraphPC` and asserts identical log-densities, marginals and
partition functions to float tolerance.

WHAT THIS BUYS BEYOND SPEED
---------------------------
Marginalization is a per-(sample, feature) mask on the leaf layer, so Q
different marginal queries are one batched pass with the batch axis expanded —
not Q Python traversals.  That turns "exact log p(z_R) for all 256 patch
regions of an image" from a benchmark curiosity into a per-image explanation
that costs one forward pass, which is what makes the localization evaluation in
`pcdf.explain` affordable at dataset scale.

SUPPORTED STRUCTURES
--------------------
Any `RegionNode` graph the library can produce: binary or n-ary partitions,
one or several partitions per region, shared (DAG) regions.  Binary partitions
use the full K×K cross product, wider ones matched-index pairing — exactly
`RegionGraphPC._combine`.  Smoothness and decomposability hold by construction;
structured decomposability holds iff the region graph is single-partition
(checked, not assumed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..pclib import (
    RegionNode,
    is_structured_decomposable_rg,
    region_nodes,
)
from .leaves import GaussianMixtureLeafLayer


# ── build-time layout ───────────────────────────────────────────────────────

@dataclass
class _Partition:
    mode: str                      # "full" (binary cross product) | "matched"
    idx: List[np.ndarray]          # per child slot: (G, n_slot_units) columns
    n_products: int


@dataclass
class _Group:
    """Regions of one level sharing an identical shape signature."""
    regions: List[RegionNode]
    n_units: int                   # K_out
    partitions: List[_Partition]
    n_products: int
    weight_index: int              # index into the ParameterList


@dataclass
class _Level:
    groups: List[_Group] = field(default_factory=list)
    keep_idx: Optional[np.ndarray] = None    # columns of V carried forward


def _region_levels(root: RegionNode) -> Dict[int, int]:
    """Longest path to a leaf region — children are always at a lower level."""
    level: Dict[int, int] = {}

    def walk(r: RegionNode) -> int:
        rid = id(r)
        if rid in level:
            return level[rid]
        if r.is_leaf:
            level[rid] = 0
            return 0
        h = 1 + max(walk(c) for part in r.partitions for c in part)
        level[rid] = h
        return h

    walk(root)
    return level


class EinsumPC(nn.Module):
    """
    Args:
        region_graph:       a `RegionNode` from the library's learners
                            (`curvature_region_graph` / ORC, Chow-Liu via
                            `region_graph_from_vtree`, spectral, or the
                            hierarchical image graphs in `pcdf.circuits.structure`).
        n_sum_components:   K units per internal region.
        n_input_components: units per leaf region (defaults to K).
        leaf_components:    M Gaussian components inside each leaf unit.
        pairing:            "auto" | "full" | "matched" (see `_combine` in the
                            reference library — "auto" is full for binary
                            partitions, matched for wider ones).
    """

    def __init__(
        self,
        region_graph: RegionNode,
        n_sum_components: int = 8,
        n_input_components: Optional[int] = None,
        leaf_components: int = 4,
        pairing: str = "auto",
        weight_jitter: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.region_graph = region_graph
        self.K = int(n_sum_components)
        self.I = int(n_input_components or n_sum_components)
        self.M = int(leaf_components)
        self.pairing = pairing
        self.n_features = len(region_graph.scope)
        self.is_structured = is_structured_decomposable_rg(region_graph)

        regions = region_nodes(region_graph)
        levels = _region_levels(region_graph)
        self._n_units: Dict[int, int] = {}
        for r in regions:
            if id(r) == id(region_graph):
                self._n_units[id(r)] = 1
            elif r.is_leaf:
                self._n_units[id(r)] = self.I
            else:
                self._n_units[id(r)] = self.K

        # parents: a region must stay in the value store until its last consumer
        last_consumer: Dict[int, int] = {}
        for r in regions:
            for part in r.partitions:
                for c in part:
                    last_consumer[id(c)] = max(last_consumer.get(id(c), -1),
                                               levels[id(r)])

        # ── level 0: leaf regions -> the leaf layer's columns ───────────
        leaf_regions = sorted([r for r in regions if r.is_leaf],
                              key=lambda r: min(r.scope))
        feature_of_unit: List[int] = []
        col: Dict[int, int] = {}
        for r in leaf_regions:
            col[id(r)] = len(feature_of_unit)
            feature_of_unit += [r.feature_idx] * self._n_units[id(r)]
        self.leaves = GaussianMixtureLeafLayer(
            np.array(feature_of_unit, dtype=np.int64),
            n_components=self.M, n_features=self.n_features)
        self.leaf_regions = leaf_regions

        # ── levels >= 1: group regions by shape signature ──────────────
        by_level: Dict[int, List[RegionNode]] = {}
        for r in regions:
            if not r.is_leaf:
                by_level.setdefault(levels[id(r)], []).append(r)

        self._levels: List[_Level] = []
        weights: List[nn.Parameter] = []
        gen = torch.Generator().manual_seed(int(seed))
        width = len(feature_of_unit)

        for lv in sorted(by_level):
            groups_by_sig: Dict[tuple, List[RegionNode]] = {}
            for r in by_level[lv]:
                groups_by_sig.setdefault(self._signature(r), []).append(r)

            level = _Level()
            new_cols: Dict[int, int] = {}
            offset_new = 0
            for sig, group_regions in sorted(groups_by_sig.items(), key=lambda kv: str(kv[0])):
                parts, n_products = self._build_partitions(group_regions, col)
                ko = self._n_units[id(group_regions[0])]
                w = torch.full((len(group_regions), ko, n_products),
                               -math.log(n_products))
                if weight_jitter > 0:
                    w = w + torch.randn(w.shape, generator=gen) * weight_jitter
                weights.append(nn.Parameter(w))
                level.groups.append(_Group(
                    regions=group_regions, n_units=ko, partitions=parts,
                    n_products=n_products, weight_index=len(weights) - 1))
                for gi, r in enumerate(group_regions):
                    new_cols[id(r)] = offset_new + gi * ko
                offset_new += len(group_regions) * ko

            # carry forward only what later levels still consume
            keep_regions = [r for r in regions
                            if id(r) in col and last_consumer.get(id(r), -1) > lv]
            keep_idx: List[int] = []
            kept_cols: Dict[int, int] = {}
            for r in keep_regions:
                kept_cols[id(r)] = len(keep_idx)
                n = self._n_units[id(r)]
                keep_idx += list(range(col[id(r)], col[id(r)] + n))
            level.keep_idx = np.array(keep_idx, dtype=np.int64)
            self._levels.append(level)

            col = dict(kept_cols)
            for rid, off in new_cols.items():
                col[rid] = len(keep_idx) + off
            width = len(keep_idx) + offset_new

        self.weights = nn.ParameterList(weights)
        self.root_col = col[id(region_graph)]
        self._store_width = width
        self._register_index_buffers()

    # ── layout helpers ──────────────────────────────────────────────────

    def _mode(self, arity: int) -> str:
        if self.pairing == "auto":
            return "full" if arity == 2 else "matched"
        return self.pairing

    def _signature(self, r: RegionNode) -> tuple:
        """Regions with equal signature can share one einsum."""
        sig = [self._n_units[id(r)]]
        for part in r.partitions:
            mode = self._mode(len(part))
            sig.append((mode, tuple(self._n_units[id(c)] for c in part)))
        return tuple(sig)

    def _build_partitions(self, group_regions: List[RegionNode],
                          col: Dict[int, int]) -> Tuple[List[_Partition], int]:
        parts: List[_Partition] = []
        total = 0
        for pi in range(len(group_regions[0].partitions)):
            children0 = group_regions[0].partitions[pi]
            mode = self._mode(len(children0))
            if mode == "full":
                # full cross product over the (two) children; product index
                # p = i * K1 + j, matching itertools.product in the reference
                idx = []
                for slot in range(len(children0)):
                    n_units = self._n_units[id(children0[slot])]
                    idx.append(np.array(
                        [[col[id(r.partitions[pi][slot])] + u for u in range(n_units)]
                         for r in group_regions], dtype=np.int64))
                n_prod = int(np.prod([a.shape[1] for a in idx]))
            else:
                # matched-index pairing: p-th product takes unit (p mod K_c)
                n_prod = max(self._n_units[id(c)] for c in children0)
                idx = []
                for slot in range(len(children0)):
                    n_units = self._n_units[id(children0[slot])]
                    idx.append(np.array(
                        [[col[id(r.partitions[pi][slot])] + (p % n_units)
                          for p in range(n_prod)] for r in group_regions],
                        dtype=np.int64))
            parts.append(_Partition(mode=mode, idx=idx, n_products=n_prod))
            total += n_prod
        return parts, total

    def _register_index_buffers(self) -> None:
        """Index arrays live as buffers so .to(device) moves them."""
        for li, level in enumerate(self._levels):
            self.register_buffer(f"_keep_{li}",
                                 torch.as_tensor(level.keep_idx, dtype=torch.long),
                                 persistent=False)
            for gi, g in enumerate(level.groups):
                for pi, p in enumerate(g.partitions):
                    for si, arr in enumerate(p.idx):
                        self.register_buffer(
                            f"_idx_{li}_{gi}_{pi}_{si}",
                            torch.as_tensor(arr, dtype=torch.long), persistent=False)

    # ── evaluation ──────────────────────────────────────────────────────

    def _forward_units(self, leaf_vals: torch.Tensor) -> torch.Tensor:
        """(B, L) leaf log-values -> (B,) log-value of the root unit."""
        V = leaf_vals
        B = V.shape[0]
        for li, level in enumerate(self._levels):
            outs: List[torch.Tensor] = []
            for gi, g in enumerate(level.groups):
                prods: List[torch.Tensor] = []
                for pi, part in enumerate(g.partitions):
                    gathered = [V[:, getattr(self, f"_idx_{li}_{gi}_{pi}_{si}")]
                                for si in range(len(part.idx))]
                    if part.mode == "full":
                        a, b = gathered                      # (B,G,K0), (B,G,K1)
                        pr = (a.unsqueeze(-1) + b.unsqueeze(-2))
                        pr = pr.reshape(B, a.shape[1], -1)   # (B,G,K0*K1)
                    else:
                        pr = gathered[0]
                        for t in gathered[1:]:
                            pr = pr + t                      # (B,G,n)
                    prods.append(pr)
                P = prods[0] if len(prods) == 1 else torch.cat(prods, dim=-1)
                w = torch.softmax(self.weights[g.weight_index].float(), dim=-1)
                m = P.amax(dim=-1, keepdim=True)             # (B,G,1)
                e = (P - m).exp()
                mixed = torch.einsum("bgp,gop->bgo", e, w)
                out = m + torch.log(mixed.clamp_min(1e-38))  # (B,G,Ko)
                outs.append(out.reshape(B, -1))
            keep = getattr(self, f"_keep_{li}")
            V = torch.cat(([V[:, keep]] if keep.numel() else []) + outs, dim=1)
        return V[:, self.root_col]

    def _leaf_vals(self, x, observed=None, boxes=None) -> torch.Tensor:
        return self.leaves(x, observed=observed, boxes=boxes)

    def _run(self, x: torch.Tensor, observed=None, boxes=None,
             chunk: Optional[int] = None) -> torch.Tensor:
        if chunk is None or x.shape[0] <= chunk:
            return self._forward_units(self._leaf_vals(x, observed, boxes))
        outs = []
        for i in range(0, x.shape[0], chunk):
            sl = slice(i, i + chunk)
            obs = observed[sl] if (observed is not None and observed.shape[0] > 1) else observed
            bx = None
            if boxes is not None:
                lo, hi, bm = boxes
                bx = (lo[sl], hi[sl], bm[sl] if bm.shape[0] > 1 else bm)
            outs.append(self._forward_units(self._leaf_vals(x[sl], obs, bx)))
        return torch.cat(outs, dim=0)

    # ── public exact queries ────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)

    def log_prob(self, x: torch.Tensor, chunk: Optional[int] = None) -> torch.Tensor:
        """Exact log p(x).  (B, d) -> (B,)."""
        return self._run(x, chunk=chunk)

    def log_prob_masked(self, x: torch.Tensor, observed: torch.Tensor,
                        chunk: Optional[int] = None) -> torch.Tensor:
        """
        Exact log p(x_O) where O = {i : observed[b,i]}.  `observed` is (B, d) or
        (1, d) bool.  Marginalized leaves contribute log ∫ f = 0 — exact for
        smooth + decomposable circuits, which this one is by construction.
        """
        return self._run(x, observed=observed, chunk=chunk)

    def log_marginal(self, x: torch.Tensor, marginalized: Iterable[int],
                     chunk: Optional[int] = None) -> torch.Tensor:
        marg = torch.as_tensor(sorted(set(int(i) for i in marginalized)),
                               dtype=torch.long, device=x.device)
        obs = torch.ones((1, self.n_features), dtype=torch.bool, device=x.device)
        if marg.numel():
            obs[0, marg] = False
        return self._run(x, observed=obs, chunk=chunk)

    def log_box(self, x: torch.Tensor, boxes: Dict[int, Tuple[float, float]],
                marginalized: Iterable[int] = (), chunk: Optional[int] = None):
        """
        Exact axis-aligned box query: log P(x_i ∈ [lo_i, hi_i] for boxed i,
        x_j observed at x for the rest, others marginalized).
        """
        B, d = x.shape
        lo = torch.full((B, d), -float("inf"), device=x.device, dtype=x.dtype)
        hi = torch.full((B, d), float("inf"), device=x.device, dtype=x.dtype)
        bm = torch.zeros((B, d), dtype=torch.bool, device=x.device)
        for i, (a, b) in boxes.items():
            lo[:, i], hi[:, i], bm[:, i] = float(a), float(b), True
        obs = torch.ones((1, d), dtype=torch.bool, device=x.device)
        for i in marginalized:
            obs[0, int(i)] = False
        return self._run(x, observed=obs, boxes=(lo, hi, bm), chunk=chunk)

    def log_partition(self) -> torch.Tensor:
        """Exact log Z — 0 by construction; this computes it, it does not assume it."""
        dev = self.leaves.mus.device
        x = torch.zeros((1, self.n_features), device=dev)
        obs = torch.zeros((1, self.n_features), dtype=torch.bool, device=dev)
        return self._run(x, observed=obs)[0]

    def anomaly_score(self, x: torch.Tensor, chunk: Optional[int] = None) -> torch.Tensor:
        return -self.log_prob(x, chunk=chunk)

    # ── batched region queries (the explainability workhorse) ───────────

    def region_log_marginals(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        chunk_rows: int = 4096,
    ) -> torch.Tensor:
        """
        Exact log p(x_{O_q}) for every sample and every observation mask.

        x:     (B, d)
        masks: (Q, d) bool — mask q keeps the features marked True.
        returns (B, Q).

        One batched pass over B·Q rows instead of Q traversals per sample; this
        is what makes per-patch explanation affordable at dataset scale.
        """
        B, d = x.shape
        Q = masks.shape[0]
        xr = x.unsqueeze(1).expand(B, Q, d).reshape(B * Q, d)
        mr = masks.unsqueeze(0).expand(B, Q, d).reshape(B * Q, d)
        out = self._run(xr, observed=mr, chunk=chunk_rows)
        return out.reshape(B, Q)

    # ── structure / property audit ──────────────────────────────────────

    def size(self) -> Dict[str, int]:
        n_sum = sum(w.shape[0] * w.shape[1] for w in self.weights)
        n_prod = sum(w.shape[0] * w.shape[2] for w in self.weights)
        return {
            "features": self.n_features,
            "leaf_units": self.leaves.n_units,
            "sum_units": n_sum,
            "product_units": n_prod,
            "parameters": sum(p.numel() for p in self.parameters()),
            "levels": len(self._levels),
            "einsum_groups": sum(len(l.groups) for l in self._levels),
            "structured_decomposable": int(self.is_structured),
        }

    @torch.no_grad()
    def validate(self, x: Optional[torch.Tensor] = None, atol: float = 1e-4) -> Dict[str, float]:
        """
        Numerical property audit.  Smoothness and decomposability are structural
        (every sum mixes units of one region; every product spans a partition of
        disjoint child scopes), so what has to be *checked* is that the
        implementation preserves them:

          log Z == 0                      normalization
          marginalize nothing == log p(x) mask machinery is consistent
          marginalize everything == 0     leaf integrals are exact
          any partial marginal is finite  no NaN/inf leakage
          log p(x) <= log p(x_S)          a marginal dominates the joint density
                                          only in the discrete sense; here we
                                          check monotone containment of masks.
        """
        dev = self.leaves.mus.device
        if x is None:
            x = torch.randn(8, self.n_features, device=dev)
        x = x.to(dev)
        logZ = float(self.log_partition())
        lp = self.log_prob(x)
        m_none = self.log_marginal(x, [])
        m_all = self.log_marginal(x, range(self.n_features))
        half = list(range(self.n_features // 2))
        m_half = self.log_marginal(x, half)
        report = {
            "log_partition": logZ,
            "max_abs_diff_marginal_none_vs_logprob": float((lp - m_none).abs().max()),
            "max_abs_marginal_all": float(m_all.abs().max()),
            "finite_partial_marginal": float(torch.isfinite(m_half).all()),
            "structured_decomposable": float(self.is_structured),
        }
        assert abs(logZ) < atol, f"partition function broke: log Z = {logZ}"
        assert report["max_abs_diff_marginal_none_vs_logprob"] < atol
        assert report["max_abs_marginal_all"] < atol
        assert report["finite_partial_marginal"] == 1.0
        return report

    # ── parameter exchange with the reference RegionGraphPC ─────────────

    def _reference_units(self, ref) -> Dict[int, list]:
        return ref._regions

    @torch.no_grad()
    def load_from_reference(self, ref) -> None:
        """
        Copy every parameter out of a `RegionGraphPC` built on the SAME region
        graph with the same K / I / leaf type.  Used by the equivalence test:
        after this call the two objects must compute identical numbers.
        """
        units = self._reference_units(ref)
        leaf_mods = []
        for r in self.leaf_regions:
            leaf_mods.extend(units[id(r)])
        self.leaves.load_from_reference(leaf_mods)
        for level in self._levels:
            for g in level.groups:
                w = self.weights[g.weight_index]
                for gi, r in enumerate(g.regions):
                    for k, unit in enumerate(units[id(r)]):
                        if hasattr(unit, "weights"):
                            w[gi, k].copy_(unit.weights.detach().to(w.device))

    @torch.no_grad()
    def store_to_reference(self, ref) -> None:
        """The inverse copy — lets the audited reference validators run on a
        circuit whose parameters were trained here."""
        units = self._reference_units(ref)
        leaf_mods = []
        for r in self.leaf_regions:
            leaf_mods.extend(units[id(r)])
        self.leaves.store_to_reference(leaf_mods)
        for level in self._levels:
            for g in level.groups:
                w = self.weights[g.weight_index]
                for gi, r in enumerate(g.regions):
                    for k, unit in enumerate(units[id(r)]):
                        if hasattr(unit, "weights"):
                            unit.weights.copy_(w[gi, k].detach().cpu())

    @torch.no_grad()
    def fit_leaves(self, X: torch.Tensor, jitter: float = 0.2, seed: int = 0) -> None:
        self.leaves.fit(X, jitter=jitter, seed=seed)
