"""
Probabilistic circuit components and classes — the single components file
required by CLAUDE.md.

Contents
--------
1.  Vtree data structures (VtreeLeaf / VtreeInternal), JSON save/load,
    inspection utilities, LCA-depth matrix, random vtree generators.
2.  Structure sources (Stage 0b): random baseline, single-task extraction,
    consensus vtree via the co-grouping matrix + hierarchical clustering.
3.  Leaf (input) units: GaussianLeaf, CategoricalLeaf, GaussianMixtureLeaf,
    and the heavy-tailed InputNode (Gaussian/Laplace/Student-t mixture).
4.  Inner nodes: SumNode (monotone, normalized), ProductNode, plus the legacy
    ClassifierNode.
5.  DensityPC — a structured-decomposable density estimator built top-down
    from a vtree.  Smoothness + decomposability hold by construction.
5b. RegionGraphPC — the SAME model laid out as a region-graph DAG: O(d·K²)
    instead of DensityPC's O(d·K^depth), which is what makes wide inputs
    (windowed multivariate time series, high-dim latents) representable at
    all.  All four properties are preserved; the validators need no changes.
6.  Exact inference: log-density, marginals, axis-aligned BOX queries (the
    survival / right-censoring query), MPE, partition function.  Every
    routine is memoised by node identity, so it is linear in the circuit DAG.
7.  The four property validators (smoothness, decomposability, determinism,
    structured decomposability) — each raises AssertionError loudly.
8.  Optional SOS mode: SquaredPC — subtractive mixtures with real-valued sum
    weights, normalized exactly via the squared-circuit pairwise construction
    (Loconte et al., AAAI 2025).  Layered on top of the base leaves, not a
    replacement for the monotone path.
9.  Legacy PCNet (kept for reference / ablation).

Exactness invariant: every density object in this file is exactly normalized.
The input-dependent (attentional) weights and the leaf temperature gate from
the original code are excluded because they break normalization and therefore
exact marginals.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Vtree
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(eq=True)
class VtreeLeaf:
    feature_idx: int

    @property
    def scope(self) -> FrozenSet[int]:
        return frozenset({self.feature_idx})


@dataclass(eq=True)
class VtreeInternal:
    left: "VtreeNode"
    right: "VtreeNode"

    @property
    def scope(self) -> FrozenSet[int]:
        return self.left.scope | self.right.scope

    @property
    def left_scope(self) -> FrozenSet[int]:
        return self.left.scope

    @property
    def right_scope(self) -> FrozenSet[int]:
        return self.right.scope


VtreeNode = Union[VtreeLeaf, VtreeInternal]


def vtree_depth(node: VtreeNode) -> int:
    if isinstance(node, VtreeLeaf):
        return 0
    return 1 + max(vtree_depth(node.left), vtree_depth(node.right))


def vtree_leaves(node: VtreeNode) -> List[int]:
    """Feature indices of all leaves, in left-to-right order."""
    if isinstance(node, VtreeLeaf):
        return [node.feature_idx]
    return vtree_leaves(node.left) + vtree_leaves(node.right)


def vtree_nodes(node: VtreeNode) -> List[VtreeNode]:
    """All nodes of the vtree (pre-order)."""
    if isinstance(node, VtreeLeaf):
        return [node]
    return [node] + vtree_nodes(node.left) + vtree_nodes(node.right)


# ─── Serialization ──────────────────────────────────────────────────────────

def _vtree_to_dict(node: VtreeNode) -> dict:
    if isinstance(node, VtreeLeaf):
        return {"type": "leaf", "feature_idx": node.feature_idx}
    return {
        "type": "internal",
        "scope": sorted(node.scope),  # sorted list, never a raw set
        "left": _vtree_to_dict(node.left),
        "right": _vtree_to_dict(node.right),
    }


def _vtree_from_dict(d: dict) -> VtreeNode:
    if d["type"] == "leaf":
        return VtreeLeaf(int(d["feature_idx"]))
    return VtreeInternal(_vtree_from_dict(d["left"]), _vtree_from_dict(d["right"]))


def save_vtree(node: VtreeNode, path: str) -> None:
    with open(path, "w") as f:
        json.dump(_vtree_to_dict(node), f, indent=2)


def load_vtree(path: str) -> VtreeNode:
    with open(path) as f:
        return _vtree_from_dict(json.load(f))


# ─── LCA depth matrix ───────────────────────────────────────────────────────

def lca_depth_matrix(root: VtreeNode) -> Dict[Tuple[int, int], int]:
    """
    Map each feature pair (i, j) — both orderings — to the depth of their
    lowest common ancestor in the vtree (root depth = 0).  Deeper LCA means
    the pair was kept together longer down the partition hierarchy.
    """
    M: Dict[Tuple[int, int], int] = {}

    def _walk(node: VtreeNode, depth: int) -> List[int]:
        if isinstance(node, VtreeLeaf):
            return [node.feature_idx]
        left = _walk(node.left, depth + 1)
        right = _walk(node.right, depth + 1)
        for i in left:
            for j in right:
                M[(i, j)] = depth
                M[(j, i)] = depth
        return left + right

    _walk(root, 0)
    return M


# ─── Random vtree generators ────────────────────────────────────────────────

def random_balanced_vtree(features: Sequence[int], seed: int = 0) -> VtreeNode:
    """Shuffle features, then split at the midpoint recursively (balanced)."""
    feats = list(features)
    random.Random(seed).shuffle(feats)

    def _build(fs: List[int]) -> VtreeNode:
        if len(fs) == 1:
            return VtreeLeaf(fs[0])
        mid = len(fs) // 2
        return VtreeInternal(_build(fs[:mid]), _build(fs[mid:]))

    return _build(feats)


def random_unbalanced_vtree(features: Sequence[int], seed: int = 0) -> VtreeNode:
    """Shuffle features, then split at a random point at each level (skewed)."""
    rng = random.Random(seed)
    feats = list(features)
    rng.shuffle(feats)

    def _build(fs: List[int]) -> VtreeNode:
        if len(fs) == 1:
            return VtreeLeaf(fs[0])
        cut = rng.randint(1, len(fs) - 1)
        return VtreeInternal(_build(fs[:cut]), _build(fs[cut:]))

    return _build(feats)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Structure sources (Stage 0b)
# ═══════════════════════════════════════════════════════════════════════════

def cogroup_matrix(vtrees: Sequence[VtreeNode], n_features: int) -> np.ndarray:
    """
    Co-grouping matrix M[i, j] = average normalized LCA depth of (i, j) across
    the source vtrees.  Values in [0, 1]; higher = the pair tends to stay
    together deeper in the partition hierarchy.  Diagonal is 1.
    """
    M = np.zeros((n_features, n_features))
    for vt in vtrees:
        depth_max = max(vtree_depth(vt), 1)
        lca = lca_depth_matrix(vt)
        for (i, j), d in lca.items():
            M[i, j] += d / depth_max
    if vtrees:
        M /= len(vtrees)
    np.fill_diagonal(M, 1.0)
    return M


def consensus_vtree(
    vtrees: Sequence[VtreeNode],
    n_features: int,
    fallback_seed: int = 0,
) -> VtreeNode:
    """
    Consensus structure from multiple source vtrees: co-grouping matrix →
    distance (1 − M) → average-linkage hierarchical clustering → vtree.

    Degrades gracefully: one source returns it unchanged; an empty list falls
    back to a random balanced vtree.

    NOTE (critical alignment dependency, dev.md §3.2): consensus is only
    meaningful when the feature dimensions of the source domains are aligned
    (shared projection fϕ or an explicit alignment step beforehand).
    """
    if len(vtrees) == 0:
        return random_balanced_vtree(list(range(n_features)), seed=fallback_seed)
    if len(vtrees) == 1:
        return vtrees[0]

    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    M = cogroup_matrix(vtrees, n_features)
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    tree = to_tree(Z)

    def _convert(node) -> VtreeNode:
        if node.is_leaf():
            return VtreeLeaf(int(node.id))
        return VtreeInternal(_convert(node.left), _convert(node.right))

    return _convert(tree)


def single_task_vtree(pc: "DensityPC") -> VtreeNode:
    """Structure learned on one task (negative control): the PC's own vtree."""
    return extract_vtree(pc.root)


# ─── Learned (Chow-Liu / HCLT-style) vtree ──────────────────────────────────

def mutual_information_matrix(X, n_bins: int = 8, kind: str = "auto") -> np.ndarray:
    """
    Pairwise mutual information estimate between features of X (B×d).

      kind="continuous": Gaussian-copula estimate via Spearman rank
        correlation, MI = −½ log(1 − ρ²) — robust and free of binning.
      kind="discrete":   histogram MI on integer-valued data.
      kind="auto":       discrete if X is integer-valued with few levels.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]

    if kind == "auto":
        is_int = np.allclose(X, np.round(X))
        few_levels = all(len(np.unique(X[:, j])) <= n_bins * 4 for j in range(d))
        kind = "discrete" if (is_int and few_levels) else "continuous"

    M = np.zeros((d, d))
    if kind == "continuous":
        from scipy.stats import rankdata

        R = np.apply_along_axis(rankdata, 0, X)
        rho = np.corrcoef(R, rowvar=False)
        rho = np.clip(np.nan_to_num(rho), -0.999, 0.999)
        M = -0.5 * np.log(1.0 - rho ** 2)
    else:
        for i in range(d):
            for j in range(i + 1, d):
                joint, _, _ = np.histogram2d(X[:, i], X[:, j], bins=n_bins)
                p = joint / max(joint.sum(), 1.0)
                pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
                mask = p > 0
                M[i, j] = M[j, i] = float(
                    (p[mask] * np.log(p[mask] / (pi @ pj)[mask])).sum()
                )
    np.fill_diagonal(M, 0.0)
    return M


def chow_liu_vtree(
    X,
    n_bins: int = 8,
    kind: str = "auto",
    min_balance: float = 1.0 / 3.0,
) -> VtreeNode:
    """
    Learned vtree (HCLT-style; Chow & Liu 1968, Liu & Van den Broeck 2021):

      1. Estimate the pairwise mutual-information matrix from data.
      2. Build the Chow-Liu tree = maximum spanning tree under MI.
      3. Recursively cut the WEAKEST edge among sufficiently balanced cuts
         (both sides ≥ min_balance of the variables when possible), so
         strongly dependent features stay together deep in the vtree.

    Compared to a random vtree, the resulting structured-decomposable PC
    spends its mixture capacity on the dependencies that exist in the data —
    expressiveness rises while every circuit property is preserved (the vtree
    is just better; nothing about the construction changes).
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    d = X.shape[1]
    if d == 1:
        return VtreeLeaf(0)

    from scipy.sparse.csgraph import minimum_spanning_tree

    M = mutual_information_matrix(X, n_bins=n_bins, kind=kind)
    # max spanning tree = min spanning tree on negated weights (the small
    # epsilon keeps the complete graph connected when MI is exactly 0)
    mst = minimum_spanning_tree(-(M + 1e-9)).tocoo()
    edges = [(int(i), int(j), float(M[i, j])) for i, j in zip(mst.row, mst.col)]
    adj: Dict[int, Dict[int, float]] = {i: {} for i in range(d)}
    for i, j, w in edges:
        adj[i][j] = w
        adj[j][i] = w

    def component(start: int, banned: Tuple[int, int], allowed: FrozenSet[int]) -> FrozenSet[int]:
        """Vertices reachable from start within `allowed`, not crossing `banned`."""
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in allowed and v not in seen and (u, v) != banned and (v, u) != banned:
                    seen.add(v)
                    stack.append(v)
        return frozenset(seen)

    def split(nodes: FrozenSet[int]) -> VtreeNode:
        if len(nodes) == 1:
            return VtreeLeaf(next(iter(nodes)))
        local_edges = [(i, j, w) for i, j, w in edges if i in nodes and j in nodes]
        candidates = []
        for i, j, w in local_edges:
            left = component(i, (i, j), nodes)
            right = nodes - left
            candidates.append((min(len(left), len(right)), -w, left, right))
        floor_size = max(1, int(len(nodes) * min_balance))
        balanced = [c for c in candidates if c[0] >= floor_size]
        if balanced:
            # weakest dependency among balanced cuts (-w max → w min)
            _, _, left, right = max(balanced, key=lambda c: c[1])
        else:
            # no balanced cut exists: take the most balanced, then weakest
            _, _, left, right = max(candidates, key=lambda c: (c[0], c[1]))
        return VtreeInternal(split(left), split(right))

    return split(frozenset(range(d)))


# ─── Curvature-guided vtrees ─────────────────────────────────────────────────
#
# A product node asserts independence across its scope cut, and the modeling
# error of that assertion is the mutual information crossing the cut.  The
# weakest-MST-edge rule above is a rank-1 surrogate for "find the cut with the
# least aggregate dependence": it sees one pairwise weight.  Ollivier-Ricci
# curvature scores an edge by the optimal-transport cost between its
# endpoints' whole neighborhoods, so negatively curved edges are
# neighborhood-aware bottlenecks (Topping et al., ICLR 2022: negative
# curvature ⇔ information bottleneck; Ni et al., Sci Rep 2019: Ricci flow
# community detection).  Cutting there places product nodes where the
# cross-scope dependence is smallest.
#
# Any of these learners returns a valid vtree, so all four circuit properties
# hold regardless of the method — only structure QUALITY is at stake.
# The honest adversary for the curvature methods is recursive spectral
# normalized cut of the MI matrix (spectral_vtree below): if spectral ≈ ORC
# on held-out NLL/AUROC, the geometry is decoration.

def sparsify_mi_graph(M: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """
    Symmetric k-NN sparsification of the dense MI similarity matrix, unioned
    with the maximum spanning tree so the graph stays connected.

    ORC degenerates on dense graphs (every neighborhood overlaps everything,
    curvatures compress toward a constant), so sparsification is required —
    and k is an explicit, ablatable hyperparameter, not a hidden choice.
    Default k = max(3, ⌈log₂ d⌉).
    """
    d = M.shape[0]
    if k is None:
        k = max(3, int(np.ceil(np.log2(max(d, 2)))))
    k = min(k, d - 1)
    A = np.zeros_like(M)
    order = np.argsort(-M, axis=1)
    for i in range(d):
        kept = 0
        for j in order[i]:
            if j == i:
                continue
            A[i, j] = A[j, i] = max(M[i, j], 1e-9)
            kept += 1
            if kept >= k:
                break
    from scipy.sparse.csgraph import minimum_spanning_tree

    mst = minimum_spanning_tree(-(M + 1e-9)).tocoo()
    for i, j in zip(mst.row, mst.col):
        A[i, j] = A[j, i] = max(M[i, j], 1e-9)
    return A


def _graph_edges(A: np.ndarray) -> List[Tuple[int, int]]:
    ii, jj = np.nonzero(np.triu(A, 1))
    return list(zip(ii.tolist(), jj.tolist()))


def _w1_exact(mu: np.ndarray, nu: np.ndarray, C: np.ndarray) -> float:
    """Exact Wasserstein-1 between two small discrete measures (per-edge
    supports after sparsification are tiny, so the LP is trivial)."""
    from scipy.optimize import linprog

    n, m = len(mu), len(nu)
    A_eq = np.zeros((n + m - 1, n * m))
    for i in range(n):
        A_eq[i, i * m:(i + 1) * m] = 1.0          # row marginals = mu
    for j in range(m - 1):                         # col marginals = nu
        A_eq[n + j, j::m] = 1.0                    # (last is redundant)
    b_eq = np.concatenate([mu, nu[:-1]])
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                  method="highs")
    return float(res.fun)


def ollivier_ricci_curvature(
    A: np.ndarray, alpha: float = 0.5
) -> Dict[Tuple[int, int], float]:
    """
    Ollivier-Ricci curvature κ(x,y) = 1 − W₁(m_x, m_y)/d(x,y) for every edge
    of a sparsified similarity graph A (weights = dependence strength).

    Conventions: edge LENGTH = 1/weight (strong dependence = close), ground
    distance d = shortest path on those lengths, and m_x is the α-lazy random
    walk measure (mass α at x, the rest over neighbors ∝ edge weight).
    κ < 0 marks bridges between communities; κ > 0 marks intra-community
    edges.  W₁ is solved exactly per edge.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    d = A.shape[0]
    L = np.where(A > 0, 1.0 / np.maximum(A, 1e-12), 0.0)
    D = shortest_path(csr_matrix(L), directed=False)

    def measure(x: int) -> Tuple[np.ndarray, np.ndarray]:
        nbrs = np.nonzero(A[x])[0]
        w = A[x, nbrs] / A[x, nbrs].sum()
        return (np.concatenate([[x], nbrs]),
                np.concatenate([[alpha], (1.0 - alpha) * w]))

    kappa: Dict[Tuple[int, int], float] = {}
    for i, j in _graph_edges(A):
        si, mi = measure(i)
        sj, mj = measure(j)
        C = D[np.ix_(si, sj)]
        kappa[(i, j)] = 1.0 - _w1_exact(mi, mj, C) / D[i, j]
    return kappa


def forman_curvature(A: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Weighted Forman-Ricci curvature (Sreejith et al. 2016, unit node weights)
    on the sparsified similarity graph, with edge weights = lengths 1/A:

        F(e) = 2 − √ℓ_e · ( Σ_{e'∼x, e'≠e} 1/√ℓ_{e'}
                          + Σ_{e'∼y, e'≠e} 1/√ℓ_{e'} )

    Closed form, no optimal transport — the cheap O(E·deg) proxy for high
    dimensions.  Same sign semantics as ORC: bridges are strongly negative.
    """
    L = np.where(A > 0, 1.0 / np.maximum(A, 1e-12), 0.0)
    F: Dict[Tuple[int, int], float] = {}
    for i, j in _graph_edges(A):
        le = L[i, j]
        s = 0.0
        for u, v in ((i, j), (j, i)):
            nbrs = np.nonzero(A[u])[0]
            s += sum(1.0 / np.sqrt(L[u, n]) for n in nbrs if n != v)
        F[(i, j)] = 2.0 - np.sqrt(le) * s
    return F


_CURVATURES = {"ollivier": ollivier_ricci_curvature, "forman": forman_curvature}


def curvature_sign_stability(
    X,
    curvature: str = "ollivier",
    n_boot: int = 20,
    k: Optional[int] = None,
    seed: int = 0,
    n_bins: int = 8,
    kind: str = "auto",
) -> Dict[Tuple[int, int], float]:
    """
    MI estimates are noisy at small n, so curvature signs may be too.  Fix
    the sparsified topology from the full sample, bootstrap-resample the
    rows, re-estimate MI weights on the SAME edge set, recompute curvature,
    and report per edge the fraction of resamples whose curvature sign agrees
    with the full-sample sign.  Edges with stability well below 1 should not
    be trusted as cut sites.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    curv = _CURVATURES[curvature]
    M = mutual_information_matrix(X, n_bins=n_bins, kind=kind)
    A = sparsify_mi_graph(M, k=k)
    ref = curv(A)

    rng = np.random.default_rng(seed)
    agree = {e: 0 for e in ref}
    mask = A > 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        Mb = mutual_information_matrix(X[idx], n_bins=n_bins, kind=kind)
        Ab = np.where(mask, np.maximum(Mb, 1e-9), 0.0)
        kb = curv(Ab)
        for e in ref:
            agree[e] += int(np.sign(kb[e]) == np.sign(ref[e]))
    return {e: agree[e] / n_boot for e in ref}


def _components(adj: Dict[int, set], nodes: List[int]) -> List[frozenset]:
    seen: set = set()
    comps = []
    for s in nodes:
        if s in seen:
            continue
        comp, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in comp:
                    comp.add(v)
                    stack.append(v)
        seen |= comp
        comps.append(frozenset(comp))
    return comps


def _two_sides(comps: List[frozenset]) -> Tuple[frozenset, frozenset]:
    """Greedy size-balanced grouping of ≥2 components into two sides."""
    left, right = set(), set()
    for c in sorted(comps, key=len, reverse=True):
        (left if len(left) <= len(right) else right).update(c)
    return frozenset(left), frozenset(right)


def curvature_vtree(
    X,
    curvature: str = "ollivier",
    k: Optional[int] = None,
    min_balance: float = 1.0 / 3.0,
    flow_iters: int = 0,
    n_bins: int = 8,
    kind: str = "auto",
) -> VtreeNode:
    """
    Curvature-guided learned vtree: recursively bipartition the variable set
    by deleting the most negatively curved edges of the sparsified MI graph
    until it disconnects — product nodes are introduced at the dependency
    bottlenecks, where the cross-scope dependence (the product node's exact
    modeling error) is smallest.

    curvature: "ollivier" (exact W₁ per edge) or "forman" (closed form, for
        high d).
    flow_iters > 0: Ricci-flow reweighting (Ni et al. 2019) before ranking —
        each iteration sets w ← w·(1+κ̂) with κ̂ normalized to [−1, 1], so
        bottleneck edges weaken and community edges strengthen.
    min_balance: same balance floor as chow_liu_vtree; among splits found
        while deleting edges, the first meeting the floor wins, else the most
        balanced one seen.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    d = X.shape[1]
    if d == 1:
        return VtreeLeaf(0)
    curv = _CURVATURES[curvature]
    M = mutual_information_matrix(X, n_bins=n_bins, kind=kind)

    def split(nodes: FrozenSet[int]) -> VtreeNode:
        if len(nodes) == 1:
            return VtreeLeaf(next(iter(nodes)))
        if len(nodes) == 2:
            a, b = sorted(nodes)
            return VtreeInternal(VtreeLeaf(a), VtreeLeaf(b))
        idx = sorted(nodes)
        A = sparsify_mi_graph(M[np.ix_(idx, idx)], k=k)
        for _ in range(flow_iters):
            kap = curv(A)
            scale = max(abs(v) for v in kap.values()) or 1.0
            for (i, j), kv in kap.items():
                A[i, j] = A[j, i] = max(A[i, j] * (1.0 + kv / scale), 1e-9)
        kap = curv(A)

        # delete edges from most negative curvature up, tracking the best
        # (most balanced) split; accept the first one meeting the floor
        adj = {i: set(np.nonzero(A[i])[0].tolist()) for i in range(len(idx))}
        order = sorted(kap, key=lambda e: (kap[e], e))
        floor = max(1, int(len(nodes) * min_balance))
        best: Optional[Tuple[int, frozenset, frozenset]] = None
        for i, j in order:
            adj[i].discard(j)
            adj[j].discard(i)
            comps = _components(adj, list(range(len(idx))))
            if len(comps) < 2:
                continue
            left, right = _two_sides(comps)
            size = min(len(left), len(right))
            if best is None or size > best[0]:
                best = (size, left, right)
            if size >= floor:
                break
        assert best is not None  # the deletion loop must disconnect the graph
        _, left, right = best
        to_global = lambda s: frozenset(idx[i] for i in s)
        return VtreeInternal(split(to_global(left)), split(to_global(right)))

    return split(frozenset(range(d)))


def spectral_vtree(
    X,
    min_balance: float = 1.0 / 3.0,
    n_bins: int = 8,
    kind: str = "auto",
) -> VtreeNode:
    """
    Recursive two-way NORMALIZED CUT (Shi & Malik 2000) of the dense MI
    similarity matrix — the honest adversary for the curvature methods, and
    the baseline any reviewer will ask for: vanilla spectral hierarchical
    partitioning with no transport geometry.  Fiedler vector of the
    normalized Laplacian, sweep cut minimizing Ncut among splits meeting the
    balance floor.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    d = X.shape[1]
    if d == 1:
        return VtreeLeaf(0)
    M = mutual_information_matrix(X, n_bins=n_bins, kind=kind)

    def split(nodes: FrozenSet[int]) -> VtreeNode:
        if len(nodes) == 1:
            return VtreeLeaf(next(iter(nodes)))
        if len(nodes) == 2:
            a, b = sorted(nodes)
            return VtreeInternal(VtreeLeaf(a), VtreeLeaf(b))
        idx = sorted(nodes)
        n = len(idx)
        W = M[np.ix_(idx, idx)] + 1e-9
        np.fill_diagonal(W, 0.0)
        deg = W.sum(1)
        dinv = 1.0 / np.sqrt(deg)
        lsym = np.eye(n) - dinv[:, None] * W * dinv[None, :]
        _, vecs = np.linalg.eigh(lsym)
        fiedler = dinv * vecs[:, 1]
        order = np.argsort(fiedler)

        floor = max(1, int(n * min_balance))
        best = None
        for t in range(1, n):
            S, Sc = order[:t], order[t:]
            cut = W[np.ix_(S, Sc)].sum()
            ncut = cut / deg[S].sum() + cut / deg[Sc].sum()
            # min Ncut among floor-meeting splits; if none meets the floor,
            # the most balanced split (then min Ncut) wins
            meets = min(t, n - t) >= floor
            key = (1, 0, -ncut) if meets else (0, min(t, n - t), -ncut)
            if best is None or key > best[0]:
                best = (key, S, Sc)
        _, S, Sc = best
        to_global = lambda s: frozenset(idx[i] for i in s)
        return VtreeInternal(split(to_global(S)), split(to_global(Sc)))

    return split(frozenset(range(d)))


# ─── Temporal (time × channel) vtrees ────────────────────────────────────────
#
# A windowed multivariate series flattened row-major, index = t·C + c, has a
# structure the learners above have to rediscover but which physics hands us
# for free: adjacent timesteps are dependent, and sensors group by
# sub-component.  These constructors encode that prior directly, and they are
# the honest hand-built adversary for the learned vtrees — if a dyadic time
# split matches chow_liu / ORC on held-out NLL, structure LEARNING buys nothing
# here even though structure itself does.

def window_index(t: int, c: int, n_channels: int) -> int:
    """Flat feature index of channel c at step t (row-major window layout)."""
    return t * n_channels + c


def _balanced_over(indices: Sequence[int]) -> VtreeNode:
    """Balanced vtree over a list of feature indices, order preserved."""
    idx = list(indices)
    if len(idx) == 1:
        return VtreeLeaf(idx[0])
    mid = len(idx) // 2
    return VtreeInternal(_balanced_over(idx[:mid]), _balanced_over(idx[mid:]))


def time_channel_vtree(
    n_steps: int,
    n_channels: int,
    mode: str = "time",
    channel_groups: Optional[Sequence[Sequence[int]]] = None,
) -> VtreeNode:
    """
    Vtree over a flattened (n_steps × n_channels) window.

      mode="time":    split the TIME axis dyadically; at a single timestep,
                      a balanced vtree over channels.  Encodes temporal
                      locality — neighbouring timesteps stay together deepest.
      mode="channel": split the CHANNEL axis first; at a single channel, a
                      balanced vtree over its time series.  Encodes per-sensor
                      locality — each channel's trajectory is one sub-circuit.
      channel_groups: with mode="channel", an explicit sensor grouping (e.g.
                      the fan / LPC / HPC / LPT / HPT sub-components of a
                      turbofan).  Groups are split before individual channels,
                      so physically coupled sensors share a sub-circuit.

    Any of these is a valid vtree, so all four circuit properties hold for all
    of them; only structure QUALITY differs.
    """
    if mode == "time":
        def build(steps: Sequence[int]) -> VtreeNode:
            if len(steps) == 1:
                t = steps[0]
                return _balanced_over([window_index(t, c, n_channels)
                                       for c in range(n_channels)])
            mid = len(steps) // 2
            return VtreeInternal(build(steps[:mid]), build(steps[mid:]))
        return build(list(range(n_steps)))

    if mode == "channel":
        def channel_subtree(c: int) -> VtreeNode:
            return _balanced_over([window_index(t, c, n_channels)
                                   for t in range(n_steps)])

        def build(chans: Sequence[int]) -> VtreeNode:
            if len(chans) == 1:
                return channel_subtree(chans[0])
            mid = len(chans) // 2
            return VtreeInternal(build(chans[:mid]), build(chans[mid:]))

        if channel_groups is None:
            return build(list(range(n_channels)))
        groups = [list(g) for g in channel_groups if len(g) > 0]
        covered = sorted(c for g in groups for c in g)
        if covered != list(range(n_channels)):
            raise ValueError(
                f"channel_groups must partition 0..{n_channels - 1}, got {covered}")

        def build_groups(gs: Sequence[Sequence[int]]) -> VtreeNode:
            if len(gs) == 1:
                return build(gs[0])
            mid = len(gs) // 2
            return VtreeInternal(build_groups(gs[:mid]), build_groups(gs[mid:]))
        return build_groups(groups)

    raise KeyError(f"Unknown temporal vtree mode {mode!r} (use 'time' or 'channel')")


VTREE_METHODS = ("chow_liu", "spectral", "orc", "forman", "random")
TEMPORAL_VTREE_METHODS = ("time", "channel")


# ═══════════════════════════════════════════════════════════════════════════
# 2c. Region graphs — the structure a vtree is a special case of
# ═══════════════════════════════════════════════════════════════════════════
#
# A vtree forces two things that are assumptions, not requirements:
#   BINARY   every scope splits in exactly two,
#   SINGLE   every scope splits exactly one way.
#
# A region graph drops both.  A REGION is a scope; a PARTITION of that region
# is a tuple of child regions with disjoint scopes covering it.  A region may
# carry several alternative partitions.  Vtree == region graph in which every
# region is binary and has exactly one partition.
#
# Why this matters for curvature specifically: deleting the most negatively
# curved edges of the dependency graph naturally disconnects it into k ≥ 2
# components.  `curvature_vtree` then throws that away — `_two_sides` greedily
# glues the components back into two sides so the result fits a binary vtree.
# The region graph keeps the k-way split the geometry actually found, and can
# keep several cut depths as ALTERNATIVE partitions of the same region.
#
# PROPERTY BOOKKEEPING (this is the whole trade, state it plainly):
#   smoothness       always holds  (units of a region share its scope)
#   decomposability  always holds  (a partition has disjoint child scopes)
#   structured dec.  holds IFF every region has exactly ONE partition
#   determinism      unchanged (absent for full-support mixtures)
# So exact density, exact marginals and exact box/survival queries survive any
# region graph — those need only smoothness + decomposability.  Squaring (SOS)
# and circuit multiplication need structured decomposability, hence a
# single-partition region graph; `SquaredPC` enforces that and says so.

@dataclass
class RegionNode:
    """
    One region of a region graph: a scope plus its alternative partitions.
    `partitions` empty means a leaf region (its scope must be a single
    variable).  Regions are shared by identity, so the graph is a DAG.
    """
    scope: FrozenSet[int]
    partitions: List[Tuple["RegionNode", ...]] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.partitions) == 0

    @property
    def feature_idx(self) -> int:
        if not self.is_leaf:
            raise ValueError("feature_idx is only defined for leaf regions")
        return next(iter(self.scope))

    def __repr__(self) -> str:
        return (f"RegionNode(|scope|={len(self.scope)}, "
                f"partitions={[tuple(len(c.scope) for c in p) for p in self.partitions]})")


def region_nodes(root: RegionNode) -> List[RegionNode]:
    """Every distinct region reachable from root (by identity)."""
    seen, out, stack = set(), [], [root]
    while stack:
        r = stack.pop()
        if id(r) in seen:
            continue
        seen.add(id(r))
        out.append(r)
        for part in r.partitions:
            stack.extend(part)
    return out


def is_structured_decomposable_rg(root: RegionNode) -> bool:
    """A region graph induces structured decomposability iff it is a tree of
    partitions — one partition per region."""
    return all(len(r.partitions) <= 1 for r in region_nodes(root))


def region_graph_arity(root: RegionNode) -> int:
    return max((len(p) for r in region_nodes(root) for p in r.partitions), default=0)


def region_graph_from_vtree(vtree: VtreeNode) -> RegionNode:
    """Lift a vtree into the equivalent binary single-partition region graph."""
    cache: Dict[FrozenSet[int], RegionNode] = {}

    def build(node: VtreeNode) -> RegionNode:
        scope = node.scope
        hit = cache.get(scope)
        if hit is not None:
            return hit
        if isinstance(node, VtreeLeaf):
            r = RegionNode(scope)
        else:
            r = RegionNode(scope, [(build(node.left), build(node.right))])
        cache[scope] = r
        return r

    return build(vtree)


def _rg_leaf(scope: FrozenSet[int], cache: Dict[FrozenSet[int], RegionNode]) -> RegionNode:
    hit = cache.get(scope)
    if hit is None:
        hit = cache[scope] = RegionNode(scope)
    return hit


def curvature_region_graph(
    X,
    curvature: str = "ollivier",
    k: Optional[int] = None,
    n_partitions: int = 1,
    max_arity: int = 4,
    min_balance: float = 1.0 / 3.0,
    flow_iters: int = 0,
    n_bins: int = 8,
    kind: str = "auto",
) -> RegionNode:
    """
    Build a region graph DIRECTLY from Ollivier-Ricci / Forman curvature, with
    no binary vtree in between.

    At each region: sparsify the MI graph, rank edges by curvature, and delete
    from the most negative upward.  The moment the graph disconnects, the
    resulting components ARE the partition — k-way, exactly as the geometry
    found it, capped at `max_arity` (excess components are merged into the
    smallest sides to keep the product arity bounded).

    n_partitions > 1 keeps deleting and records each successive, finer
    disconnection as an ALTERNATIVE partition of the same region, so a sum node
    mixes several different decompositions of its scope.  That is strictly more
    expressive than any vtree — and it gives up structured decomposability, so
    SOS and circuit multiplication are no longer available (exact density,
    marginals and box queries are unaffected).  n_partitions=1 keeps the graph
    structured-decomposable.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    d = X.shape[1]
    curv = _CURVATURES[curvature]
    M = mutual_information_matrix(X, n_bins=n_bins, kind=kind)
    cache: Dict[FrozenSet[int], RegionNode] = {}

    def cut_sequence(idx: List[int]) -> List[List[FrozenSet[int]]]:
        """
        Successive component sets as curvature-ranked edges are deleted,
        ordered from coarsest (2-way) to finest.

        The stopping rule is the geometry's, not a hyperparameter: keep
        deleting only while the next edge still has NEGATIVE curvature, i.e.
        while it is still a bottleneck between communities.  Once curvature
        turns positive we would be cutting inside a community, which is exactly
        the dependence a product node must not break.  So the ARITY of the
        partition is chosen by how many genuine bottlenecks the region has,
        capped at `max_arity` — a binary vtree is the special case where that
        count happens to be one.
        """
        A = sparsify_mi_graph(M[np.ix_(idx, idx)], k=k)
        for _ in range(flow_iters):
            kap = curv(A)
            scale = max(abs(v) for v in kap.values()) or 1.0
            for (i, j), kv in kap.items():
                A[i, j] = A[j, i] = max(A[i, j] * (1.0 + kv / scale), 1e-9)
        kap = curv(A)
        adj = {i: set(np.nonzero(A[i])[0].tolist()) for i in range(len(idx))}
        order = sorted(kap, key=lambda e: (kap[e], e))
        found: List[List[FrozenSet[int]]] = []
        seen_sigs = set()
        for i, j in order:
            negative = kap[(i, j)] < 0
            adj[i].discard(j)
            adj[j].discard(i)
            comps = _components(adj, list(range(len(idx))))
            if len(comps) < 2:
                continue
            glob = [frozenset(idx[v] for v in c) for c in comps]
            sig = frozenset(glob)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            found.append(glob)
            if n_partitions > 1 and len(found) >= n_partitions:
                break                      # collected enough alternatives
            if len(comps) >= max_arity:
                break
            # past the bottlenecks: stop once we have at least one valid split
            if not negative and found and n_partitions == 1:
                break
        return found

    def cap_arity(comps: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
        """Merge the smallest components until the arity fits `max_arity`."""
        comps = sorted(comps, key=len, reverse=True)
        while len(comps) > max_arity:
            a, b = comps.pop(), comps.pop()
            comps.append(a | b)
            comps.sort(key=len, reverse=True)
        return comps

    def build(scope: FrozenSet[int]) -> RegionNode:
        hit = cache.get(scope)
        if hit is not None:
            return hit
        if len(scope) == 1:
            return _rg_leaf(scope, cache)
        region = RegionNode(scope)
        cache[scope] = region                     # register before recursing
        idx = sorted(scope)
        if len(scope) == 2:
            region.partitions = [tuple(_rg_leaf(frozenset({v}), cache) for v in idx)]
            return region
        seqs = cut_sequence(idx) or [[frozenset({v}) for v in idx]]
        # single-partition: take the WIDEST split the bottlenecks supported.
        # multi-partition: keep the coarse-to-fine alternatives as a mixture
        # over decompositions of the same scope.
        seqs = list(reversed(seqs)) if n_partitions == 1 else seqs
        for comps in seqs[:max(n_partitions, 1)]:
            comps = cap_arity(comps)
            if len(comps) < 2:
                continue
            region.partitions.append(tuple(build(c) for c in comps))
        if not region.partitions:                 # degenerate: split off one var
            first = frozenset({idx[0]})
            region.partitions = [(build(first), build(scope - first))]
        return region

    return build(frozenset(range(d)))


def spectral_region_graph(
    X, n_partitions: int = 1, max_arity: int = 4, min_balance: float = 1.0 / 3.0,
    n_bins: int = 8, kind: str = "auto",
) -> RegionNode:
    """
    The honest adversary for `curvature_region_graph`, at matched arity:
    recursive normalized-cut region graph.  Multi-way splits come from applying
    the sweep cut repeatedly to the largest side, so any advantage of curvature
    is not merely "more than two children".
    """
    vt = spectral_vtree(X, min_balance=min_balance, n_bins=n_bins, kind=kind)
    rg = region_graph_from_vtree(vt)
    if max_arity <= 2:
        return rg
    return flatten_region_graph(rg, max_arity=max_arity)


def flatten_region_graph(root: RegionNode, max_arity: int = 4) -> RegionNode:
    """
    Collapse chains of binary partitions into wider ones, up to `max_arity`.
    Used to give a binary structure the same product arity as a curvature
    region graph so the two can be compared at matched shape.
    """
    cache: Dict[FrozenSet[int], RegionNode] = {}

    def children_of(r: RegionNode, budget: int) -> List[RegionNode]:
        if r.is_leaf or budget <= 1:
            return [r]
        out: List[RegionNode] = []
        for c in r.partitions[0]:
            if not c.is_leaf and len(out) + len(c.partitions[0]) <= budget:
                out.extend(children_of(c, budget - len(out)))
            else:
                out.append(c)
        return out

    def build(r: RegionNode) -> RegionNode:
        hit = cache.get(r.scope)
        if hit is not None:
            return hit
        if r.is_leaf:
            return _rg_leaf(r.scope, cache)
        new = RegionNode(r.scope)
        cache[r.scope] = new
        kids = children_of(r, max_arity)
        if len(kids) < 2:
            kids = list(r.partitions[0])
        new.partitions = [tuple(build(c) for c in kids)]
        return new

    return build(root)


# ─── Chain / HMM region graphs: the tractable answer to "like an LSTM" ───────
#
# A balanced vtree over a flattened window is ORDER-BLIND in a specific way: the
# circuit is a finite mixture of factorised distributions, so if the per-step
# marginals are similar (a stationary window), permuting the timesteps barely
# changes the density.  That is exactly the observed `decouple` failure — a
# time-permuted channel scores at chance.
#
# The fix is not more components, it is a different REGION GRAPH.  A right-
# linear ("caterpillar") chain over time,
#
#     R_t = {t, t+1, …, w}  partitioned as  ({t}, R_{t+1})
#
# turns each region's K units into the K states of a hidden Markov chain: unit i
# of R_t is p(x_{t:w} | s_t = i), the sum weights from R_t to R_{t+1} ARE the
# transition matrix, and evaluating the circuit IS the forward algorithm.  This
# is a proper HMM with Gaussian-mixture emissions — the classic tractable
# sequence model — expressed as a circuit, so it keeps smoothness,
# decomposability, structured decomposability and exact marginals/box queries.
#
# Relation to an LSTM: same recurrent shape, but the state is a DISCRETE latent
# with K values instead of a continuous vector, which is precisely the trade
# that buys exact inference.  Expressiveness is governed by K.  (Stacking
# independent chains — a factorial HMM — is NOT tractable in general, so that is
# not the way to scale it; increase K, or enrich the per-step emission.)

def chain_region_graph(
    n_steps: int,
    n_channels: int = 1,
    emission: str = "factorized",
    channel_groups: Optional[Sequence[Sequence[int]]] = None,
    reverse: bool = False,
) -> RegionNode:
    """
    Right-linear (HMM-shaped) region graph over a flattened (n_steps ×
    n_channels) window, feature index = t·C + c.

    emission: how the C channels of one timestep are decomposed.
      "factorized"  one product over all channels (independent given the state,
                    the standard HMM emission — dependence is carried by the
                    latent).
      "chain"       the channels are themselves chained, so within-step
                    cross-channel structure gets its own latent.
      "grouped"     one product over `channel_groups`, each group a sub-chain.

    Unlike a balanced vtree, EVERY timestep sits at a different depth of the
    same chain, so the density is not invariant to permuting the steps.
    """
    cache: Dict[FrozenSet[int], RegionNode] = {}
    idx = lambda t, c: t * n_channels + c

    def step_region(t: int) -> RegionNode:
        feats = [idx(t, c) for c in range(n_channels)]
        scope = frozenset(feats)
        hit = cache.get(scope)
        if hit is not None:
            return hit
        if len(feats) == 1:
            return _rg_leaf(scope, cache)
        r = RegionNode(scope)
        cache[scope] = r
        if emission == "factorized":
            r.partitions = [tuple(_rg_leaf(frozenset({f}), cache) for f in feats)]
        elif emission == "grouped":
            groups = channel_groups or [list(range(n_channels))]
            parts = []
            for g in groups:
                gs = frozenset(idx(t, c) for c in g)
                parts.append(_rg_leaf(gs, cache) if len(gs) == 1
                             else _sub_chain(sorted(gs), cache))
            r.partitions = [tuple(parts)]
        elif emission == "chain":
            r.partitions = [tuple(_sub_chain(feats, cache).partitions[0])]
        else:
            raise KeyError(f"unknown emission {emission!r}")
        return r

    def _sub_chain(feats: List[int], cch: Dict[FrozenSet[int], RegionNode]) -> RegionNode:
        scope = frozenset(feats)
        hit = cch.get(scope)
        if hit is not None:
            return hit
        if len(feats) == 1:
            return _rg_leaf(scope, cch)
        r = RegionNode(scope)
        cch[scope] = r
        head = _rg_leaf(frozenset({feats[0]}), cch)
        r.partitions = [(head, _sub_chain(feats[1:], cch))]
        return r

    steps = list(range(n_steps))
    if reverse:
        steps = steps[::-1]

    def suffix(i: int) -> RegionNode:
        scope = frozenset(idx(t, c) for t in steps[i:] for c in range(n_channels))
        hit = cache.get(scope)
        if hit is not None:
            return hit
        if i == len(steps) - 1:
            return step_region(steps[i])
        r = RegionNode(scope)
        cache[scope] = r
        r.partitions = [(step_region(steps[i]), suffix(i + 1))]
        return r

    return suffix(0)


def delta_window_transform(
    X, n_steps: int, n_channels: int, keep_first: bool = True
):
    """
    Replace each window with (x_1, Δx_2, …, Δx_w) per channel, where
    Δx_t = x_t − x_{t−1}.

    This is a UNIT-DETERMINANT linear map (unit lower-triangular per channel),
    so a density learned on the transformed window is a valid density on the
    original one with NO Jacobian correction — every exactness guarantee is
    preserved verbatim, and log p is comparable across the transform.

    Why it matters here: differences are what make a window order-sensitive.
    Permuting timesteps leaves the per-step marginals almost unchanged but
    scrambles the increments, so the `decouple` anomaly becomes visible to a
    model that would otherwise be nearly permutation-invariant.  It is the
    cheapest way to give a window circuit temporal structure, and it composes
    with (rather than replaces) the chain region graph.
    """
    is_torch = isinstance(X, torch.Tensor)
    A = X if is_torch else torch.as_tensor(np.asarray(X), dtype=torch.float32)
    W = A.reshape(-1, n_steps, n_channels).clone()
    out = W.clone()
    out[:, 1:, :] = W[:, 1:, :] - W[:, :-1, :]
    if not keep_first:
        out[:, 0, :] = 0.0
    out = out.reshape(len(A), -1)
    return out if is_torch else out.numpy()


REGION_GRAPH_METHODS = ("orc", "forman", "spectral", "vtree", "chain")


def learned_region_graph(
    X, method: str = "orc", n_partitions: int = 1, max_arity: int = 4,
    seed: int = 0, **kwargs
) -> RegionNode:
    """Dispatcher over region-graph learners (see REGION_GRAPH_METHODS)."""
    if method in ("orc", "ollivier"):
        return curvature_region_graph(X, curvature="ollivier",
                                      n_partitions=n_partitions,
                                      max_arity=max_arity, **kwargs)
    if method == "forman":
        return curvature_region_graph(X, curvature="forman",
                                      n_partitions=n_partitions,
                                      max_arity=max_arity, **kwargs)
    if method == "spectral":
        return spectral_region_graph(X, n_partitions=n_partitions,
                                     max_arity=max_arity, **kwargs)
    if method == "vtree":
        return region_graph_from_vtree(learned_vtree(X, method="chow_liu", seed=seed))
    raise KeyError(f"Unknown region-graph method {method!r} "
                   f"(use one of {REGION_GRAPH_METHODS})")


def learned_vtree(X, method: str = "chow_liu", seed: int = 0, **kwargs) -> VtreeNode:
    """
    Dispatcher over all vtree learners (every method yields a valid vtree, so
    the four circuit properties hold for all of them — matched-budget
    comparisons only ever differ in structure quality):

      "chow_liu"  MI max-spanning-tree + weakest-edge recursion (incumbent)
      "spectral"  recursive normalized cut of the MI matrix (the adversary)
      "orc"       Ollivier-Ricci curvature bottleneck cuts (exact W₁)
      "forman"    Forman-Ricci variant (closed form, for high d)
      "random"    balanced random vtree (control)
    """
    if method == "chow_liu":
        return chow_liu_vtree(X, **kwargs)
    if method == "spectral":
        return spectral_vtree(X, **kwargs)
    if method in ("orc", "ollivier"):
        return curvature_vtree(X, curvature="ollivier", **kwargs)
    if method == "forman":
        return curvature_vtree(X, curvature="forman", **kwargs)
    if method == "random":
        d = X.shape[1]
        return random_balanced_vtree(list(range(d)), seed=seed)
    raise KeyError(f"Unknown vtree method {method!r} (use one of {VTREE_METHODS})")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Leaf (input) units
# ═══════════════════════════════════════════════════════════════════════════

class LeafNode(nn.Module):
    """
    Base class for all input units.  A leaf is a normalized univariate
    distribution over one feature dimension.

    Subclass contract:
      log_density(v)        log f(v) for raw values v (any shape, elementwise)
      mode()                argmax_v f(v) as a float (exact for unimodal leaves)
      pair_log_integral(o)  log ∫ f(v) g(v) dv against another leaf of the
                            same type — needed only for the SOS mode.
    """

    def __init__(self, feature_idx: int):
        super().__init__()
        self.feature_idx = feature_idx

    def log_density(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log f(x[:, feature_idx]) — shape (B,)."""
        return self.log_density(x[:, self.feature_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)

    def log_integral(self) -> torch.Tensor:
        """log ∫ f(v) dv = 0 — every leaf is normalized by construction."""
        return torch.tensor(0.0)

    def mode(self) -> float:
        raise NotImplementedError

    # ── interval (box) queries ───────────────────────────────────────────
    #
    # log ∫_lo^hi f(v) dv.  Together with smoothness + decomposability this
    # is what makes an AXIS-ALIGNED BOX query exact on the whole circuit
    # (eval_log_marginal(..., boxes=...)): products factorise over disjoint
    # scopes and sums are mixtures, so a per-leaf interval mass propagates
    # exactly the same way a per-leaf density does.
    #
    # This is the query that turns a joint circuit over (features, τ) into a
    # survival model: S(t | x) = P(τ > t | x) is the box [t, ∞) on τ, and the
    # RIGHT-CENSORED log-likelihood of a unit last seen alive at time c is
    # exactly log P(τ > c | x) — no approximation, no separate hazard head.

    def log_cdf(self, v: torch.Tensor) -> torch.Tensor:
        """log F(v) = log ∫_{-∞}^{v} f."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement log_cdf")

    def log_interval(self, lo, hi) -> torch.Tensor:
        """
        log ∫_lo^hi f(v) dv.  ±inf endpoints are allowed; (-inf, +inf) must
        return log_integral() = 0.  For discrete leaves the interval is
        INCLUSIVE of both endpoints.  Endpoints may be python floats (result:
        scalar tensor) or per-sample tensors of shape (B,) (result: (B,)).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support interval/box queries; "
            "use GaussianLeaf, GaussianMixtureLeaf, or CategoricalLeaf for "
            "any feature you need to integrate over a sub-interval.")

    def pair_log_integral(self, other: "LeafNode") -> Tuple[torch.Tensor, torch.Tensor]:
        """(log |∫ f g|, sign) — required by the SOS mode only."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support the SOS mode; "
            "use GaussianLeaf, GaussianMixtureLeaf, or CategoricalLeaf."
        )


def _as_endpoint(v, ref: torch.Tensor) -> torch.Tensor:
    """Interval endpoint (python float or per-sample tensor) as a tensor."""
    if isinstance(v, torch.Tensor):
        return v.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(float(v), dtype=ref.dtype, device=ref.device)


def _gauss_log_interval(mu: torch.Tensor, sigma: torch.Tensor, lo, hi) -> torch.Tensor:
    """
    Elementwise log[ Φ((hi−μ)/σ) − Φ((lo−μ)/σ) ], numerically stable in both
    tails.  Endpoints may be python floats (including ±inf) or per-sample
    tensors, so a batch with a DIFFERENT censoring time per unit is one
    vectorised call rather than one circuit pass per distinct threshold.

    Two algebraically equal forms are computed,
        lower-anchored: log Φ(z_hi) + log1p(−exp(log Φ(z_lo) − log Φ(z_hi)))
        upper-anchored: log Φ(−z_lo) + log1p(−exp(log Φ(−z_hi) − log Φ(−z_lo)))
    and the one whose subtraction is better conditioned is selected.  A naive
    Φ(z_hi) − Φ(z_lo) cancels catastrophically (underflows to exactly 0) for
    intervals far out in a tail — precisely the survival queries we care about.
    The upper-anchored form also degrades correctly at ±inf, since
    log Φ(−inf) = −inf and log Φ(+inf) = 0.
    """
    lo_t, hi_t = _as_endpoint(lo, mu), _as_endpoint(hi, mu)
    z_lo, z_hi = (lo_t - mu) / sigma, (hi_t - mu) / sigma
    l_lo, l_hi = torch.special.log_ndtr(z_lo), torch.special.log_ndtr(z_hi)
    u_lo, u_hi = torch.special.log_ndtr(-z_lo), torch.special.log_ndtr(-z_hi)
    lower_anchored = l_hi + torch.log1p(-torch.exp((l_lo - l_hi).clamp(max=-1e-30)))
    upper_anchored = u_lo + torch.log1p(-torch.exp((u_hi - u_lo).clamp(max=-1e-30)))
    # anchor on whichever side holds less mass: that is the side where the two
    # log-CDFs are furthest apart and the log1p argument is furthest from 0
    out = torch.where(l_hi < u_lo, lower_anchored, upper_anchored)
    return torch.nan_to_num(out, nan=-1e30, neginf=-1e30)


class GaussianLeaf(LeafNode):
    """Single Gaussian input unit."""

    def __init__(self, feature_idx: int, mu_init: float = 0.0, sigma_init: float = 1.0):
        super().__init__(feature_idx)
        self.mu = nn.Parameter(torch.tensor(float(mu_init)))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.log_sigma) + 1e-5

    def fit(self, X) -> None:
        """Median / MAD initialisation from data (robust to outliers)."""
        vals = _column(X, self.feature_idx)
        mu = float(np.median(vals))
        sigma = float(np.median(np.abs(vals - mu)) + 1e-6) * 1.4826
        with torch.no_grad():
            self.mu.fill_(mu)
            self.log_sigma.fill_(_inv_softplus(sigma))

    def log_density(self, v: torch.Tensor) -> torch.Tensor:
        s = self.sigma
        return -0.5 * ((v - self.mu) / s) ** 2 - torch.log(s) - 0.5 * LOG_2PI

    def mode(self) -> float:
        return float(self.mu)

    def log_cdf(self, v: torch.Tensor) -> torch.Tensor:
        return torch.special.log_ndtr((v - self.mu) / self.sigma)

    def log_interval(self, lo: float, hi: float) -> torch.Tensor:
        return _gauss_log_interval(self.mu, self.sigma, lo, hi)

    def pair_log_integral(self, other: "GaussianLeaf") -> Tuple[torch.Tensor, torch.Tensor]:
        # ∫ N(v; μ1, σ1²) N(v; μ2, σ2²) dv = N(μ1 − μ2; 0, σ1² + σ2²)
        var = self.sigma ** 2 + other.sigma ** 2
        la = -0.5 * (self.mu - other.mu) ** 2 / var - 0.5 * torch.log(var) - 0.5 * LOG_2PI
        return la, torch.tensor(1.0)


class GaussianMixtureLeaf(LeafNode):
    """Mixture-of-Gaussians input unit (monotone, normalized)."""

    def __init__(self, feature_idx: int, n_components: int = 4):
        super().__init__(feature_idx)
        self.n_components = n_components
        self.mus = nn.Parameter(torch.linspace(-1.0, 1.0, n_components))
        self.log_sigmas = nn.Parameter(torch.zeros(n_components))
        self.logits = nn.Parameter(torch.zeros(n_components))

    @property
    def sigmas(self) -> torch.Tensor:
        return F.softplus(self.log_sigmas) + 1e-5

    def fit(self, X) -> None:
        """Quantile-spread initialisation of component means."""
        vals = _column(X, self.feature_idx)
        qs = np.quantile(vals, np.linspace(0.1, 0.9, self.n_components))
        spread = float(np.std(vals) + 1e-6) / max(self.n_components, 1)
        with torch.no_grad():
            self.mus.copy_(torch.tensor(qs, dtype=torch.float32))
            self.log_sigmas.fill_(_inv_softplus(max(spread, 1e-3)))
            self.logits.fill_(0.0)

    def log_density(self, v: torch.Tensor) -> torch.Tensor:
        s = self.sigmas
        z = (v.unsqueeze(-1) - self.mus) / s
        comp = -0.5 * z ** 2 - torch.log(s) - 0.5 * LOG_2PI
        return torch.logsumexp(F.log_softmax(self.logits, dim=0) + comp, dim=-1)

    def mode(self) -> float:
        # Best component mean (exact when components are well separated;
        # for K = 1 it is the exact mode).
        with torch.no_grad():
            dens = self.log_density(self.mus.detach())
        return float(self.mus[int(dens.argmax())])

    def log_cdf(self, v: torch.Tensor) -> torch.Tensor:
        z = (v.unsqueeze(-1) - self.mus) / self.sigmas
        return torch.logsumexp(
            F.log_softmax(self.logits, dim=0) + torch.special.log_ndtr(z), dim=-1)

    def log_interval(self, lo, hi) -> torch.Tensor:
        lo_t, hi_t = _as_endpoint(lo, self.mus), _as_endpoint(hi, self.mus)
        # endpoints get a trailing component axis so per-sample (B,) endpoints
        # broadcast against the (K,) component parameters -> (B, K)
        lo_b = lo_t.unsqueeze(-1) if lo_t.dim() > 0 else lo_t
        hi_b = hi_t.unsqueeze(-1) if hi_t.dim() > 0 else hi_t
        comp = _gauss_log_interval(self.mus, self.sigmas, lo_b, hi_b)
        return torch.logsumexp(F.log_softmax(self.logits, dim=0) + comp, dim=-1)

    def pair_log_integral(self, other: "GaussianMixtureLeaf") -> Tuple[torch.Tensor, torch.Tensor]:
        lw1 = F.log_softmax(self.logits, dim=0)
        lw2 = F.log_softmax(other.logits, dim=0)
        var = (self.sigmas ** 2).unsqueeze(-1) + (other.sigmas ** 2).unsqueeze(0)
        diff = self.mus.unsqueeze(-1) - other.mus.unsqueeze(0)
        cross = -0.5 * diff ** 2 / var - 0.5 * torch.log(var) - 0.5 * LOG_2PI
        la = torch.logsumexp(lw1.unsqueeze(-1) + lw2.unsqueeze(0) + cross, dim=(0, 1))
        return la, torch.tensor(1.0)


class CategoricalLeaf(LeafNode):
    """Categorical input unit over {0, …, n_categories − 1}."""

    def __init__(self, feature_idx: int, n_categories: int = 2):
        super().__init__(feature_idx)
        self.n_categories = n_categories
        self.logits = nn.Parameter(torch.zeros(n_categories))

    def fit(self, X) -> None:
        """Laplace-smoothed empirical frequencies."""
        vals = _column(X, self.feature_idx).astype(int)
        counts = np.bincount(vals, minlength=self.n_categories).astype(float) + 1.0
        with torch.no_grad():
            self.logits.copy_(torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32))

    def log_density(self, v: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(self.logits, dim=0)
        return lp[v.long()]

    def mode(self) -> float:
        return float(self.logits.argmax())

    def log_cdf(self, v: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(self.logits, dim=0)
        ks = torch.arange(self.n_categories, device=lp.device)
        mask = ks <= v.unsqueeze(-1)
        return torch.logsumexp(lp.masked_fill(~mask, -float("inf")), dim=-1)

    def log_interval(self, lo, hi) -> torch.Tensor:
        """
        log Σ_{k=⌈lo⌉}^{⌊hi⌋} p(k) — INCLUSIVE of both endpoints.  This is the
        ordinal-bin route to survival queries: with τ discretised into bins,
        P(τ > t) is the suffix sum log_interval(t + 1, inf), exact by
        construction and with no CDF approximation anywhere.

        Endpoints may be floats (±inf allowed) or per-sample tensors of shape
        (B,), in which case the result is (B,) — one call covers a batch whose
        units were each censored at a different time.
        """
        lp = F.log_softmax(self.logits, dim=0)
        ks = torch.arange(self.n_categories, device=lp.device, dtype=lp.dtype)
        lo_t, hi_t = _as_endpoint(lo, lp), _as_endpoint(hi, lp)
        mask = (ks >= torch.ceil(lo_t).unsqueeze(-1)) & (ks <= torch.floor(hi_t).unsqueeze(-1))
        out = torch.logsumexp(lp.expand_as(mask).masked_fill(~mask, -float("inf")), dim=-1)
        return torch.nan_to_num(out, nan=-1e30, neginf=-1e30)

    def pair_log_integral(self, other: "CategoricalLeaf") -> Tuple[torch.Tensor, torch.Tensor]:
        # Σ_k p(k) q(k)
        la = torch.logsumexp(
            F.log_softmax(self.logits, dim=0) + F.log_softmax(other.logits, dim=0), dim=0
        )
        return la, torch.tensor(1.0)


# ─── Heavy-tailed InputNode (original component, normalization restored) ────

@torch.jit.script
def log_gaussian_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.9189385


@torch.jit.script
def log_laplace_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    b = sigma * 0.70710678
    return -torch.abs(x - mu) / b - (torch.log(sigma) + 0.3465736)


@torch.jit.script
def log_student_jit(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    z = (x - mu) / (sigma + 1e-12)
    term1 = torch.lgamma((nu + 1) * 0.5)
    term2 = torch.lgamma(nu * 0.5)
    term3 = 0.5 * torch.log(nu * 3.14159265)
    term4 = torch.log(sigma + 1e-12)
    term5 = ((nu + 1) * 0.5) * torch.log1p((z ** 2) / nu)
    return term1 - term2 - term3 - term4 - term5


class InputNode(LeafNode):
    """
    Mixture of Gaussian / Laplace / Student-t over a single feature dimension.
    All three components share location mu, so the mixture mode is exactly mu.

    The temperature gate of the original implementation is removed: scaling
    log-densities breaks normalization and therefore exact marginals.
    """

    def __init__(self, feature_idx: int, mu_init: float = 0.0, sigma_init: float = 1.0):
        super().__init__(feature_idx)
        self.mu = nn.Parameter(torch.tensor(float(mu_init)))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))
        self.log_nu = nn.Parameter(torch.log(torch.tensor(5.0)))
        self.logits = nn.Parameter(torch.zeros(3))

    def fit(self, X) -> None:
        """Median / MAD initialisation from data."""
        vals = _column(X, self.feature_idx)
        mu = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu)) + 1e-6)
        sigma = mad * 1.4826
        with torch.no_grad():
            self.mu.fill_(mu)
            self.log_sigma.fill_(np.log(sigma))
            self.log_nu.fill_(np.log(5.0))
            self.logits.fill_(0.0)

    def log_density(self, v: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.log_sigma) + 1e-6
        nu = F.softplus(self.log_nu) + 1e-3
        stack = torch.stack(
            [
                log_gaussian_jit(v, self.mu, sigma),
                log_laplace_jit(v, self.mu, sigma),
                log_student_jit(v, self.mu, sigma, nu),
            ],
            dim=0,
        )
        w = torch.log_softmax(self.logits, dim=0).view(3, *([1] * v.dim()))
        return torch.logsumexp(w + stack, dim=0)

    def mode(self) -> float:
        return float(self.mu)

    def log_interval(self, lo: float, hi: float) -> torch.Tensor:
        """
        Not available: the Student-t component has no closed-form CDF in
        torch, and this project does not trade exactness for a quadrature
        approximation.  Only the BOXED features need this method, so the usual
        fix is a per-feature leaf_factory: heavy-tailed InputNode leaves on the
        sensor window, GaussianLeaf/CategoricalLeaf on the variable you
        integrate over (e.g. time-to-failure τ).
        """
        raise NotImplementedError(
            "InputNode has no closed-form CDF (Student-t component). Give the "
            "boxed feature a GaussianLeaf / GaussianMixtureLeaf / "
            "CategoricalLeaf via a per-feature leaf_factory.")


def _fit_leaves_with_jitter(model: nn.Module, X: torch.Tensor, jitter: float) -> None:
    """
    Shared fit_leaves implementation: closed-form leaf init plus a
    symmetry-breaking perturbation.

    The jitter is NOT cosmetic.  `fit` is a deterministic function of the data,
    so without it the K sibling leaves of every mixture start out identical;
    identical siblings receive identical gradients and stay identical forever,
    the mixture collapses to a product of marginals, and the circuit silently
    degenerates into an independence model.  For a joint circuit over
    (window, τ) that failure is invisible in the training loss but fatal at
    query time: p(τ | x) comes out independent of x — a flat survival curve
    for every unit.

    Every parameterisation therefore needs its own break, not just scalar-μ
    leaves:
      mu / log_sigma  (GaussianLeaf, InputNode)  -> perturb the location
      mus / log_sigmas (GaussianMixtureLeaf)     -> perturb component locations
      logits          (CategoricalLeaf)          -> perturb the log-probs
    """
    for m in model.modules():
        if not isinstance(m, LeafNode):
            continue
        m.fit(X)
        if jitter <= 0:
            continue
        with torch.no_grad():
            if hasattr(m, "mu") and hasattr(m, "log_sigma"):
                sigma = F.softplus(m.log_sigma) + 1e-6
                m.mu.add_(torch.randn(()) * jitter * sigma)
            elif hasattr(m, "mus") and hasattr(m, "log_sigmas"):
                sigmas = F.softplus(m.log_sigmas) + 1e-6
                m.mus.add_(torch.randn_like(m.mus) * jitter * sigmas)
            elif hasattr(m, "logits"):
                m.logits.add_(torch.randn_like(m.logits) * jitter)


def _column(X, idx: int) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return np.asarray(X)[:, idx].reshape(-1)


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) = y."""
    return float(np.log(np.expm1(max(y, 1e-8))))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Inner nodes
# ═══════════════════════════════════════════════════════════════════════════

class AbstractNode(nn.Module):
    def __init__(self, child_nodes=None):
        super().__init__()
        self.child_nodes = nn.ModuleList(child_nodes) if child_nodes else nn.ModuleList()

    def forward(self, x):
        raise NotImplementedError


class SumNode(AbstractNode):
    """
    Monotone weighted mixture over children sharing the same scope.
    Weights are softmax-normalized, so normalized children imply a normalized
    mixture (partition function 1).
    """

    def __init__(self, child_nodes, weights=None):
        super().__init__(child_nodes)
        n = len(child_nodes)
        if weights is None:
            self.weights = nn.Parameter(torch.full((n,), -math.log(n)))
        else:
            w_t = torch.tensor(weights, dtype=torch.float32)
            self.weights = nn.Parameter(torch.log(w_t / w_t.sum()))

    def log_weights(self) -> torch.Tensor:
        return torch.log_softmax(self.weights, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stack = torch.stack([c(x) for c in self.child_nodes], dim=0)
        w_shape = [-1] + [1] * (stack.dim() - 1)
        return torch.logsumexp(stack + self.log_weights().view(*w_shape), dim=0)


class ProductNode(AbstractNode):
    """Product over children with disjoint scopes (sums log-probs)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([c(x) for c in self.child_nodes], dim=0).sum(dim=0)


class ClassifierNode(AbstractNode):
    """Root node for the legacy classifier variant; not used in DensityPC."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([c(x) for c in self.child_nodes], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. DensityPC — vtree-respecting density estimator
# ═══════════════════════════════════════════════════════════════════════════

LeafFactory = Callable[[int], LeafNode]


class DensityPC(nn.Module):
    """
    A structured-decomposable PC whose scope-partition hierarchy is defined by
    a vtree.  Smoothness and decomposability hold by construction; structured
    decomposability holds with respect to `vtree_root`.

    Args:
        vtree_root:        Root of the vtree over feature indices 0..d-1.
        n_sum_components:  Number of product components under each SumNode.
                           K=1 gives a deterministic tree PC (no mixtures).
        leaf_factory:      feature_idx -> LeafNode.  Defaults to InputNode
                           (heavy-tailed mixture).  Use GaussianLeaf,
                           GaussianMixtureLeaf, or CategoricalLeaf as needed.
    """

    def __init__(
        self,
        vtree_root: VtreeNode,
        n_sum_components: int = 3,
        leaf_factory: Optional[LeafFactory] = None,
    ):
        super().__init__()
        self.vtree = vtree_root
        self.n_sum_components = n_sum_components
        self.leaf_factory = leaf_factory or InputNode
        self.root = self._build(vtree_root)

    def _build(self, node: VtreeNode) -> nn.Module:
        if isinstance(node, VtreeLeaf):
            return self.leaf_factory(node.feature_idx)
        prod_nodes = [
            ProductNode([self._build(node.left), self._build(node.right)])
            for _ in range(self.n_sum_components)
        ]
        if len(prod_nodes) == 1:
            return prod_nodes[0]
        return SumNode(prod_nodes)

    # ── fitting ──────────────────────────────────────────────────────────

    def fit_leaves(self, X: torch.Tensor, jitter: float = 0.05) -> None:
        """
        Data-driven initialisation of all leaves (shape B×d).  `jitter`
        perturbs scalar-location leaves by jitter·σ·ε: without it, the K
        sibling leaves of every mixture start identical and gradient symmetry
        keeps them identical forever — the mixture silently degenerates to a
        product of marginals and the vtree structure cannot matter.
        """
        _fit_leaves_with_jitter(self, X, jitter)

    # ── exact inference ──────────────────────────────────────────────────

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """log p(z), shape (B,)."""
        return self.root(z)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)

    def log_marginal(self, z: torch.Tensor, marginalized: Iterable[int]) -> torch.Tensor:
        """
        Exact log p(z_observed) with the given feature indices integrated out.
        The values of z at marginalized positions are ignored.
        """
        return eval_log_marginal(self.root, z, marginalized)

    def log_box(
        self,
        z: torch.Tensor,
        boxes: Dict[int, Tuple[float, float]],
        marginalized: Iterable[int] = (),
    ) -> torch.Tensor:
        """Exact axis-aligned box query; see RegionGraphPC.log_box."""
        return eval_log_marginal(self.root, z, marginalized, boxes=boxes)

    def log_partition(self) -> torch.Tensor:
        """Exact log partition function (≈ 0 by construction)."""
        return log_partition(self.root)

    def mpe(
        self, evidence: Optional[Dict[int, float]] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Most probable explanation given partial evidence {feature_idx: value}.
        Exact for deterministic circuits (e.g. n_sum_components=1); for
        non-deterministic circuits this is the standard max-product
        approximation.  Returns (assignment of shape (d,), log value).
        """
        return mpe(self.root, evidence)

    def anomaly_score(self, z: torch.Tensor) -> torch.Tensor:
        """NLL anomaly score: higher = more anomalous."""
        return -self.log_prob(z)

    # ── validation ───────────────────────────────────────────────────────

    def validate(self) -> None:
        """Assert smoothness, decomposability and structured decomposability."""
        validate_circuit(self.root)
        validate_structured_decomposability(self.root, self.vtree)


# ═══════════════════════════════════════════════════════════════════════════
# 5b. RegionGraphPC — the same circuit as a DAG instead of a tree
# ═══════════════════════════════════════════════════════════════════════════
#
# WHY THIS EXISTS.  DensityPC._build recurses independently for each of the K
# product components, so a leaf position is instantiated K^depth times: the
# parameter count is O(d · K^depth).  That is affordable for d ≈ 64 latent
# dimensions and fatal for anything wider.  A 14-sensor × 8-step window is
# d = 112, depth ≈ 7, K = 3 → 3^7 = 2187 copies per feature ≈ 245k leaf
# modules; a realistic 25 × 100 window is not representable at all.
#
# THE FIX (RAT-SPN / EiNet region-graph layout).  Every vtree node owns K
# *units* over its scope instead of K independent sub-circuits:
#
#     region(v)   = K sum units, all over scope(v)
#     products(v) = K_left × K_right binary products, pairing the child units
#     each sum unit of region(v) mixes ALL of products(v), with its own weights
#
# so the products are built once and SHARED by the K parents.  Size drops to
#     O(d · K²) products, O(d · K) sums, O(d · K) leaves
# i.e. linear in d at fixed K, and the depth exponent is gone.
#
# WHAT THIS COSTS: nothing structural.  Smoothness holds (every unit of a
# region has that region's scope, so a sum mixes equal-scope children);
# decomposability holds (products pair left-scope with right-scope, disjoint by
# the vtree); STRUCTURED decomposability holds w.r.t. the same vtree (every
# product splits exactly at a vtree node); determinism is unchanged (absent for
# full-support mixtures, present at K = 1) — exactly as in the tree layout.
# The validators need no changes: they already memoise by id(node).
#
# WHAT IT DOES COST: the naive recursive forward() must not be used, because a
# shared node would be recomputed once per path — parameters linear, compute
# exponential.  Evaluation therefore goes through the memoised module-level
# routines (eval_log_prob / eval_log_marginal / log_partition / mpe).


class RegionGraphPC(nn.Module):
    """
    Structured-decomposable PC laid out as a region-graph DAG.  Drop-in
    replacement for DensityPC — same constructor signature, same query API
    (log_prob / log_marginal / log_box / log_partition / mpe / anomaly_score /
    validate) — but O(d·K²) instead of O(d·K^depth) in size.

    Args:
        vtree_root:         vtree over feature indices.
        n_sum_components:   K, the number of units per internal region.
        leaf_factory:       feature_idx -> LeafNode.  A per-feature factory is
                            the intended way to mix leaf types (e.g. heavy-
                            tailed InputNode on sensors, CategoricalLeaf on a
                            discretised time-to-failure variable).
        n_input_components: units per LEAF region (defaults to K).  Setting it
                            to 1 gives a single shared leaf per feature.
        weight_jitter:      sd of the random perturbation applied to every sum
                            node's weights at construction.  REQUIRED for
                            learning, not a nicety — see below.
        seed:               seed for that perturbation.

    Why weight_jitter must be > 0
    -----------------------------
    In the tree layout each mixture component owns a private sub-circuit, so
    jittering the leaves is enough to tell components apart.  In the DAG the K
    units of a region are sum nodes over the SAME shared product list, so with
    uniform initial weights they are identical FUNCTIONS; and because each unit
    is paired symmetrically with every unit of the sibling region above it,
    they also receive identical gradients.  The symmetry is exact and it never
    breaks: all K units stay equal forever, every region collapses to one
    effective component, and the circuit silently degenerates into a single
    product of marginals.  Nothing in the training loss reveals this — it shows
    up only in a query, e.g. p(τ | x) coming out independent of x.
    """

    def __init__(
        self,
        vtree_root: Union[VtreeNode, RegionNode],
        n_sum_components: int = 3,
        leaf_factory: Optional[LeafFactory] = None,
        n_input_components: Optional[int] = None,
        weight_jitter: float = 0.5,
        seed: int = 0,
        pairing: str = "auto",
    ):
        super().__init__()
        # accept either a vtree (binary, single-partition) or a full region
        # graph (n-ary, optionally multi-partition).  The vtree is kept when it
        # was supplied, because structured-decomposability validation needs it.
        if isinstance(vtree_root, RegionNode):
            self.vtree = None
            self.region_graph = vtree_root
        else:
            self.vtree = vtree_root
            self.region_graph = region_graph_from_vtree(vtree_root)
        self.n_sum_components = int(n_sum_components)
        self.n_input_components = int(n_input_components or n_sum_components)
        self.leaf_factory = leaf_factory or InputNode
        self.pairing = pairing
        self._regions: Dict[int, List[nn.Module]] = {}
        self.root = self._build_rg(self.region_graph, n_units=1)[0]
        if weight_jitter > 0:
            gen = torch.Generator().manual_seed(int(seed))
            with torch.no_grad():
                for m in self.modules():
                    if isinstance(m, SumNode):
                        m.weights.add_(torch.randn(m.weights.shape, generator=gen)
                                       * weight_jitter)

    # ── construction ─────────────────────────────────────────────────────

    def _combine(self, child_units: List[List[nn.Module]]) -> List[nn.Module]:
        """
        Products for one partition.  Binary partitions use the FULL cross
        product (K² products, the standard EiNet layer).  Wider partitions use
        MATCHED-index pairing (K products), because the full cross product of an
        arity-a partition is K^a — the very blowup the region-graph layout
        exists to remove.
        """
        arity = len(child_units)
        mode = self.pairing
        if mode == "auto":
            mode = "full" if arity == 2 else "matched"
        if mode == "full":
            import itertools
            return [ProductNode(list(combo)) for combo in itertools.product(*child_units)]
        n = max(len(u) for u in child_units)
        return [ProductNode([u[j % len(u)] for u in child_units]) for j in range(n)]

    def _rg_child_units(self, region: RegionNode) -> int:
        return (self.n_input_components if region.is_leaf else self.n_sum_components)

    def _build_rg(self, region: RegionNode, n_units: int) -> List[nn.Module]:
        """
        Units of a region-graph node.  A region carrying several partitions
        mixes products from ALL of them in the same sum node — still smooth
        (every product has the region's scope) and decomposable (each partition
        has disjoint children), but NOT structured decomposable, which is the
        documented cost of multi-partition region graphs.
        """
        key = id(region)
        cached = self._regions.get(key)
        if cached is not None:
            return cached
        if region.is_leaf:
            units: List[nn.Module] = [self.leaf_factory(region.feature_idx)
                                      for _ in range(max(n_units, 1))]
            self._regions[key] = units
            return units
        products: List[nn.Module] = []
        for part in region.partitions:
            child_units = [self._build_rg(c, self._rg_child_units(c)) for c in part]
            products.extend(self._combine(child_units))
        if len(products) == 1:
            units = [products[0]]
        else:
            units = [SumNode(products) for _ in range(max(n_units, 1))]
        self._regions[key] = units
        return units

    def _build_region(self, node: VtreeNode, n_units: int) -> List[nn.Module]:
        """
        The `n_units` distributions over scope(node).  Memoised per vtree node
        so a region visited from several parents is built (and later shared)
        exactly once — that memo IS the DAG.  (A vtree is a tree, so the only
        node with a caller-chosen unit count is the root, which always gets 1.)
        """
        key = id(node)
        cached = self._regions.get(key)
        if cached is not None:
            return cached

        if isinstance(node, VtreeLeaf):
            units: List[nn.Module] = [
                self.leaf_factory(node.feature_idx)
                for _ in range(max(n_units, 1))
            ]
        else:
            left = self._build_region(node.left, self._child_units(node.left))
            right = self._build_region(node.right, self._child_units(node.right))
            # one product per (left unit, right unit) pair, built ONCE
            products = [ProductNode([l, r]) for l in left for r in right]
            if len(products) == 1:
                # a single component: the mixture is degenerate, so skip the
                # sum node entirely (keeps K=1 circuits deterministic)
                units = [products[0]]
            else:
                # every unit mixes the SAME product list with its own weights
                units = [SumNode(products) for _ in range(max(n_units, 1))]

        self._regions[key] = units
        return units

    def _child_units(self, node: VtreeNode) -> int:
        """Leaf regions get n_input_components units, internal regions get K."""
        return (self.n_input_components if isinstance(node, VtreeLeaf)
                else self.n_sum_components)

    # ── fitting ──────────────────────────────────────────────────────────

    def fit_leaves(self, X: torch.Tensor, jitter: float = 0.05) -> None:
        """Data-driven leaf initialisation; see DensityPC.fit_leaves.  Shared
        leaves are visited once (nn.Module.modules() de-duplicates)."""
        _fit_leaves_with_jitter(self, X, jitter)

    # ── exact inference (all memoised ⇒ linear in the DAG) ───────────────

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return eval_log_prob(self.root, z)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return eval_log_prob(self.root, z)

    def log_marginal(self, z: torch.Tensor, marginalized: Iterable[int]) -> torch.Tensor:
        return eval_log_marginal(self.root, z, marginalized)

    def log_box(
        self,
        z: torch.Tensor,
        boxes: Dict[int, Tuple[float, float]],
        marginalized: Iterable[int] = (),
    ) -> torch.Tensor:
        """
        log P(features in `boxes` fall in their intervals, others observed at z
        or marginalized).  Exact for smooth + decomposable circuits whose boxed
        leaves implement log_interval.  This is the survival / censoring query.
        """
        return eval_log_marginal(self.root, z, marginalized, boxes=boxes)

    def log_partition(self) -> torch.Tensor:
        return log_partition(self.root)

    def mpe(self, evidence: Optional[Dict[int, float]] = None) -> Tuple[torch.Tensor, float]:
        return mpe(self.root, evidence)

    def anomaly_score(self, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(z)

    # ── validation / introspection ───────────────────────────────────────

    def validate(self, require_structured: Optional[bool] = None) -> None:
        """
        Always checks smoothness + decomposability (the mandatory pair that
        guarantees exact density, marginals and box queries).

        Structured decomposability is checked only when the structure actually
        induces it — i.e. every region has one partition.  A multi-partition
        region graph deliberately gives it up; pass require_structured=True to
        assert it anyway (which will fail loudly, as it should).
        """
        validate_circuit(self.root)
        structured = is_structured_decomposable_rg(self.region_graph)
        if require_structured is None:
            require_structured = structured
        if not require_structured:
            return
        if not structured:
            raise AssertionError(
                "Structured decomposability was required but the region graph "
                "has regions with multiple partitions. Rebuild with "
                "n_partitions=1 if you need SOS / circuit multiplication.")
        validate_structured_decomposability(self.root, self.vtree or self.region_graph)

    @property
    def is_structured_decomposable(self) -> bool:
        return is_structured_decomposable_rg(self.region_graph)

    def size(self) -> Dict[str, int]:
        return circuit_size(self.root)


def circuit_size(root: nn.Module) -> Dict[str, int]:
    """
    Count DISTINCT nodes of a circuit (by identity, so DAG sharing is counted
    once) plus its trainable parameter count.  The tree-vs-DAG comparison this
    makes possible is the point of RegionGraphPC.
    """
    seen: set = set()
    counts = {"leaf": 0, "sum": 0, "product": 0}

    def _walk(node: nn.Module) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, LeafNode):
            counts["leaf"] += 1
            return
        if isinstance(node, (SumNode, _SignedSum)):
            counts["sum"] += 1
        elif isinstance(node, (ProductNode, _SignedProduct, ClassifierNode)):
            counts["product"] += 1
        for c in node.child_nodes:
            _walk(c)

    _walk(root)
    n_params = 0
    if isinstance(root, nn.Module):
        n_params = sum(p.numel() for p in root.parameters())
    return {**counts, "nodes": len(seen), "parameters": n_params}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Exact inference routines (module-level, work on any valid circuit)
# ═══════════════════════════════════════════════════════════════════════════

def eval_log_marginal(
    root: nn.Module,
    x: torch.Tensor,
    marginalized: Iterable[int] = (),
    boxes: Optional[Dict[int, Tuple[float, float]]] = None,
) -> torch.Tensor:
    """
    Exact marginal / box query on any smooth + decomposable circuit.

    Each feature is handled in exactly one of three ways:
      observed      (default)          leaf contributes log f(x_i)
      marginalized  (in `marginalized`) leaf contributes log ∫ f = 0
      boxed         (key in `boxes`)    leaf contributes log ∫_lo^hi f

    `boxes` maps feature index -> (lo, hi) with ±inf allowed; boxing is the
    strict generalisation of marginalizing ((-inf, +inf) is the same query),
    and it is what makes survival / censored-likelihood queries exact.

    MEMOISED BY id(node): the evaluation is a bottom-up pass over the circuit
    DAG in which every node is computed exactly once.  Without this, a node
    shared by several parents (which is the whole point of the region-graph
    layout in RegionGraphPC) would be recomputed once per path, making the
    COMPUTE exponential in depth even though the PARAMETERS are linear.
    """
    marg = frozenset(marginalized)
    boxes = dict(boxes or {})
    overlap = marg & set(boxes)
    if overlap:
        raise ValueError(f"features {sorted(overlap)} are both marginalized and boxed")
    cache: Dict[int, torch.Tensor] = {}
    zeros = x.new_zeros(x.shape[0])

    def _eval(node: nn.Module) -> torch.Tensor:
        nid = id(node)
        hit = cache.get(nid)
        if hit is not None:
            return hit
        if isinstance(node, LeafNode):
            i = node.feature_idx
            if i in boxes:
                lo, hi = boxes[i]
                out = node.log_interval(lo, hi).to(x.dtype) + zeros
            elif i in marg:
                out = node.log_integral().to(x.dtype) + zeros
            else:
                out = node.log_prob(x)
        elif isinstance(node, ProductNode):
            out = torch.stack([_eval(c) for c in node.child_nodes], dim=0).sum(dim=0)
        elif isinstance(node, SumNode):
            stack = torch.stack([_eval(c) for c in node.child_nodes], dim=0)
            out = torch.logsumexp(stack + node.log_weights().unsqueeze(-1), dim=0)
        else:
            raise TypeError(f"Unknown circuit node type: {type(node)}")
        cache[nid] = out
        return out

    return _eval(root)


def eval_log_prob(root: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """log p(x) on any valid circuit, DAG-safe (memoised).  Shape (B,)."""
    return eval_log_marginal(root, x, ())


def log_partition(root: nn.Module) -> torch.Tensor:
    """
    Exact log partition function: replace every leaf with its integral and
    propagate.  Returns ≈ 0 for circuits with normalized leaves and softmax
    sum weights.  Memoised by id(node), so it is linear in the DAG size.
    """
    cache: Dict[int, torch.Tensor] = {}

    def _eval(node: nn.Module) -> torch.Tensor:
        nid = id(node)
        hit = cache.get(nid)
        if hit is not None:
            return hit
        if isinstance(node, LeafNode):
            out = node.log_integral()
        elif isinstance(node, ProductNode):
            out = torch.stack([_eval(c) for c in node.child_nodes]).sum()
        elif isinstance(node, SumNode):
            stack = torch.stack([_eval(c) for c in node.child_nodes])
            out = torch.logsumexp(stack + node.log_weights(), dim=0)
        else:
            raise TypeError(f"Unknown circuit node type: {type(node)}")
        cache[nid] = out
        return out

    return _eval(root)


def mpe(
    root: nn.Module, evidence: Optional[Dict[int, float]] = None
) -> Tuple[torch.Tensor, float]:
    """
    Max-product upward pass with traceback.  Exact MPE for deterministic
    circuits; the standard max-product approximation otherwise.

    Returns (assignment tensor of shape (d,), log value), where d is
    max feature index + 1.
    """
    ev = dict(evidence or {})
    cache: Dict[int, Tuple[torch.Tensor, Dict[int, float]]] = {}

    def _mpe(node: nn.Module) -> Tuple[torch.Tensor, Dict[int, float]]:
        nid = id(node)
        hit = cache.get(nid)
        if hit is not None:
            return hit
        cache[nid] = out = _mpe_uncached(node)
        return out

    def _mpe_uncached(node: nn.Module) -> Tuple[torch.Tensor, Dict[int, float]]:
        if isinstance(node, LeafNode):
            idx = node.feature_idx
            value = ev[idx] if idx in ev else node.mode()
            log_val = node.log_density(torch.tensor(float(value)))
            return log_val, {idx: float(value)}
        if isinstance(node, ProductNode):
            total = torch.tensor(0.0)
            assignment: Dict[int, float] = {}
            for c in node.child_nodes:
                lv, a = _mpe(c)
                total = total + lv
                assignment.update(a)
            return total, assignment
        if isinstance(node, SumNode):
            log_w = node.log_weights()
            best_val, best_assign = None, None
            for k, c in enumerate(node.child_nodes):
                lv, a = _mpe(c)
                lv = lv + log_w[k]
                if best_val is None or lv > best_val:
                    best_val, best_assign = lv, a
            return best_val, best_assign
        raise TypeError(f"Unknown circuit node type: {type(node)}")

    log_val, assignment = _mpe(root)
    d = max(assignment) + 1
    out = torch.zeros(d)
    for i, v in assignment.items():
        out[i] = v
    return out, float(log_val)


# ═══════════════════════════════════════════════════════════════════════════
# 6b. Cardinality-constrained routing distribution (SIMPLE / ProbMoE)
# ═══════════════════════════════════════════════════════════════════════════
#
# Exact tractable inference over WHICH and HOW MANY experts a routed mixture
# selects, following ProbMoE (Zhao, Shao, Van den Broeck, Zeng, ICML 2026) and
# the SIMPLE estimator (Ahmed, Zeng, Niepert, Van den Broeck, ICLR 2023).
#
# Given N experts with per-expert Bernoulli selection probability p_i = σ(r_i),
# the unconstrained product measure over subsets S ⊆ [N] is
#     P(S) = ∏_{i∈S} p_i ∏_{i∉S} (1 − p_i).
# Conditioning on a chosen cardinality |S| = k gives the exact-k distribution
#     P_k(S) = (1/Z_k) ∏_{i∈S} p_i ∏_{i∉S}(1 − p_i),   Z_k = Σ_{|S|=k} ∏…,
# whose normaliser Z_k is the (weighted) elementary symmetric polynomial,
# computed exactly by an O(N·k) dynamic program — the recursion below.  From it:
#   • the cardinality posterior  P(|S|=k | k∈[k_min,k_max]) = Z_k / Σ_{k'} Z_{k'};
#   • the selection marginals    m_j = P(j ∈ S | constraint) = ∂ log Z / ∂ log p_j
#     (Eq. 6 of ProbMoE), obtained here by autograd, treating the non-selection
#     factors (1 − p_j) as constants exactly as the paper specifies.
#
# IMPORTANT (exactness invariant, see module docstring): this routing
# distribution is an AUXILIARY tractable query.  It is deliberately NOT used to
# define any density p(x): an input-conditional router would make the mixture
# weights depend on x and break normalisation.  The density side keeps
# data-independent SumNode weights; this section only supplies exact queries
# OVER THE ROUTING LATENT (cardinality, selection marginals) for use as
# additional anomaly signals alongside — never in place of — the exact density.


def cardinality_log_normalizers(
    log_p: torch.Tensor, log_1mp: torch.Tensor
) -> torch.Tensor:
    """
    Exact log-normalisers of the exact-k subset distribution, for every k.

    Args:
        log_p:   (B, N) log selection probabilities      log σ(r).
        log_1mp: (B, N) log non-selection probabilities   log σ(−r).
    Returns:
        logZ: (B, N+1) with logZ[:, k] = log Σ_{|S|=k} ∏_{i∈S} p_i ∏_{i∉S}(1−p_i).

    The forward dynamic program (a log-space convolution) is exact and fully
    differentiable.  logZ[:, 0] = Σ_i log(1−p_i) and Σ_k Z_k = 1.
    """
    if log_p.dim() != 2:
        raise ValueError(f"expected (B, N) log_p, got shape {tuple(log_p.shape)}")
    B, N = log_p.shape
    # Finite log-zero floor for unreachable cardinalities: exp(LOGZERO)=0 so the
    # normalisers stay exact, but logaddexp keeps a defined gradient (a true
    # −inf gives 0/0 = NaN backward when an exact-k marginal is differentiated).
    LOGZERO = torch.tensor(-1e30, dtype=log_p.dtype, device=log_p.device)
    neg_inf = LOGZERO.expand(B, 1)
    # dp[:, t] = log Σ over size-t subsets of the experts seen so far
    dp = torch.cat([torch.zeros(B, 1, dtype=log_p.dtype, device=log_p.device),
                    neg_inf.expand(B, N).clone()], dim=1)          # (B, N+1)
    for i in range(N):
        lp = log_p[:, i:i + 1]
        l1 = log_1mp[:, i:i + 1]
        keep = dp + l1                                             # i not selected
        shifted = torch.cat([neg_inf, dp[:, :-1]], dim=1)          # size t-1 → t
        take = shifted + lp                                        # i selected
        dp = torch.logaddexp(keep, take)
    return dp


def cardinality_log_posterior(
    logits: torch.Tensor, k_min: int = 1, k_max: Optional[int] = None
) -> torch.Tensor:
    """
    log P(|S| = k | k ∈ [k_min, k_max]) for the router, shape (B, k_max−k_min+1).

    `logits` are the per-expert router logits r (B, N); p_i = σ(r_i).  This is
    the ProbMoE Dynamic-k cardinality distribution (Eq. 9): the exact-k
    normalisers restricted to the feasible range and renormalised.
    """
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    logZ = cardinality_log_normalizers(log_p, log_1mp)             # (B, N+1)
    N = logits.shape[1]
    k_max = N if k_max is None else k_max
    if not (0 <= k_min <= k_max <= N):
        raise ValueError(f"need 0 ≤ k_min ≤ k_max ≤ N={N}, got [{k_min},{k_max}]")
    logZ_range = logZ[:, k_min:k_max + 1]
    return logZ_range - torch.logsumexp(logZ_range, dim=1, keepdim=True)


def cardinality_moments(
    logits: torch.Tensor, k_min: int = 1, k_max: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Summary statistics of the cardinality posterior P(k | x): the MAP cardinality
    k*, the expected cardinality E[k], and the Shannon entropy H[P(k|x)] (nats).
    All exact.  Each value has shape (B,).
    """
    N = logits.shape[1]
    k_max = N if k_max is None else k_max
    logpost = cardinality_log_posterior(logits, k_min, k_max)      # (B, R)
    post = logpost.exp()
    ks = torch.arange(k_min, k_max + 1, dtype=logits.dtype, device=logits.device)
    entropy = -(post * logpost).sum(dim=1)
    return {
        "map": k_min + torch.argmax(logpost, dim=1),
        "expected": (post * ks).sum(dim=1),
        "entropy": entropy,
    }


def selection_marginals(
    logits: torch.Tensor, k_min: int = 1, k_max: Optional[int] = None
) -> torch.Tensor:
    """
    Exact selection marginals m_j = P(j ∈ S | k ∈ [k_min, k_max]), shape (B, N).

    Computed as m_j = ∂ log Z / ∂ log p_j (ProbMoE Eq. 6) by autograd, holding
    the non-selection factors log(1−p_j) constant — exactly the paper's
    construction.  Z here is the range-marginal normaliser Σ_{k∈[k_min,k_max]}
    Z_k.  Row sums equal E[k | x].  Differentiates w.r.t. log p only (the
    logits are detached), so this is a routing-analysis query, not a path for
    training gradients to flow back into the experts.
    """
    N = logits.shape[1]
    k_max = N if k_max is None else k_max
    # local grad scope: m_j = ∂logZ/∂log p_j is computed by autograd even when
    # the caller is inside torch.no_grad() (e.g. routing-analysis at eval time).
    with torch.enable_grad():
        log_p = F.logsigmoid(logits).detach().clone().requires_grad_(True)
        log_1mp = F.logsigmoid(-logits).detach()
        logZ = cardinality_log_normalizers(log_p, log_1mp)         # (B, N+1)
        logZ_range = logZ[:, k_min:k_max + 1]
        log_total = torch.logsumexp(logZ_range, dim=1)             # (B,)
        (m,) = torch.autograd.grad(log_total.sum(), log_p)
    return m.detach()


def _compute_scope(
    node: nn.Module,
    cache: Optional[Dict] = None,
    check_smooth: bool = True,
    check_decomp: bool = True,
) -> FrozenSet[int]:
    """
    Recursively compute the scope of a circuit node, optionally verifying
    smoothness and decomposability as a side effect.  Memoised by node id, so
    DAG-shaped circuits are traversed once per node.
    """
    if cache is None:
        cache = {}
    nid = id(node)
    if nid in cache:
        return cache[nid]

    if isinstance(node, LeafNode):
        scope = frozenset({node.feature_idx})

    elif isinstance(node, (ProductNode, _SignedProduct)):
        child_scopes = [
            _compute_scope(c, cache, check_smooth, check_decomp) for c in node.child_nodes
        ]
        combined: FrozenSet[int] = frozenset()
        for i, si in enumerate(child_scopes):
            if check_decomp:
                for j in range(i + 1, len(child_scopes)):
                    sj = child_scopes[j]
                    assert si.isdisjoint(sj), (
                        f"Decomposability violated: ProductNode children {i} and {j} "
                        f"share feature(s) {si & sj}"
                    )
            combined = combined | si
        scope = combined

    elif isinstance(node, (SumNode, ClassifierNode, _SignedSum)):
        child_scopes = [
            _compute_scope(c, cache, check_smooth, check_decomp) for c in node.child_nodes
        ]
        if child_scopes:
            ref = child_scopes[0]
            if check_smooth:
                for i, cs in enumerate(child_scopes[1:], 1):
                    assert cs == ref, (
                        f"Smoothness violated: SumNode child 0 has scope {set(ref)} "
                        f"but child {i} has scope {set(cs)}"
                    )
            scope = frozenset().union(*child_scopes)
        else:
            scope = frozenset()

    else:
        raise TypeError(f"Unknown circuit node type: {type(node)}")

    cache[nid] = scope
    return scope


def validate_smoothness(root: nn.Module) -> None:
    """Property 1: every sum node's children share the same scope."""
    _compute_scope(root, check_smooth=True, check_decomp=False)


def validate_decomposability(root: nn.Module) -> None:
    """Property 2: every product node's children have disjoint scopes."""
    _compute_scope(root, check_smooth=False, check_decomp=True)


def validate_circuit(root: nn.Module) -> None:
    """
    Verify smoothness and decomposability (the two mandatory properties that
    guarantee exact density + exact marginals).  Raises AssertionError with a
    descriptive message on the first violation.
    """
    _compute_scope(root)


def validate_determinism(
    root: nn.Module, x: torch.Tensor, log_zero_threshold: float = -1e8
) -> None:
    """
    Property 3 (empirical check): for every complete assignment in x, at most
    one child of each sum node outputs a nonzero value (log output above
    `log_zero_threshold`).  Mixtures of full-support leaves (Gaussians) are
    NOT deterministic and will fail — that is the correct outcome.  Circuits
    with no sum nodes (n_sum_components=1) are trivially deterministic.
    """
    for module in root.modules() if hasattr(root, "modules") else [root]:
        if isinstance(module, SumNode):
            outs = torch.stack([c(x) for c in module.child_nodes], dim=0)  # (K, B)
            nonzero = (outs > log_zero_threshold).sum(dim=0)
            assert (nonzero <= 1).all(), (
                f"Determinism violated: a SumNode has {int(nonzero.max())} children "
                f"with nonzero output for at least one input assignment"
            )


def validate_structured_decomposability(
    root: nn.Module, structure: Union[VtreeNode, "RegionNode"]
) -> None:
    """
    Property 4: every product node splits its scope exactly as the given
    structure prescribes.

    `structure` may be a vtree (binary) or a single-partition region graph
    (n-ary).  Multi-partition region graphs do not induce structured
    decomposability at all and are rejected up front with an explanation
    rather than a confusing per-product failure.
    """
    if isinstance(structure, RegionNode):
        bad = [r for r in region_nodes(structure) if len(r.partitions) > 1]
        if bad:
            raise AssertionError(
                f"{len(bad)} region(s) carry multiple partitions, so this region "
                "graph does not induce structured decomposability by "
                "construction (exact density/marginals/box queries are "
                "unaffected; SOS and circuit multiplication are not available).")
        allowed = {
            frozenset(c.scope for c in r.partitions[0])
            for r in region_nodes(structure) if r.partitions
        }
    else:
        allowed = {
            frozenset({n.left_scope, n.right_scope})
            for n in vtree_nodes(structure)
            if isinstance(n, VtreeInternal)
        }
    cache: Dict = {}
    seen: set = set()

    def _check(node: nn.Module) -> None:
        if id(node) in seen:      # DAG-safe: each node checked once, not once per path
            return
        seen.add(id(node))
        if isinstance(node, LeafNode):
            return
        if isinstance(node, (ProductNode, _SignedProduct)):
            partition = frozenset(
                _compute_scope(c, cache, False, False) for c in node.child_nodes
            )
            assert partition in allowed, (
                f"Structured decomposability violated: product splits scope into "
                f"{[set(s) for s in partition]}, which matches no vtree node"
            )
        for c in node.child_nodes:
            _check(c)

    _check(root)


def extract_vtree(root: nn.Module) -> VtreeNode:
    """
    Extract the implicit vtree from a circuit built by DensityPC or a
    structurally compatible hand-built circuit.

      LeafNode      → VtreeLeaf
      ProductNode   → VtreeInternal (binary)
      Sum/Classifier→ transparent (delegates to first child)

    Raises ValueError if a ProductNode has anything other than 2 children.
    Memoised by id(node) so it is linear on DAG-shaped circuits.
    """
    cache: Dict[int, VtreeNode] = {}

    def _extract(node: nn.Module) -> VtreeNode:
        nid = id(node)
        hit = cache.get(nid)
        if hit is not None:
            return hit
        cache[nid] = out = _extract_uncached(node)
        return out

    def _extract_uncached(node: nn.Module) -> VtreeNode:
        if isinstance(node, LeafNode):
            return VtreeLeaf(node.feature_idx)
        if isinstance(node, (ProductNode, _SignedProduct)):
            if len(node.child_nodes) != 2:
                raise ValueError(
                    f"extract_vtree requires binary ProductNodes; "
                    f"found {len(node.child_nodes)} children"
                )
            return VtreeInternal(
                left=_extract(node.child_nodes[0]),
                right=_extract(node.child_nodes[1]),
            )
        if isinstance(node, (SumNode, ClassifierNode, _SignedSum)):
            return _extract(node.child_nodes[0])
        raise TypeError(f"Cannot extract vtree from node type: {type(node)}")

    return _extract(root)


# ═══════════════════════════════════════════════════════════════════════════
# 8. SOS mode — squared circuits (subtractive mixtures)
# ═══════════════════════════════════════════════════════════════════════════

class _SignedSum(nn.Module):
    """Sum node with unconstrained real weights (may be negative)."""

    def __init__(self, child_nodes, init_scale: float = 0.5, seed_tensor: Optional[torch.Tensor] = None):
        super().__init__()
        self.child_nodes = nn.ModuleList(child_nodes)
        n = len(child_nodes)
        init = seed_tensor if seed_tensor is not None else 1.0 + init_scale * torch.randn(n)
        self.weights = nn.Parameter(init)


class _SignedProduct(nn.Module):
    """Binary product node for the SOS circuit."""

    def __init__(self, child_nodes):
        super().__init__()
        self.child_nodes = nn.ModuleList(child_nodes)


def _signed_lse(logabs: torch.Tensor, signs: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-sum-exp over signed values represented as (log|v|, sign)."""
    m = logabs.max(dim=dim, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = (signs * torch.exp(logabs - m)).sum(dim=dim)
    return m.squeeze(dim) + torch.log(s.abs() + 1e-38), torch.sign(s)


class SquaredPC(nn.Module):
    """
    Sum-of-squares PC (Loconte et al., AAAI 2025): a structured-decomposable
    circuit c(x) with real-valued (possibly negative) sum weights, modeling

        p(x) = c(x)² / Z,    Z = ∫ c(x)² dx.

    Z is computed exactly via the pairwise (squared-circuit) recursion, which
    is tractable because the circuit is structured decomposable: products
    align left-left / right-right on the shared vtree.  Marginals use the
    same recursion with observed leaves evaluated and marginalized leaves
    pair-integrated, so they are exact as well.

    Leaves must implement pair_log_integral (GaussianLeaf,
    GaussianMixtureLeaf, CategoricalLeaf).  MPE is not supported (hard for
    squared circuits).
    """

    def __init__(
        self,
        vtree_root: VtreeNode,
        n_sum_components: int = 3,
        leaf_factory: Optional[LeafFactory] = None,
        seed: int = 0,
        region_graph: bool = False,
    ):
        """
        region_graph=True lays the squared circuit out as a shared-region DAG
        (see RegionGraphPC) instead of a tree, so it is O(d·K²) rather than
        O(d·K^depth) and wide inputs become representable.  Default stays False
        for bit-exact reproduction of previously published tree-layout runs.

        Cost note: the exact partition function pairs every node with every
        node in the same region, so SOS is quadratic in K where the monotone
        circuit is linear — K^4 product pairs per region.  Keep K small (2–4).
        """
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self._gen = gen
        self.n_sum_components = n_sum_components
        self.leaf_factory = leaf_factory or GaussianLeaf
        self._regions: Dict[int, List[nn.Module]] = {}

        if isinstance(vtree_root, RegionNode):
            # Squaring is tractable ONLY for structured-decomposable circuits
            # (Loconte et al., AAAI 2025): the pairwise recursion multiplies two
            # copies of the circuit by aligning children position-by-position,
            # which requires both copies to decompose their scope identically.
            # A region with two different partitions has no such alignment and
            # the partition function stops being computable in polynomial time.
            if not is_structured_decomposable_rg(vtree_root):
                raise ValueError(
                    "SquaredPC needs a structured-decomposable structure, but this "
                    "region graph has regions with multiple partitions. Rebuild it "
                    "with n_partitions=1, or use the monotone RegionGraphPC (whose "
                    "density, marginals and box queries stay exact either way).")
            self.vtree = None
            self.rg = vtree_root
            self.region_graph = True
            self.root = self._build_rg_signed(vtree_root, n_units=1)[0]
        else:
            self.vtree = vtree_root
            self.rg = region_graph_from_vtree(vtree_root)
            self.region_graph = region_graph
            self.root = (self._build_region(vtree_root, n_units=1)[0] if region_graph
                         else self._build(vtree_root))

    def _build(self, node: VtreeNode) -> nn.Module:
        if isinstance(node, VtreeLeaf):
            return self.leaf_factory(node.feature_idx)
        prods = [
            _SignedProduct([self._build(node.left), self._build(node.right)])
            for _ in range(self.n_sum_components)
        ]
        if len(prods) == 1:
            return prods[0]
        init = 1.0 + 0.5 * torch.randn(len(prods), generator=self._gen)
        return _SignedSum(prods, seed_tensor=init)

    def _build_rg_signed(self, region: RegionNode, n_units: int) -> List[nn.Module]:
        """Signed shared-region build over an n-ary, single-partition region
        graph.  Wider partitions use matched-index pairing, exactly as in the
        monotone circuit, so the pairwise partition recursion still aligns
        children position by position."""
        key = id(region)
        cached = self._regions.get(key)
        if cached is not None:
            return cached
        K = self.n_sum_components
        if region.is_leaf:
            units: List[nn.Module] = [self.leaf_factory(region.feature_idx)
                                      for _ in range(max(n_units, 1))]
            self._regions[key] = units
            return units
        part = region.partitions[0]
        child_units = [self._build_rg_signed(c, K) for c in part]
        if len(part) == 2:
            import itertools
            products = [_SignedProduct(list(combo))
                        for combo in itertools.product(*child_units)]
        else:
            n = max(len(u) for u in child_units)
            products = [_SignedProduct([u[j % len(u)] for u in child_units])
                        for j in range(n)]
        if len(products) == 1:
            units = [products[0]]
        else:
            units = []
            for _ in range(max(n_units, 1)):
                init = 1.0 + 0.5 * torch.randn(len(products), generator=self._gen)
                units.append(_SignedSum(products, seed_tensor=init))
        self._regions[key] = units
        return units

    def _build_region(self, node: VtreeNode, n_units: int) -> List[nn.Module]:
        """Shared-region DAG build; mirrors RegionGraphPC._build_region.  The
        signed weights are already random at init, so the unit-symmetry problem
        that forces weight_jitter in the monotone circuit does not arise."""
        key = id(node)
        cached = self._regions.get(key)
        if cached is not None:
            return cached
        K = self.n_sum_components
        if isinstance(node, VtreeLeaf):
            units: List[nn.Module] = [self.leaf_factory(node.feature_idx)
                                      for _ in range(max(n_units, 1))]
        else:
            left = self._build_region(node.left, K)
            right = self._build_region(node.right, K)
            products = [_SignedProduct([l, r]) for l in left for r in right]
            if len(products) == 1:
                units = [products[0]]
            else:
                units = []
                for _ in range(max(n_units, 1)):
                    init = 1.0 + 0.5 * torch.randn(len(products), generator=self._gen)
                    units.append(_SignedSum(products, seed_tensor=init))
        self._regions[key] = units
        return units

    # ── signed forward: c(x) in (log|·|, sign) form ──────────────────────

    def _signed_forward(
        self, node: nn.Module, x: torch.Tensor, cache: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # memoised by node identity: mandatory under the region-graph layout,
        # where a node is reached from many parents (harmless for the tree)
        if cache is None:
            cache = {}
        nid = id(node)
        hit = cache.get(nid)
        if hit is not None:
            return hit
        if isinstance(node, LeafNode):
            la = node.log_prob(x)
            out = (la, torch.ones_like(la))
        elif isinstance(node, _SignedProduct):
            la_l, s_l = self._signed_forward(node.child_nodes[0], x, cache)
            la_r, s_r = self._signed_forward(node.child_nodes[1], x, cache)
            out = (la_l + la_r, s_l * s_r)
        elif isinstance(node, _SignedSum):
            las, ss = zip(*[self._signed_forward(c, x, cache) for c in node.child_nodes])
            w = node.weights
            la = torch.stack(las, dim=0) + torch.log(w.abs() + 1e-38).unsqueeze(-1)
            s = torch.stack(ss, dim=0) * torch.sign(w).unsqueeze(-1)
            out = _signed_lse(la, s, dim=0)
        else:
            raise TypeError(f"Unknown SOS node type: {type(node)}")
        cache[nid] = out
        return out

    # ── pairwise recursion: ∫ c_n c_m over marginalized vars, evaluated
    #    at x on observed vars ─────────────────────────────────────────────

    def _pair_eval(
        self,
        n: nn.Module,
        m: nn.Module,
        x: Optional[torch.Tensor],
        marg: FrozenSet[int],
        cache: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (id(n), id(m))
        if key in cache:
            return cache[key]

        if isinstance(n, LeafNode) and isinstance(m, LeafNode):
            if n.feature_idx in marg:
                la, s = n.pair_log_integral(m)
                if x is not None:
                    la = la + x.new_zeros(x.shape[0])
                    s = s + torch.zeros_like(la)  # broadcast sign to (B,)
            else:
                la = n.log_prob(x) + m.log_prob(x)
                s = torch.ones_like(la)
            out = (la, s)
        elif isinstance(n, _SignedProduct) and isinstance(m, _SignedProduct):
            # align children position by position: valid because the structure
            # is structured decomposable, so both copies split the scope alike
            if len(n.child_nodes) != len(m.child_nodes):
                raise TypeError("misaligned SOS products (structure is not "
                                "structured decomposable)")
            la, sg = None, None
            for cn, cm in zip(n.child_nodes, m.child_nodes):
                l_i, s_i = self._pair_eval(cn, cm, x, marg, cache)
                la, sg = (l_i, s_i) if la is None else (la + l_i, sg * s_i)
            out = (la, sg)
        elif isinstance(n, _SignedSum) and isinstance(m, _SignedSum):
            las, ss = [], []
            for a, ca in enumerate(n.child_nodes):
                for b, cb in enumerate(m.child_nodes):
                    la, s = self._pair_eval(ca, cb, x, marg, cache)
                    las.append(la + torch.log(n.weights[a].abs() + 1e-38)
                               + torch.log(m.weights[b].abs() + 1e-38))
                    ss.append(s * torch.sign(n.weights[a]) * torch.sign(m.weights[b]))
            out = _signed_lse(torch.stack(las, dim=0), torch.stack(ss, dim=0), dim=0)
        else:
            raise TypeError(f"Mismatched SOS node pair: {type(n)} / {type(m)}")

        cache[key] = out
        return out

    # ── public API (mirrors DensityPC) ───────────────────────────────────

    def fit_leaves(self, X: torch.Tensor, jitter: float = 0.05) -> None:
        _fit_leaves_with_jitter(self, X, jitter)

    def log_partition(self) -> torch.Tensor:
        """Exact log Z = log ∫ c(x)² dx via the pairwise recursion."""
        scope = frozenset(vtree_leaves(self.vtree)) if self.vtree is not None \
            else frozenset(self.rg.scope)
        la, s = self._pair_eval(self.root, self.root, None, scope, {})
        assert float(s) > 0, "SOS partition function must be positive"
        return la

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Exact log p(z) = 2 log|c(z)| − log Z, shape (B,)."""
        la, _ = self._signed_forward(self.root, z)
        return 2.0 * la - self.log_partition()

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)

    def log_marginal(self, z: torch.Tensor, marginalized: Iterable[int]) -> torch.Tensor:
        """Exact marginal of the squared density (pairwise recursion)."""
        marg = frozenset(marginalized)
        la, _ = self._pair_eval(self.root, self.root, z, marg, {})
        return la - self.log_partition()

    def anomaly_score(self, z: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(z)

    def validate(self) -> None:
        validate_circuit(self.root)
        validate_structured_decomposability(self.root, self.vtree or self.rg)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Legacy PCNet (kept for reference / ablation)
# ═══════════════════════════════════════════════════════════════════════════

class PCNet(nn.Module):
    """
    Original random-structure PCNet from bak/.
    Not guaranteed to be smooth or decomposable.
    Use DensityPC for formally valid circuits.
    """

    def __init__(self, n_classes: int = 2, max_depth: int = 4,
                 max_branching: int = 3, seed: int = 42):
        super().__init__()
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.seed = seed
        self.root: Optional[nn.Module] = None
        self.vtree: Optional[VtreeNode] = None

    def init_network(self, inputs: torch.Tensor) -> Optional[VtreeNode]:
        """
        Build the random circuit and record the implied scope-partition tree.
        Returns the extracted vtree (best-effort; may not be fully valid).
        """
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        C = inputs.shape[1]

        current_nodes: list = []
        for i in range(C):
            node = InputNode(i)
            node.fit(inputs)
            current_nodes.append(node)

        self.leaves = nn.ModuleList(current_nodes)

        nodes = current_nodes
        for depth in range(1, self.max_depth + 1):
            if depth == self.max_depth:
                nodes = self._reduce_to_n(nodes, self.n_classes)
                self.root = ClassifierNode(nodes)
                break
            if depth % 2 == 0:
                nodes = self._prod_layer(nodes)
            else:
                nodes = self._sum_layer(nodes)

        if self.root is None:
            self.root = ClassifierNode(self._reduce_to_n(current_nodes, self.n_classes))

        try:
            self.vtree = extract_vtree(self.root.child_nodes[0])
        except (ValueError, IndexError, TypeError):
            self.vtree = None

        return self.vtree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.root is None:
            raise RuntimeError("Call init_network first.")
        return self.root(x)

    # ── internal helpers ─────────────────────────────────────────────────

    def _pairwise(self, nodes):
        pool = list(nodes)
        random.shuffle(pool)
        pairs = []
        while len(pool) >= 2:
            pairs.append((pool.pop(), pool.pop()))
        if pool:
            pairs.append((pool.pop(),))
        return pairs

    def _prod_layer(self, nodes):
        next_nodes = []
        for pair in self._pairwise(nodes):
            if len(pair) == 1:
                next_nodes.append(pair[0])
            else:
                next_nodes.append(ProductNode(list(pair)))
        return next_nodes

    def _sum_layer(self, nodes):
        next_nodes = []
        pool = list(nodes)
        random.shuffle(pool)
        while pool:
            k = random.randint(2, self.max_branching)
            group = [pool.pop() for _ in range(min(k, len(pool)))]
            if len(group) == 1:
                next_nodes.append(group[0])
                continue
            next_nodes.append(SumNode(group))
        return next_nodes

    def _reduce_to_n(self, nodes, n):
        if len(nodes) < n:
            return nodes + [nodes[-1]] * (n - len(nodes))
        chunk = int(np.ceil(len(nodes) / n))
        reduced = []
        for i in range(0, len(nodes), chunk):
            group = nodes[i: i + chunk]
            if len(group) == 1:
                reduced.append(group[0])
            else:
                reduced.append(SumNode(group))
        return reduced[:n]
