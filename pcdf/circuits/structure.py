"""
Structure learning for the deepfake circuits — all of it delegated to
`src/probabilistic_circuits.py`, which is the audited implementation.

Two levels of structure are used, and they answer different questions.

FLAT (`build_structure`)
    A region graph over all d coordinates, learned by one of:
      chow_liu   MI max-spanning-tree, weakest-edge recursion (the incumbent)
      orc        Ollivier-Ricci curvature bottleneck cuts, exact W₁ per edge
      orc_multi  same, but keeping several cut depths as ALTERNATIVE partitions
                 (strictly more expressive; gives up structured decomposability)
      forman     closed-form curvature — the cheap proxy for large d
      spectral   recursive normalized cut (the honest adversary for curvature)
      random     balanced random region graph (the collapse control)
    Product nodes assert independence across a cut, and the exact modelling
    error of that assertion is the mutual information crossing it — so where to
    cut is the entire structure-learning problem, and curvature scores a cut by
    the transport cost between whole neighborhoods rather than one pairwise MI.

HIERARCHICAL (`build_image_structure`)
    For patch-token features the variables are (patch p, channel c) with
    index p·C + c, and a flat learner on d = P·C is both intractable (ORC needs
    an exact W₁ per edge) and structurally blind.  So the graph is composed:

        patch level    learn over P super-variables (patch summaries) with any
                       method above, or use the kd-tree spatial prior
        channel level  learn once over C channels, instantiated per patch

    The payoff is not just tractability: EVERY patch is a region of the
    resulting graph, so `log p(z_patch)` and `log p(z_patch | rest)` are exact
    queries on scopes the circuit already represents — localization comes from
    the structure, not from a post-hoc attribution heuristic.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..pclib import (
    RegionNode,
    chow_liu_vtree,
    curvature_region_graph,
    curvature_sign_stability,
    forman_curvature,
    is_structured_decomposable_rg,
    mutual_information_matrix,
    ollivier_ricci_curvature,
    random_balanced_vtree,
    region_graph_arity,
    region_graph_from_vtree,
    region_nodes,
    sparsify_mi_graph,
    spectral_region_graph,
)

FLAT_METHODS = ("chow_liu", "orc", "orc_multi", "forman", "spectral", "random")


# ── flat structures ─────────────────────────────────────────────────────────

def build_structure(
    X: np.ndarray,
    method: str = "orc",
    max_arity: int = 4,
    n_partitions: int = 2,
    k: Optional[int] = None,
    seed: int = 0,
    **kwargs,
) -> RegionNode:
    """Region graph over the columns of X.  See module docstring for methods."""
    X = np.asarray(X, dtype=np.float32)
    d = X.shape[1]
    if method == "chow_liu":
        return region_graph_from_vtree(chow_liu_vtree(X, **kwargs))
    if method == "orc":
        return curvature_region_graph(X, curvature="ollivier", n_partitions=1,
                                      max_arity=max_arity, k=k, **kwargs)
    if method == "orc_multi":
        return curvature_region_graph(X, curvature="ollivier",
                                      n_partitions=n_partitions,
                                      max_arity=max_arity, k=k, **kwargs)
    if method == "forman":
        return curvature_region_graph(X, curvature="forman", n_partitions=1,
                                      max_arity=max_arity, k=k, **kwargs)
    if method == "spectral":
        return spectral_region_graph(X, max_arity=max_arity, **kwargs)
    if method == "random":
        return region_graph_from_vtree(random_balanced_vtree(list(range(d)), seed=seed))
    raise KeyError(f"unknown structure method {method!r}; use one of {FLAT_METHODS}")


# ── hierarchical (image) structures ─────────────────────────────────────────

def _remap(rg: RegionNode, mapping: Sequence[int]) -> RegionNode:
    """Rebuild a region graph with feature i renamed to mapping[i]."""
    cache: Dict[int, RegionNode] = {}

    def walk(r: RegionNode) -> RegionNode:
        if id(r) in cache:
            return cache[id(r)]
        scope = frozenset(mapping[i] for i in r.scope)
        new = RegionNode(scope)
        cache[id(r)] = new
        new.partitions = [tuple(walk(c) for c in part) for part in r.partitions]
        return new

    return walk(rg)


def kd_patch_region_graph(grid_h: int, grid_w: int) -> RegionNode:
    """
    Spatial prior over the patch grid: recursive alternating horizontal /
    vertical halving, so every region is a contiguous rectangle.  This is the
    hand-built adversary for the learned patch structures — if it matches ORC
    on held-out NLL, learning spatial structure buys nothing (structure itself
    still might).
    """
    cache: Dict[frozenset, RegionNode] = {}

    def build(patches: List[Tuple[int, int]], horizontal: bool) -> RegionNode:
        scope = frozenset(r * grid_w + c for r, c in patches)
        hit = cache.get(scope)
        if hit is not None:
            return hit
        node = RegionNode(scope)
        cache[scope] = node
        if len(patches) > 1:
            key = 0 if horizontal else 1
            ps = sorted(patches, key=lambda rc: (rc[key], rc[1 - key]))
            mid = len(ps) // 2
            node.partitions = [(build(ps[:mid], not horizontal),
                                build(ps[mid:], not horizontal))]
        return node

    return build([(r, c) for r in range(grid_h) for c in range(grid_w)], True)


def patch_summaries(Z: np.ndarray, n_patches: int, n_channels: int) -> np.ndarray:
    """
    (N, P·C) -> (N·?, P): a scalar per patch used to estimate patch-to-patch
    dependence.  The projection is the leading principal direction of the
    pooled channel data, i.e. the single most informative linear summary; using
    the channel mean instead would throw away sign structure.
    """
    N = Z.shape[0]
    blocks = Z.reshape(N, n_patches, n_channels)
    pooled = blocks.reshape(-1, n_channels)
    pooled = pooled - pooled.mean(0, keepdims=True)
    # leading right singular vector, on a subsample for speed
    sub = pooled[np.random.default_rng(0).choice(
        len(pooled), size=min(len(pooled), 20000), replace=False)]
    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    return (blocks @ Vt[0]).astype(np.float32)          # (N, P)


def build_image_structure(
    Z: np.ndarray,
    grid_h: int,
    grid_w: int,
    n_channels: int,
    patch_method: str = "kd",
    channel_method: str = "orc",
    max_arity: int = 4,
    k: Optional[int] = None,
    seed: int = 0,
    channel_sample: int = 20000,
) -> Tuple[RegionNode, Dict[int, RegionNode]]:
    """
    Compose a patch-level region graph with a per-patch channel region graph.

    Returns (root region graph, {patch index -> its RegionNode}).  The second
    return value is what the explainer queries: each patch's region is a real
    node of the circuit, so its marginal and conditional are exact.
    """
    Z = np.asarray(Z, dtype=np.float32)
    P = grid_h * grid_w
    assert Z.shape[1] == P * n_channels, \
        f"expected {P * n_channels} features, got {Z.shape[1]}"

    # channel structure: learned once on channel vectors pooled over patches
    pooled = Z.reshape(-1, n_channels)
    rng = np.random.default_rng(seed)
    if len(pooled) > channel_sample:
        pooled = pooled[rng.choice(len(pooled), channel_sample, replace=False)]
    channel_rg = build_structure(pooled, method=channel_method,
                                 max_arity=max_arity, k=k, seed=seed)

    # patch structure
    if patch_method == "kd":
        patch_rg = kd_patch_region_graph(grid_h, grid_w)
    else:
        patch_rg = build_structure(patch_summaries(Z, P, n_channels),
                                   method=patch_method, max_arity=max_arity,
                                   k=k, seed=seed)

    # expand: every patch super-variable becomes its channel block
    patch_regions: Dict[int, RegionNode] = {}
    cache: Dict[int, RegionNode] = {}

    def expand(r: RegionNode) -> RegionNode:
        if id(r) in cache:
            return cache[id(r)]
        if r.is_leaf:
            p = r.feature_idx
            block = _remap(channel_rg, [p * n_channels + c for c in range(n_channels)])
            patch_regions[p] = block
            cache[id(r)] = block
            return block
        scope = frozenset(p * n_channels + c for p in r.scope for c in range(n_channels))
        new = RegionNode(scope)
        cache[id(r)] = new
        new.partitions = [tuple(expand(c) for c in part) for part in r.partitions]
        return new

    root = expand(patch_rg)
    return root, patch_regions


# ── diagnostics, persistence ────────────────────────────────────────────────

def structure_report(rg: RegionNode) -> Dict[str, float]:
    regions = region_nodes(rg)
    sizes = [len(r.scope) for r in regions]
    return {
        "n_regions": len(regions),
        "n_leaf_regions": sum(1 for r in regions if r.is_leaf),
        "max_arity": region_graph_arity(rg),
        "structured_decomposable": int(is_structured_decomposable_rg(rg)),
        "mean_region_scope": float(np.mean(sizes)),
        "max_region_scope": int(max(sizes)),
        "multi_partition_regions": sum(1 for r in regions if len(r.partitions) > 1),
    }


def curvature_diagnostics(X: np.ndarray, k: Optional[int] = None,
                          n_boot: int = 10, seed: int = 0) -> Dict[str, float]:
    """
    Is the curvature signal real or an artefact of MI noise?  Reports the
    fraction of edges whose Ollivier-Ricci sign survives bootstrap resampling,
    plus the ORC/Forman rank agreement — cheap evidence for or against the
    geometry story before spending a training run on it.
    """
    from scipy.stats import spearmanr

    X = np.asarray(X, dtype=np.float32)
    M = mutual_information_matrix(X)
    A = sparsify_mi_graph(M, k=k)
    orc = ollivier_ricci_curvature(A)
    frm = forman_curvature(A)
    edges = sorted(orc)
    stab = curvature_sign_stability(X, curvature="ollivier", n_boot=n_boot,
                                    k=k, seed=seed)
    rho = float(spearmanr([orc[e] for e in edges], [frm[e] for e in edges]).statistic)
    return {
        "n_edges": len(edges),
        "frac_negative_curvature": float(np.mean([orc[e] < 0 for e in edges])),
        "mean_sign_stability": float(np.mean(list(stab.values()))),
        "frac_edges_stable_90": float(np.mean([v >= 0.9 for v in stab.values()])),
        "orc_forman_rank_corr": rho,
    }


def save_structure(rg: RegionNode, path: str | Path) -> None:
    """Structure learning is the slow part; cache it next to the features."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(rg, fh)
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump(structure_report(rg), fh, indent=2)


def load_structure(path: str | Path) -> RegionNode:
    with open(path, "rb") as fh:
        return pickle.load(fh)
