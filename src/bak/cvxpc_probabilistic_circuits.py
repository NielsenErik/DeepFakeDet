# ============================================================
# Imports
# ============================================================
import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Union
import os

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.stats import norm
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import gabor, laplace, sobel
from torch import nn
from tqdm import trange
from collections import Counter

from utils import print_configs, print_debugging, print_info


# ======================================================================
# THE FOUR PROBABILISTIC-CIRCUIT PROPERTIES (enforced by this module)
# ----------------------------------------------------------------------
# A probabilistic circuit is a VALID, TRACTABLE density p(x) only if its
# structure satisfies the structural properties of Choi, Vergari & Van den
# Broeck, "Probabilistic Circuits: A Unifying Framework for Tractable
# Probabilistic Models" (2020). For this project the queries we need are
# (i) exact log p(x) for per-pixel scoring/segmentation and (ii) exact
# MARGINALS over arbitrary feature subsets — the latter is the explicit
# requirement of Khosravi, Vergari & Van den Broeck, "Why Is This an
# Outlier? Explaining Outliers by Submodular Optimization of Marginal
# Distributions" (TPM 2022): outlier-feature detection queries the joint
# p_theta(x_S) for many subsets S, which is tractable *iff* the circuit is
# smooth + decomposable.
#
#   1. SMOOTHNESS (completeness): every SumNode's children share ONE scope.
#        -> enforced in SumNode.__init__ ; checked by validate_smoothness.
#        Needed so a sum integrates to a normalized mixture (and so
#        marginalizing a variable is exact).
#
#   2. DECOMPOSABILITY: every ProductNode's children have DISJOINT scopes.
#        -> enforced in ProductNode.__init__ ; checked by
#        validate_decomposability. Needed so a product factorizes the
#        density over disjoint blocks (and so a marginal factorizes too).
#
#   3. DETERMINISM (selectivity): for any complete input at most one child
#        of each sum is non-zero. Only required for exact MAP/MPE. Our sum
#        nodes are SOFT mixtures of full-support leaves, so determinism is
#        intentionally NOT satisfied (validate_determinism documents this).
#
#   4. STRUCTURED DECOMPOSABILITY: every product over a given scope splits
#        it the SAME way, prescribed by a single vtree. We build every
#        sub-circuit top-down from one shared vtree, so this holds by
#        construction; checked by validate_structured_decomposability.
#
# Normalization (Z = 1) is a consequence, not a separate property: the
# leaves are normalized univariate densities and every mixture weight is a
# (log_)softmax, so smoothness + decomposability + normalized leaves give a
# circuit with partition function 1. The previous implementation broke this
# by multiplying each leaf log-density by a learnable `gate`; that scaling
# is removed here (the `gate` attribute is kept, fixed to 1.0, only for
# checkpoint / diagnostic back-compat).
# ======================================================================


# ======================================================================
# JIT math kernels (normalized univariate log-densities)
# ======================================================================
@torch.jit.script
def log_gaussian_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # 0.5 * log(2*pi) ~= 0.9189385
    return -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.9189385


@torch.jit.script
def log_laplace_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # scale b = sigma / sqrt(2); log(2b) = log(sigma) + 0.5*log(2) = log(sigma) + 0.3465736
    b = sigma * 0.70710678
    return -torch.abs(x - mu) / b - (torch.log(sigma) + 0.3465736)


@torch.jit.script
def log_student_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    z = (x - mu) / (sigma + 1e-12)
    term1 = torch.lgamma((nu + 1) * 0.5)
    term2 = torch.lgamma(nu * 0.5)
    term3 = 0.5 * torch.log(nu * 3.14159265)
    term4 = torch.log(sigma + 1e-12)
    term5 = ((nu + 1) * 0.5) * torch.log1p((z ** 2) / nu)
    return term1 - term2 - term3 - term4 - term5


def _scope_of(node) -> FrozenSet[int]:
    """Scope of a circuit child, tolerant of non-Node wrappers (e.g. SubPCNet
    experts in MultiPCNet) which expose no scope."""
    return getattr(node, "scope", frozenset())


# ======================================================================
# Vtree (variable-tree): the single scope-partition hierarchy that makes
# every product node decompose identically -> structured decomposability.
# ======================================================================
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


def vtree_nodes(node: VtreeNode) -> List[VtreeNode]:
    if isinstance(node, VtreeLeaf):
        return [node]
    return [node] + vtree_nodes(node.left) + vtree_nodes(node.right)


def random_balanced_vtree(features, seed: int = 0) -> VtreeNode:
    """Shuffle the features, then split at the midpoint recursively (balanced
    binary vtree). Any binary vtree yields a structured-decomposable circuit;
    only the *quality* of the structure changes with the split rule."""
    feats = list(features)
    random.Random(seed).shuffle(feats)

    def _build(fs):
        if len(fs) == 1:
            return VtreeLeaf(fs[0])
        mid = len(fs) // 2
        return VtreeInternal(_build(fs[:mid]), _build(fs[mid:]))

    return _build(feats)


# ============================================================
# Node Classes
# ============================================================
class Node(ABC):
    def __init__(self, children=None):
        self.children = children if children else []
        # Scope = set of feature (channel) indices this node is a density over.
        # Set by every concrete subclass and used to enforce the properties.
        self.scope: FrozenSet[int] = frozenset()

    @abstractmethod
    def evaluate(self, x, cache=None):
        pass

    def _memoize(self, x, cache, compute):
        """Evaluate `compute` once per node instance per forward pass.

        Vtree circuits are DAGs (sub-circuits are shared across many parents),
        so a naive recursion recomputes shared nodes exponentially often. The
        cache is keyed by node identity for one forward pass.
        """
        if cache is None:
            cache = {}
        key = id(self)
        if key in cache:
            return cache[key]
        out = compute(x, cache)
        cache[key] = out
        return out


# ======================================================
# Input Node (leaf): a NORMALIZED univariate mixture density
# ======================================================
class InputNode(nn.Module, Node):
    """
    Leaf unit: a normalized mixture over {Gaussian, Laplace, Student-t} for a
    single feature (channel) index. All three components share location mu and
    scale sigma, so the leaf is a proper density that integrates to 1.

    Normalization note: the previous version multiplied the log-density by a
    learnable `gate`, which breaks Z = 1 and therefore exact marginals. The
    gate is removed from the math; `self.gate` is kept as a fixed (non-trainable)
    buffer = 1.0 purely so old checkpoints / diagnostics that reference it still
    load.
    """
    def __init__(self, feature_idx, mu_init=0.0, sigma_init=1.0, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children=[])

        self.feature_idx = feature_idx
        self.device = device
        self.scope = frozenset({feature_idx})

        # Shared location and scale
        self.mu = nn.Parameter(torch.tensor(float(mu_init), dtype=torch.float32, device=device))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init), dtype=torch.float32, device=device)))

        # Shape parameter for Student-t
        self.log_nu = nn.Parameter(torch.log(torch.tensor(5.0, dtype=torch.float32, device=device)))

        # Mixture weights over distributions: [Gaussian, Laplace, Student-t]
        self.logits = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))

        # Kept ONLY for back-compat (checkpoints / save_dict / plots). Fixed to
        # 1.0 and excluded from learnable params; it never scales the density.
        self.register_buffer("gate", torch.tensor(1.0, device=device))

        # Feature name
        self.feature_name = f"feat_{feature_idx}"

    def get_learnable_params(self):
        # NOTE: gate intentionally excluded (it must not scale the density).
        return [self.mu, self.log_sigma, self.log_nu, self.logits]

    def _log_gaussian(self, x, mu, sigma):
        return log_gaussian_jit(x, mu, sigma)

    def _log_laplace(self, x, mu, sigma):
        return log_laplace_jit(x, mu, sigma)

    def _log_student(self, x, mu, sigma, nu):
        return log_student_jit(x, mu, sigma, nu)

    def fit(self, X, jitter: float = 0.0):
        """
        Robust median/MAD initialization of (mu, sigma) for this channel.

        Args:
            X: (B, C, H, W) or (B, C) feature tensor / array.
            jitter: optional symmetry-breaking perturbation of mu by
                jitter * sigma * N(0,1). Sibling leaves of a mixture start
                identical otherwise and gradient symmetry keeps them identical
                forever (the mixture silently collapses to a single component).
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        vals = X[:, self.feature_idx].reshape(-1)

        mu_init = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu_init)) + 1e-6)
        sigma_init = mad * 1.4826  # consistent with Gaussian std
        nu_init = 5.0

        with torch.no_grad():
            self.mu.copy_(torch.tensor(mu_init, dtype=torch.float32, device=self.device))
            self.log_sigma.copy_(torch.log(torch.tensor(sigma_init, dtype=torch.float32, device=self.device)))
            self.log_nu.copy_(torch.log(torch.tensor(nu_init, dtype=torch.float32, device=self.device)))
            self.logits.copy_(torch.zeros(3, dtype=torch.float32, device=self.device))
            if jitter > 0:
                self.mu.add_(torch.randn((), device=self.device) * jitter * float(sigma_init))

    def evaluate(self, X, cache=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        return self._memoize(X, cache, self._compute)

    def _compute(self, X, cache):
        vals = X[:, self.feature_idx]
        sigma = F.softplus(self.log_sigma) + 1e-6
        nu = F.softplus(self.log_nu) + 1e-3

        log_gauss = self._log_gaussian(vals, self.mu, sigma)
        log_lapl = self._log_laplace(vals, self.mu, sigma)
        log_stud = self._log_student(vals, self.mu, sigma, nu)

        logpdfs = torch.stack([log_gauss, log_lapl, log_stud], dim=0)
        # log_softmax weights -> the mixture is normalized (a valid density).
        log_w = torch.log_softmax(self.logits, dim=0).view(-1, *([1] * (logpdfs.ndim - 1)))
        log_mix = torch.logsumexp(log_w + logpdfs, dim=0)
        return log_mix  # NO gate scaling: keeps Z = 1.


# ======================================================
# Sum Node: a mixture. SMOOTHNESS (completeness) is enforced here.
# ======================================================
class SumNode(Node, nn.Module):
    def __init__(self, children, weights=None, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children)
        if len(children) == 0:
            raise ValueError("SumNode requires at least one child.")

        # --- Smoothness: all children must share exactly one scope ---
        scopes = {_scope_of(c) for c in children}
        scopes.discard(frozenset())  # tolerate scope-less wrappers
        if len(scopes) > 1:
            raise ValueError(
                f"SumNode violates SMOOTHNESS: children have differing scopes {scopes}."
            )
        self.scope = next(iter(scopes)) if scopes else frozenset()

        n = len(children)
        if weights is None:
            weights = torch.ones(n, device=device) / n
        elif not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32, device=device)
        log_w = torch.log(weights / weights.sum())
        self.weights = nn.Parameter(log_w, requires_grad=True)
        self.device = device

    def get_learnable_params(self):
        return [self.weights]

    def evaluate(self, x, cache=None):
        return self._memoize(x, cache, self._compute)

    def _compute(self, x, cache):
        """log( sum_i w_i * p_i ) = logsumexp(log w_i + log p_i)."""
        child_logs = torch.stack([c.evaluate(x, cache) for c in self.children], dim=0)
        log_w = torch.log_softmax(self.weights, dim=0).view(-1, *([1] * (child_logs.ndim - 1)))
        return torch.logsumexp(log_w + child_logs, dim=0)


# ======================================================
# Product Node: a factorization. DECOMPOSABILITY is enforced here.
# ======================================================
class ProductNode(Node, nn.Module):
    def __init__(self, children):
        nn.Module.__init__(self)
        Node.__init__(self, children)
        if len(children) == 0:
            raise ValueError("ProductNode requires at least one child.")

        # --- Decomposability: child scopes must be pairwise disjoint ---
        union: set = set()
        for c in children:
            sc = _scope_of(c)
            overlap = union & sc
            if overlap:
                raise ValueError(
                    f"ProductNode violates DECOMPOSABILITY: overlapping child scopes {overlap}."
                )
            union |= sc
        self.scope = frozenset(union)

    def evaluate(self, x, cache=None):
        return self._memoize(x, cache, self._compute)

    def _compute(self, x, cache):
        """log( prod_i p_i ) = sum_i log p_i  (over disjoint scopes)."""
        child_logs = torch.stack([c.evaluate(x, cache) for c in self.children], dim=0)
        return torch.sum(child_logs, dim=0)


# ======================================================
# Gate / Residual: binary SUM nodes (kept for grammar back-compat).
# They are convex log-mixtures of two children, so they are valid SUM nodes
# and must obey smoothness: both children share the same scope.
# ======================================================
class GateNode(Node, nn.Module):
    def __init__(self, a, b, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, [a, b])
        sa, sb = _scope_of(a), _scope_of(b)
        if sa and sb and sa != sb:
            raise ValueError(
                f"GateNode violates SMOOTHNESS: children scopes {sa} != {sb}."
            )
        self.scope = sa or sb
        self.alpha = nn.Parameter(torch.tensor(0.0, device=device))
        self.device = device

    def get_learnable_params(self):
        return [self.alpha]

    def evaluate(self, x, cache=None):
        return self._memoize(x, cache, self._compute)

    def _compute(self, x, cache):
        gate = torch.sigmoid(self.alpha)
        left, right = self.children
        llog = left.evaluate(x, cache)
        rlog = right.evaluate(x, cache)
        return torch.logaddexp(llog + torch.log(gate + 1e-8),
                               rlog + torch.log(1 - gate + 1e-8))


class ResidualNode(Node, nn.Module):
    def __init__(self, base, sub, device="cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, [base, sub])
        sa, sb = _scope_of(base), _scope_of(sub)
        if sa and sb and sa != sb:
            raise ValueError(
                f"ResidualNode violates SMOOTHNESS: children scopes {sa} != {sb}."
            )
        self.scope = sa or sb
        self.beta = nn.Parameter(torch.tensor(0.5, device=device))
        self.device = device

    def get_learnable_params(self):
        return [self.beta]

    def evaluate(self, x, cache=None):
        return self._memoize(x, cache, self._compute)

    def _compute(self, x, cache):
        b = self.children[0].evaluate(x, cache)
        s = self.children[1].evaluate(x, cache)
        mix = torch.sigmoid(self.beta)
        return torch.logsumexp(torch.stack([
            b + torch.log(1 - mix + 1e-8),
            s + torch.log(mix + 1e-8)
        ]), dim=0)


# ======================================================
# Classifier Node: discriminative head over per-class circuits.
# ------------------------------------------------------
# This is NOT a generative circuit node: it stacks the (valid, full-scope)
# per-class density circuits into class logits = class-conditional
# log-likelihoods, used for generative classification / segmentation
# (argmax over classes). Each child remains an exact, normalized PC; the
# four properties are asserted on the children, not on the stack.
# ======================================================
class ClassifierNode(Node, nn.Module):
    def __init__(self, children):
        nn.Module.__init__(self)
        Node.__init__(self, children)
        scopes = {_scope_of(c) for c in children}
        scopes.discard(frozenset())
        # All class circuits should be densities over the SAME full scope.
        self.scope = next(iter(scopes)) if len(scopes) == 1 else frozenset().union(*scopes) if scopes else frozenset()

    def evaluate(self, x, cache=None):
        """Return (B, n_classes, H, W) [or (B, n_classes)] log-probs."""
        if cache is None:
            cache = {}
        child_logs = [c.evaluate(x, cache) for c in self.children]
        return torch.stack(child_logs, dim=1)


# ============================================================
# Graph Classes
# ============================================================
class AbstractGraph:
    def __init__(self):
        self.nodes, self.edges = set(), {}

    def add_node(self, node):
        self.nodes.add(node)
        self.edges.setdefault(node, set())

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must be added before edge")
        self.edges[parent].add(child)


class DirectedAcyclicGraph(AbstractGraph):
    def add_edge(self, parent, child):
        super().add_edge(parent, child)
        if self._has_cycle():
            self.edges[parent].remove(child)
            raise ValueError("Adding this edge creates a cycle")

    def _has_cycle(self):
        visited, stack = set(), set()

        def visit(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            if any(visit(c) for c in self.edges.get(node, [])):
                return True
            stack.remove(node)
            return False

        return any(visit(n) for n in self.nodes)


# ============================================================
# The four property validators
# ============================================================
def _compute_scope(node, cache=None, check_smooth=True, check_decomp=True) -> FrozenSet[int]:
    """Recursively compute a node's scope, optionally asserting smoothness and
    decomposability. Memoized by id so DAGs are traversed once per node."""
    if cache is None:
        cache = {}
    nid = id(node)
    if nid in cache:
        return cache[nid]

    if isinstance(node, InputNode):
        scope = frozenset({node.feature_idx})

    elif isinstance(node, ProductNode):
        child_scopes = [_compute_scope(c, cache, check_smooth, check_decomp) for c in node.children]
        combined: FrozenSet[int] = frozenset()
        for i, si in enumerate(child_scopes):
            if check_decomp:
                for j in range(i + 1, len(child_scopes)):
                    sj = child_scopes[j]
                    assert si.isdisjoint(sj), (
                        f"Decomposability violated: ProductNode children {i},{j} "
                        f"share feature(s) {set(si & sj)}"
                    )
            combined = combined | si
        scope = combined

    elif isinstance(node, (SumNode, GateNode, ResidualNode, ClassifierNode)):
        child_scopes = [_compute_scope(c, cache, check_smooth, check_decomp) for c in node.children]
        if child_scopes:
            ref = child_scopes[0]
            # ClassifierNode is discriminative: do not require smoothness on it.
            if check_smooth and not isinstance(node, ClassifierNode):
                for i, cs in enumerate(child_scopes[1:], 1):
                    assert cs == ref, (
                        f"Smoothness violated: {type(node).__name__} child 0 has scope "
                        f"{set(ref)} but child {i} has scope {set(cs)}"
                    )
            scope = frozenset().union(*child_scopes)
        else:
            scope = frozenset()
    else:
        # Unknown / wrapper node (e.g. SubPCNet): fall back to its declared scope.
        scope = _scope_of(node)

    cache[nid] = scope
    return scope


def validate_smoothness(root) -> None:
    """Property 1: every sum node's children share the same scope."""
    _compute_scope(root, check_smooth=True, check_decomp=False)


def validate_decomposability(root) -> None:
    """Property 2: every product node's children have disjoint scopes."""
    _compute_scope(root, check_smooth=False, check_decomp=True)


def validate_circuit(root) -> None:
    """Verify smoothness AND decomposability (the two mandatory properties for
    exact density + exact marginals). Raises AssertionError on first violation."""
    _compute_scope(root, check_smooth=True, check_decomp=True)


def validate_determinism(root, x, log_zero_threshold: float = -1e8) -> None:
    """Property 3 (empirical): for every input at most one child of each sum is
    non-zero. Mixtures of full-support leaves are NOT deterministic and will
    fail here — that is the correct, expected outcome (determinism is only
    needed for exact MAP/MPE, which this project does not use)."""
    def _walk(node, cache):
        if id(node) in cache:
            return
        cache.add(id(node))
        if isinstance(node, SumNode):
            outs = torch.stack([c.evaluate(x) for c in node.children], dim=0)
            nonzero = (outs > log_zero_threshold).reshape(len(node.children), -1).sum(dim=0)
            assert (nonzero <= 1).all(), (
                f"Determinism violated: a SumNode has up to {int(nonzero.max())} "
                f"non-zero children for some input."
            )
        for c in getattr(node, "children", []):
            _walk(c, cache)
    _walk(root, set())


def validate_structured_decomposability(root, vtree: VtreeNode) -> None:
    """Property 4: every product node splits its scope exactly as `vtree`
    prescribes. Requires binary product nodes (which the vtree builder makes)."""
    allowed = {
        frozenset({n.left_scope, n.right_scope})
        for n in vtree_nodes(vtree)
        if isinstance(n, VtreeInternal)
    }
    cache: set = set()

    def _check(node):
        if id(node) in cache:
            return
        cache.add(id(node))
        if isinstance(node, ProductNode):
            assert len(node.children) == 2, (
                "Structured decomposability requires binary product nodes; found "
                f"one with {len(node.children)} children."
            )
            split = frozenset({_scope_of(node.children[0]), _scope_of(node.children[1])})
            assert split in allowed, (
                f"Structured decomposability violated: product split {split} is not "
                f"prescribed by the vtree."
            )
        for c in getattr(node, "children", []):
            _check(c)

    _check(root)


# ============================================================
# Probabilistic Circuit
# ============================================================
class ProbabilisticCircuit(nn.Module):
    def __init__(self, n_classes=2, device="cpu"):
        nn.Module.__init__(self)
        self.graph = DirectedAcyclicGraph()
        self.root, self.n_classes, self.device = None, n_classes, device

    def register_circuit_modules(self):
        """Expose the circuit's learnable parameters through the native
        nn.Module machinery (.parameters(), .to(), .state_dict()).

        We register the parameters in a flat ParameterList rather than the
        nodes themselves: the nodes are both nn.Module AND Node, and Node uses a
        ``children`` LIST attribute that shadows nn.Module.children() — so the
        native ._apply() recursion used by .to() breaks if a node is a
        registered sub-module. The ParameterList holds the SAME Parameter
        objects the nodes use, so moving them with .to() moves them for the
        nodes too. Call once the graph is fully built (end of init_network)."""
        seen, flat = set(), []
        for p in self.params:
            if id(p) not in seen:
                seen.add(id(p))
                flat.append(p)
        self._circuit_params = nn.ParameterList(flat)

    def set_root(self, node):
        if node not in self.graph.nodes:
            raise ValueError("Root must be in the graph")
        # Bypass nn.Module.__setattr__ so `root` is a plain attribute, not a
        # registered sub-module (see register_circuit_modules for why).
        object.__setattr__(self, "root", node)

    def add_node(self, node):
        self.graph.add_node(node)

    def add_edge(self, parent, child):
        self.graph.add_edge(parent, child)
        if child not in parent.children:
            parent.children.append(child)

    def evaluate(self, x):
        return self.root.evaluate(x) if self.root else None

    def get_nodes(self):
        return self.graph.nodes

    def get_edges(self):
        return [(p, c) for p, cs in self.graph.edges.items() for c in cs]

    def validate(self):
        """Assert the structural properties of the (generative) sub-circuits.

        ClassifierNode is a discriminative head, so smoothness/decomposability
        are checked on each of its children (the per-class density circuits),
        and structured decomposability is checked against self.vtree when set.
        """
        if self.root is None:
            raise RuntimeError("Circuit not initialized; call init_network first.")
        children = self.root.children if isinstance(self.root, ClassifierNode) else [self.root]
        for c in children:
            validate_circuit(c)
            if getattr(self, "vtree", None) is not None:
                validate_structured_decomposability(c, self.vtree)
        print_info("PC validation passed: smoothness + decomposability"
                   + (" + structured decomposability" if getattr(self, "vtree", None) else ""))


    def visualize(self, direction="LR", layer_sep=2.0, node_sep=5.0, save_path=None):
        """
        Visualize the probabilistic circuit as a tree-like hierarchical layout with proper spacing.

        :param direction: 'TB' (top-to-bottom) or 'LR' (left-to-right)
        :param layer_sep: spacing between layers (higher = more vertical spacing)
        :param node_sep: spacing between nodes in the same layer (higher = more horizontal spacing)
        """
        G = nx.DiGraph()
        colors = {}

        # Add nodes with labels and colors
        for node in self.get_nodes():
            if isinstance(node, InputNode):
                label = "In"
                label += f"\n{node.feature_name}"
                colors[node] = "lightgreen"
            elif isinstance(node, SumNode):
                label = "Sum"
                weights_str = ','.join([f"{w:.2f}\n" for w in node.weights.detach().cpu().numpy()])
                label += f"\n(w={weights_str})"
                colors[node] = "skyblue"
            elif isinstance(node, ProductNode):
                label = "Product"
                colors[node] = "lightcoral"
            elif isinstance(node, ClassifierNode):
                label = "Classifier"
                colors[node] = "orange"
            else:
                label = node.__class__.__name__
                colors[node] = "lightgray"
            G.add_node(node, label=label)

        # Add edges
        for parent, child in self.get_edges():
            G.add_edge(parent, child)

        for n in G.nodes:
            G.nodes[n]['rankdir'] = 'TB' if direction == "TB" else 'LR'
        G.graph['graph'] = {'ranksep': str(layer_sep), 'nodesep': str(node_sep)}

        # Hierarchical layout
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        if direction == "LR":
            pos = {k: (y, -x) for k, (x, y) in pos.items()}

        labels = nx.get_node_attributes(G, 'label')
        node_colors = [colors[n] for n in G.nodes]

        plt.figure(figsize=(16, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_size=4000,
            node_color=node_colors,
            font_size=7,
            font_color="black",
            edgecolors="k",
            linewidths=0.7,
            arrows=True,
            arrowsize=12
        )
        plt.title("Probabilistic Circuit Tree Visualization", fontsize=16)
        if save_path is not None:
            save_path = os.path.join(save_path, "pc_tree.png")
            plt.savefig(save_path, bbox_inches='tight')
            print_info(f"PC visualization saved to {save_path}")


# ============================================================
# PCNet — a smooth + decomposable + structured-decomposable circuit
# built top-down from a single shared vtree over the C feature channels.
# ============================================================
class PCNet(ProbabilisticCircuit):
    """
    Per-class probabilistic-circuit segmenter.

    For each of the `n_classes` classes we build ONE exact density circuit over
    all C channel features, using a region graph induced by a single shared
    vtree. A ClassifierNode stacks the per-class log-densities into
    (B, n_classes, H, W) class logits; argmax over classes gives the
    segmentation. Because every class circuit is built from the same vtree:

        * SumNodes mix products of identical scope        -> smoothness
        * ProductNodes combine disjoint left/right scopes -> decomposability
        * all products follow the shared vtree split      -> structured decomp.
        * leaves are normalized mixtures, weights softmax -> Z = 1

    Back-compat hyperparameters:
        max_branching : sets region width if the *_components args are None.
        max_depth     : retained for the public signature; it does NOT cap
                        recursion (capping would drop variables and break
                        coverage). Depth follows the data dimensionality.
    """
    def __init__(
        self, input_size=(256, 256), n_classes=2, distribution=None, device="cpu",
        max_depth=4, max_branching=3, seed=42, cv_module=None,
        n_sum_components=None, n_leaf_components=None, n_repetitions=1,
        leaf_jitter=0.05,
    ):
        super().__init__(n_classes, device)
        self.input_size = input_size
        self.distribution = distribution
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.seed = seed
        self.cv_module = cv_module
        self.n_sum_components = n_sum_components if n_sum_components is not None else max_branching
        self.n_leaf_components = n_leaf_components if n_leaf_components is not None else max_branching
        self.n_repetitions = n_repetitions
        self.leaf_jitter = leaf_jitter
        self.n_features = None
        self.vtree = None
        random.seed(self.seed)

    # --------------------------------------------------------
    # Structure construction (vtree region graph)
    # --------------------------------------------------------
    def _build_region(self, vnode: VtreeNode, inputs, rng):
        """Return a list of circuit nodes all with scope == vnode.scope.

        Leaf region  -> n_leaf_components InputNodes for the single feature.
        Internal     -> products of (left x right) sub-nodes (disjoint scopes),
                        mixed by n_sum_components sum nodes (identical scope).
        Nodes/edges are registered in the graph as they are built.
        """
        if isinstance(vnode, VtreeLeaf):
            leaves = []
            for _ in range(self.n_leaf_components):
                leaf = InputNode(vnode.feature_idx, device=self.device)
                leaf.fit(inputs, jitter=self.leaf_jitter)
                self.add_node(leaf)
                leaves.append(leaf)
            return leaves

        left_nodes = self._build_region(vnode.left, inputs, rng)
        right_nodes = self._build_region(vnode.right, inputs, rng)

        # Products: one node per (left, right) pair -> disjoint scopes.
        products = []
        for l in left_nodes:
            for r in right_nodes:
                prod = ProductNode([l, r])
                self.add_node(prod)
                self.add_edge(prod, l)
                self.add_edge(prod, r)
                products.append(prod)

        # Sums: each mixes the same products -> identical scope (smooth).
        sums = []
        for _ in range(self.n_sum_components):
            s = SumNode(list(products), device=self.device)
            self.add_node(s)
            for p in products:
                self.add_edge(s, p)
            sums.append(s)
        return sums

    def _build_class_circuit(self, inputs, rng):
        """One full-scope density circuit (a single class), possibly a mixture
        of `n_repetitions` independent region graphs over the shared vtree."""
        rep_roots = []
        for _ in range(self.n_repetitions):
            region = self._build_region(self.vtree, inputs, rng)
            if len(region) == 1:
                rep_roots.append(region[0])
            else:
                s = SumNode(list(region), device=self.device)
                self.add_node(s)
                for c in region:
                    self.add_edge(s, c)
                rep_roots.append(s)
        if len(rep_roots) == 1:
            return rep_roots[0]
        root = SumNode(list(rep_roots), device=self.device)
        self.add_node(root)
        for c in rep_roots:
            self.add_edge(root, c)
        return root

    def build_network(self, inputs):
        """Build n_classes independent full-scope circuits + a ClassifierNode."""
        rng = random.Random(self.seed)
        class_roots = [self._build_class_circuit(inputs, rng) for _ in range(self.n_classes)]
        root = ClassifierNode(class_roots)
        self.add_node(root)
        for c in class_roots:
            self.add_edge(root, c)
        return root

    def init_network(self, inputs, labels=None):
        """Initialize the structured-decomposable PCNet from a feature batch.

        Args:
            inputs: (B, C, H, W) feature tensor — C channels are the features.
            labels: unused (kept for API symmetry; training is per-class
                    generative, supervised by the loss / metrics downstream).
        """
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        print_configs("Inputs tensor shape:", tuple(inputs.shape))
        C = inputs.shape[1]
        self.n_features = C

        # One shared vtree over all C features -> structured decomposability.
        self.vtree = random_balanced_vtree(list(range(C)), seed=self.seed)

        root = self.build_network(inputs)
        self.set_root(root)

        # Coverage: each class circuit must be a density over ALL features.
        full = frozenset(range(C))
        for c in root.children:
            assert _scope_of(c) == full, (
                f"Class circuit scope {sorted(_scope_of(c))} != all features 0..{C-1}"
            )

        # Collect learnable params (leaves + sums + any gate/residual).
        self.params = []
        for n in self.get_nodes():
            if hasattr(n, "get_learnable_params"):
                self.params += n.get_learnable_params()

        # Fail loudly on any property violation.
        self.validate()

        # Register node modules so .parameters()/.to()/.train() work natively.
        self.register_circuit_modules()

        print_configs(f"Initialized PCNet with {len(self.get_nodes())} nodes, "
                      f"{sum(p.numel() for p in self.params)} trainable parameters")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    def forward(self, x, **kwargs):
        """nn.Module entry point. The experiment pipeline calls model(x); we
        delegate to evaluate (kwargs are accepted for call-site symmetry)."""
        return self.evaluate(x)

    def evaluate(self, x):
        if self.cv_module is not None:
            x = [self.cv_module.get_output(x_, return_map=True) for x_ in x]
            x = np.stack(x, axis=0)
            x = torch.from_numpy(x).float().to(self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return self.root.evaluate(x)

    def predict(self, inputs):
        with torch.no_grad():
            logits = self.evaluate(inputs)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            return preds.cpu().numpy()

    def log_prob(self, x, class_idx=0):
        """Exact log-density of the per-class circuit (the quantity the marginal
        machinery of the outlier paper operates on)."""
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return self.root.children[class_idx].evaluate(x)

    # --------------------------------------------------------
    # Serialization / introspection
    # --------------------------------------------------------
    def state_dict(self):
        """Return a unified state_dict collecting parameters from all node modules."""
        state = {}
        for node in self.get_nodes():
            if isinstance(node, nn.Module):
                for name, param in node.named_parameters():
                    key = f"{node.__class__.__name__}_{id(node)}.{name}"
                    state[key] = param.detach().clone()
        return state

    def load_state_dict(self, state_dict, strict=False):
        """Load parameters into the PCNet nodes (tolerant of structure drift)."""
        loaded, missing = 0, 0
        for node in self.get_nodes():
            if isinstance(node, nn.Module):
                for name, param in node.named_parameters():
                    key = f"{node.__class__.__name__}_{id(node)}.{name}"
                    if key in state_dict:
                        try:
                            with torch.no_grad():
                                src = state_dict[key].to(param.device)
                                if src.shape == param.shape:
                                    param.copy_(src)
                                else:
                                    min_len = min(param.numel(), src.numel())
                                    param.view(-1)[:min_len].copy_(src.view(-1)[:min_len])
                                    print_debugging(f"⚠️ Resized param {key}: old {tuple(src.shape)} → new {tuple(param.shape)}")
                            loaded += 1
                        except Exception:
                            missing += 1
                    elif not strict:
                        for k, v in state_dict.items():
                            if k.endswith(f".{name}"):
                                with torch.no_grad():
                                    param.copy_(v.to(param.device))
                                loaded += 1
                                break
                        else:
                            missing += 1
        print_debugging(f"Loaded {loaded} parameters, {missing} missing.")
        return {"loaded": loaded, "missing": missing}

    def save_dict(self):
        """Return a compact summary dictionary of the PCNet model."""
        info = {}
        info["model_type"] = self.__class__.__name__
        info["device"] = str(self.device)
        info["n_classes"] = self.n_classes
        info["input_size"] = tuple(self.input_size)
        info["max_depth"] = self.max_depth
        info["max_branching"] = self.max_branching
        info["seed"] = self.seed

        nodes = list(self.get_nodes())
        edges = list(self.get_edges())
        info["n_nodes"] = len(nodes)
        info["n_edges"] = len(edges)

        type_counts = Counter([n.__class__.__name__ for n in nodes])
        info["node_types"] = dict(type_counts)

        mus, sigmas, nus, gates = [], [], [], []
        sum_weights = []
        input_feats = []

        for node in nodes:
            if isinstance(node, InputNode):
                input_feats.append(node.feature_name)
                mus.append(node.mu.item())
                sigmas.append(torch.exp(node.log_sigma).item())
                nus.append(torch.exp(node.log_nu).item())
                gates.append(float(node.gate.item()))
            elif isinstance(node, SumNode):
                w = torch.softmax(node.weights, dim=0).detach().cpu().numpy()
                sum_weights.extend(w.tolist())

        def safe_stats(x):
            if len(x) == 0:
                return {"mean": None, "std": None, "min": None, "max": None}
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }

        info["parameter_summary"] = {
            "mu": safe_stats(mus),
            "sigma": safe_stats(sigmas),
            "nu": safe_stats(nus),
            "sum_weights": safe_stats(sum_weights),
        }

        info["n_parameters"] = sum(p.numel() for p in getattr(self, "params", []))
        info["input_features"] = input_feats
        model_memory_kb = sum(p.numel() * p.element_size() for p in getattr(self, "params", [])) / 1024
        info["memory_kb"] = model_memory_kb
        return info

    def __repr__(self):
        if not hasattr(self, "graph") or not self.root:
            return "<Uninitialized PCNet>"

        nodes = list(self.get_nodes())
        counts = {
            "InputNode": sum(isinstance(n, InputNode) for n in nodes),
            "SumNode": sum(isinstance(n, SumNode) for n in nodes),
            "ProductNode": sum(isinstance(n, ProductNode) for n in nodes),
            "ClassifierNode": sum(isinstance(n, ClassifierNode) for n in nodes),
        }

        total_params = 0
        param_summary = {}
        for n in nodes:
            if isinstance(n, InputNode):
                p = sum(pp.numel() for pp in [n.mu, n.log_sigma, n.log_nu, n.logits])
                param_summary["InputNode"] = param_summary.get("InputNode", 0) + p
                total_params += p
            elif isinstance(n, SumNode):
                p = n.weights.numel()
                param_summary["SumNode"] = param_summary.get("SumNode", 0) + p
                total_params += p

        header = [
            "─────────────────────────────────────────────",
            "🧠  Probabilistic Circuit Network Summary",
            "─────────────────────────────────────────────",
            f"📦 Classes: {self.n_classes}",
            f"📏 Input size: {self.input_size}",
            f"⚙️  Depth: {self.max_depth}, Branching: {self.max_branching}",
            f"💻 Device: {self.device}",
            "─────────────────────────────────────────────",
            "📊 Node counts:",
        ]
        for k, v in counts.items():
            header.append(f"  • {k:<15}: {v}")
        header.append("─────────────────────────────────────────────")
        header.append(f"🧩 Trainable parameters: {total_params}")
        for k, v in param_summary.items():
            header.append(f"  • {k:<15}: {v}")
        header.append("─────────────────────────────────────────────")
        header.append("📚 Structure:")
        header_text = "\n".join(header)
        model_memory_kb = sum(p.numel() * p.element_size() for p in getattr(self, "params", [])) / 1024
        header_text += f"\nMemory usage: {model_memory_kb:.2f} KB\n"

        lines = []

        def format_node(node, prefix="", is_last=True):
            connector = "└─ " if is_last else "├─ "
            if isinstance(node, ClassifierNode):
                label = f"ClassifierNode(classes={len(node.children)})"
            elif isinstance(node, SumNode):
                label = f"SumNode(children={len(node.children)}, weights={len(node.weights)})"
            elif isinstance(node, ProductNode):
                label = f"ProductNode(children={len(node.children)})"
            elif isinstance(node, InputNode):
                label = f"InputNode({node.feature_name})"
            else:
                label = node.__class__.__name__
            lines.append(f"{prefix}{connector}{label}")
            if hasattr(node, "children") and node.children:
                new_prefix = prefix + ("   " if is_last else "│  ")
                for i, child in enumerate(node.children):
                    format_node(child, new_prefix, i == len(node.children) - 1)

        format_node(self.root)
        tree_text = "\n".join(lines)
        return f"{header_text}\n{tree_text}\n─────────────────────────────────────────────"

    def info(self, show_layers=True, show_params=True, show_counts=True):
        print("─────────────────────────────────────────────")
        print("🧠  Probabilistic Circuit Network Summary")
        print("─────────────────────────────────────────────")
        print(f"📦 Classes: {self.n_classes}")
        print(f"📏 Input size: {self.input_size}")
        print(f"⚙️  Depth: {self.max_depth}, Branching: {self.max_branching}")
        print(f"💻 Device: {self.device}")
        print("─────────────────────────────────────────────")

        nodes = list(self.get_nodes())
        total_params = 0

        if show_counts:
            counts = {
                "InputNode": sum(isinstance(n, InputNode) for n in nodes),
                "SumNode": sum(isinstance(n, SumNode) for n in nodes),
                "ProductNode": sum(isinstance(n, ProductNode) for n in nodes),
                "ClassifierNode": sum(isinstance(n, ClassifierNode) for n in nodes),
            }
            print("📊 Node counts:")
            for k, v in counts.items():
                print(f"  • {k:<15}: {v}")
            print("─────────────────────────────────────────────")

        if show_params:
            param_summary = []
            for n in nodes:
                if isinstance(n, InputNode):
                    p = sum(pp.numel() for pp in [n.mu, n.log_sigma, n.log_nu, n.logits])
                    param_summary.append(("InputNode", p))
                elif isinstance(n, SumNode):
                    p = n.weights.numel()
                    param_summary.append(("SumNode", p))
            total_params = sum(p for _, p in param_summary)
            print(f"🧩 Trainable parameters: {total_params}")
            by_type = {}
            for t, p in param_summary:
                by_type[t] = by_type.get(t, 0) + p
            for t, p in by_type.items():
                print(f"  • {t:<15}: {p}")
            print("─────────────────────────────────────────────")
        model_memory_kb = sum(p.numel() * p.element_size() for p in getattr(self, "params", [])) / 1024
        print(f"📚 Memory usage: {model_memory_kb:.2f} KB")
        print("─────────────────────────────────────────────")

        return {
            "n_nodes": len(nodes),
            "n_params": total_params,
            "depth": self.max_depth,
            "branching": self.max_branching,
        }


# ============================================================
# MultiPCNet: multiple sub-circuits (experts) + learned selector
# ------------------------------------------------------------
# Each expert is a full PCNet (a valid PC). The soft selector is a smooth
# mixture over the experts (all share the full scope), so the mixture is
# itself a valid PC. The hard/gated selectors are for evaluation only.
# ============================================================
from typing import List, Literal, Optional, Tuple


# ================================
# Selector nodes
# ================================
class SelectorNode(nn.Module, Node):
    """Soft mixture selector (a smooth SumNode) over sub-circuits (experts).
    Each child returns (B, C, H, W) log-probabilities (ClassifierNode output)."""
    def __init__(self, children: List[Node], device: str = "cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children=children)
        self.device = device
        self.logits = nn.Parameter(torch.zeros(len(children), device=device))
        # Experts share the full feature scope; record it for the validators.
        scopes = {_scope_of(c) for c in children}
        scopes.discard(frozenset())
        self.scope = next(iter(scopes)) if len(scopes) == 1 else frozenset()

    def get_learnable_params(self):
        return [self.logits]

    @property
    def n_experts(self):
        return len(self.children)

    def evaluate(self, x, cache=None):
        expert_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)  # (K,B,C,H,W)
        log_w = torch.log_softmax(self.logits, dim=0).view(-1, 1, 1, 1, 1)
        return torch.logsumexp(log_w + expert_logs, dim=0)


class HardBestSelectorNode(Node):
    """Non-differentiable selector: picks the expert with the highest average
    log-likelihood for the current batch. Use for evaluation/inference only."""
    def __init__(self, children: List[Node]):
        super().__init__(children=children)

    def evaluate(self, x, cache=None):
        expert_logs = [c.evaluate(x) for c in self.children]
        stacked = torch.stack(expert_logs, dim=0)             # (K,B,C,H,W)
        scores = stacked.mean(dim=(2, 3, 4))                  # (K,B)
        best_k = scores.argmax(dim=0)                         # (B,)
        B = stacked.size(1)
        out = [stacked[best_k[b], b] for b in range(B)]
        return torch.stack(out, dim=0)


class InputGatedSelectorNode(nn.Module, Node):
    """Input-dependent gating (tiny MLP over a global descriptor of the input).
    NOTE: input-conditional weights make the mixture a *conditional* circuit;
    the per-expert densities remain exact, but the overall object is no longer
    a single normalized joint. Use for discriminative gating only."""
    def __init__(self, children: List[Node], feat_dim: int, hidden: int = 64, device: str = "cpu"):
        nn.Module.__init__(self)
        Node.__init__(self, children=children)
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, len(children))
        ).to(device)

    def get_learnable_params(self):
        return list(self.mlp.parameters())

    def evaluate(self, x, cache=None):
        raise NotImplementedError("Use evaluate_with_features(x, x_feat) for the gated selector.")

    def evaluate_with_features(self, x, x_feat: torch.Tensor):
        expert_logs = torch.stack([c.evaluate(x) for c in self.children], dim=0)  # (K,B,C,H,W)
        B = expert_logs.size(1)
        logits = self.mlp(x_feat)                         # (B, K)
        log_w = torch.log_softmax(logits, dim=1).transpose(0, 1).view(-1, B, 1, 1, 1)
        return torch.logsumexp(log_w + expert_logs, dim=0)


# ================================
# SubPCNet: thin wrapper around PCNet
# ================================
class SubPCNet:
    """Minimal wrapper to reuse PCNet as an expert. Each sub-net owns its own
    graph, parameters, etc."""
    def __init__(self, base_pcnet_cls, *,
                 input_size=(256, 256), n_classes=2, device="cpu",
                 max_depth=3, max_branching=3, seed=42, cv_module=None,
                 name: Optional[str] = None):
        self.model = base_pcnet_cls(
            input_size=input_size, n_classes=n_classes, device=device,
            max_depth=max_depth, max_branching=max_branching, seed=seed,
            cv_module=cv_module
        )
        self.name = name or f"SubPCNet_{id(self)}"
        self.device = device
        self.n_classes = n_classes

    @property
    def scope(self):
        return _scope_of(self.model.root) if getattr(self.model, "root", None) else frozenset()

    def evaluate(self, x, cache=None):
        return self.model.evaluate(x)  # (B,C,H,W) in log-space

    def init_network(self, inputs, labels):
        self.model.init_network(inputs, labels)

    def parameters(self):
        if hasattr(self.model, "params") and len(getattr(self.model, "params", [])):
            for p in self.model.params:
                yield p
        else:
            for _, p in self.model.named_parameters():
                yield p

    def save_dict(self):
        info = self.model.save_dict()
        info["subnet_name"] = self.name
        return info


# ================================
# MultiPCNet: K experts + selector root
# ================================
class MultiPCNet:
    """Build K sub-PCNets (experts) and a selector on top.

    selector_mode:
        - "soft":  SelectorNode (smooth mixture; default, trainable)
        - "hard":  HardBestSelectorNode (non-differentiable; eval only)
        - "gated": InputGatedSelectorNode (conditional gating; needs features)
    """
    def __init__(self,
                 base_pcnet_cls,
                 *,
                 n_experts: int = 3,
                 input_size: Tuple[int, int] = (256, 256),
                 n_classes: int = 2,
                 device: str = "cpu",
                 max_depth: int = 3,
                 max_branching: int = 3,
                 cv_module=None,
                 selector_mode: Literal["soft", "hard", "gated"] = "soft",
                 x_feature_fn=None,
                 x_feature_dim: Optional[int] = None,
                 seed: int = 42,
                 bootstrap: bool = False,
                 name: str = "MultiPCNet"):
        self.name = name
        self.device = device
        self.n_classes = n_classes
        self.selector_mode = selector_mode
        self.x_feature_fn = x_feature_fn
        self.bootstrap = bootstrap

        self.experts: List[SubPCNet] = []
        for k in range(n_experts):
            self.experts.append(
                SubPCNet(
                    base_pcnet_cls,
                    input_size=input_size,
                    n_classes=n_classes,
                    device=device,
                    max_depth=max_depth,
                    max_branching=max_branching,
                    seed=seed + k,
                    cv_module=cv_module,
                    name=f"Expert_{k}"
                )
            )

        if selector_mode == "soft":
            self.selector = SelectorNode([e for e in self.experts], device=device)
        elif selector_mode == "hard":
            self.selector = HardBestSelectorNode([e for e in self.experts])
        elif selector_mode == "gated":
            assert x_feature_fn is not None and x_feature_dim is not None, "gated selector requires x_feature_fn and x_feature_dim"
            self.selector = InputGatedSelectorNode([e for e in self.experts], feat_dim=x_feature_dim, device=device)
        else:
            raise ValueError(f"Unknown selector_mode: {selector_mode}")

        self.params: List[torch.nn.Parameter] = []

    def init_network(self, inputs: torch.Tensor, labels: torch.Tensor):
        B = inputs.shape[0]
        for k, expert in enumerate(self.experts):
            if self.bootstrap and B >= 4:
                idx = torch.randint(0, B, (max(1, int(0.8 * B)),), device=inputs.device)
                sub_in = inputs[idx]
                sub_lb = labels[idx] if isinstance(labels, torch.Tensor) and labels.shape[0] == B else labels
            else:
                sub_in, sub_lb = inputs, labels
            expert.init_network(sub_in, sub_lb)

        self.params = []
        for e in self.experts:
            for p in e.parameters():
                self.params.append(p)
        if isinstance(self.selector, nn.Module):
            self.params += list(self.selector.get_learnable_params()) if hasattr(self.selector, 'get_learnable_params') else list(self.selector.parameters())

        try:
            n_params = sum(p.numel() for p in self.params if p.requires_grad)
            print_configs(f"Initialized MultiPCNet with {len(self.experts)} experts; trainable params: {n_params}")
        except Exception:
            pass

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        if self.selector_mode == "gated":
            with torch.no_grad():
                x_feat = self.x_feature_fn(x)
            return self.selector.evaluate_with_features(x, x_feat)
        else:
            return self.selector.evaluate(x)

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.evaluate(inputs)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            return preds.cpu().numpy()

    def named_parameters(self):
        for k, e in enumerate(self.experts):
            if hasattr(e.model, "params") and len(getattr(e.model, "params", [])):
                for i, p in enumerate(e.model.params):
                    yield f"experts.{k}.param_{i}", p
            else:
                for name, p in e.model.named_parameters():
                    yield f"experts.{k}.{name}", p
        if isinstance(self.selector, nn.Module):
            for name, p in self.selector.named_parameters():
                yield f"selector.{name}", p

    def state_dict(self):
        state = {}
        for k, e in enumerate(self.experts):
            sd = e.model.state_dict()
            for key, tensor in sd.items():
                state[f"experts.{k}.{key}"] = tensor
        if isinstance(self.selector, nn.Module):
            for name, p in self.selector.named_parameters():
                state[f"selector.{name}"] = p.detach().clone()
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = False):
        loaded, missing = 0, 0
        for k, e in enumerate(self.experts):
            prefix = f"experts.{k}."
            sub = {k2[len(prefix):]: v for k2, v in state_dict.items() if k2.startswith(prefix)}
            stats = e.model.load_state_dict(sub, strict=strict)
            loaded += stats.get("loaded", 0)
            missing += stats.get("missing", 0)
        if isinstance(self.selector, nn.Module):
            for name, p in self.selector.named_parameters():
                key = f"selector.{name}"
                if key in state_dict:
                    with torch.no_grad():
                        src = state_dict[key].to(p.device)
                        if src.shape == p.shape:
                            p.copy_(src)
                            loaded += 1
                        else:
                            n = min(src.numel(), p.numel())
                            p.view(-1)[:n].copy_(src.view(-1)[:n])
                            loaded += 1
                else:
                    missing += 1
        try:
            print_debugging(f"MultiPCNet load_state: loaded={loaded}, missing={missing}")
        except Exception:
            pass
        return {"loaded": loaded, "missing": missing}

    def save_dict(self):
        info = {
            "model_type": "MultiPCNet",
            "name": self.name,
            "device": self.device,
            "n_classes": self.n_classes,
            "n_experts": len(self.experts),
            "selector_mode": self.selector_mode,
        }
        info["experts"] = [e.save_dict() for e in self.experts]
        if isinstance(self.selector, SelectorNode):
            with torch.no_grad():
                w = torch.softmax(self.selector.logits, dim=0).detach().cpu().numpy().tolist()
            info["selector_weights"] = w
        return info


# ================================
# Convenience trainer (optional)
# ================================
class MultiPCNetTrainer:
    """Minimal training loop hook. Plug in your own loss & dataloaders."""
    def __init__(self, model: MultiPCNet, lr: float = 1e-3, weight_decay: float = 0.0):
        self.model = model
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        return F.nll_loss(F.log_softmax(logits, dim=1), targets)

    def step(self, x: torch.Tensor, y: torch.Tensor):
        self.opt.zero_grad()
        logits = self.model.evaluate(x)
        loss = self.loss_fn(logits, y)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def eval_batch(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.model.evaluate(x)
        if y.ndim == 4 and y.size(1) == 1:
            y_ = y.squeeze(1)
        else:
            y_ = y
        pred = logits.argmax(dim=1)
        acc = (pred == y_).float().mean().item()
        return acc
