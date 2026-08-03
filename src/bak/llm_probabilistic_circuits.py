import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ======================================================
# JIT-Compiled Math Kernels (Performance Critical)
# ======================================================
@torch.jit.script
def log_gaussian_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # 0.5 * log(2*pi) ~= 0.9189385
    return -0.5 * ((x - mu) / sigma)**2 - torch.log(sigma) - 0.9189385

@torch.jit.script
def log_laplace_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # b = sigma / sqrt(2). log(2b) = log(sigma*sqrt(2)) = log(sigma) + 0.34657
    b = sigma * 0.70710678
    return -torch.abs(x - mu) / b - (torch.log(sigma) + 0.3465736)

@torch.jit.script
def log_student_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    # Stable Student-T implementation
    z = (x - mu) / (sigma + 1e-12)
    term1 = torch.lgamma((nu + 1) * 0.5)
    term2 = torch.lgamma(nu * 0.5)
    term3 = 0.5 * torch.log(nu * 3.14159265)
    term4 = torch.log(sigma + 1e-12)
    term5 = ((nu + 1) * 0.5) * torch.log1p((z ** 2) / nu)
    return term1 - term2 - term3 - term4 - term5

# ======================================================
# Tractability properties enforced by this module
# ------------------------------------------------------
# The circuit is built to be a VALID, TRACTABLE density p(x) so that
# `log p(x)` can be evaluated exactly in one pass (the only query we need
# for anomaly / hallucination detection). Concretely we guarantee:
#
#   * Smoothness (completeness): every SumNode's children share one scope.
#       -> asserted in SumNode.__init__
#   * Decomposability: every ProductNode's children have disjoint scopes.
#       -> asserted in ProductNode.__init__
#   * Structured decomposability: a single recursive vtree partition is used,
#       so all products over a given scope decompose identically (free by
#       construction; only relevant for advanced queries, kept for cleanliness).
#   * Validity / normalization (Z = 1): leaves are normalized mixture
#       densities and all mixture weights are normalized (log_softmax). There
#       are NO input-dependent sum weights and NO scalar gate on leaf
#       densities, both of which would break normalization.
#
# Determinism (selectivity) is intentionally NOT enforced: it is only needed
# for MAP/MPE queries, which we do not use. Sum nodes stay soft mixtures.
# ======================================================


# ======================================================
# Base Node
# ======================================================
class AbstractNode(nn.Module):
    def __init__(self, child_nodes=None):
        super().__init__()
        # Renamed from 'children' to 'child_nodes' to avoid conflict with nn.Module.children()
        self.child_nodes = nn.ModuleList(child_nodes) if child_nodes else nn.ModuleList()
        # Scope: the set of feature indices this node is defined over.
        # Set by every concrete subclass; used to enforce tractability.
        self.scope = frozenset()

    def forward(self, x, cache=None):
        raise NotImplementedError

    def _memoize(self, x, cache, compute):
        """Evaluate `compute` once per node instance per forward pass.

        The vtree construction shares sub-circuits (it is a DAG, not a tree),
        so without memoization a naive recursive forward recomputes shared
        nodes exponentially. The cache is keyed by node identity.
        """
        if cache is None:
            cache = {}
        key = id(self)
        if key in cache:
            return cache[key]
        out = compute(cache)
        cache[key] = out
        return out


# ======================================================
# Input Node (leaf): a normalized univariate mixture density
# ======================================================
class InputNode(AbstractNode):
    def __init__(self, feature_idx, mu_init=0.0, sigma_init=1.0,
                 sigma_min=1e-2, sigma_max=1e2, floor_frac=0.5):
        # InputNodes have no children
        super().__init__()
        self.feature_idx = feature_idx
        # Scope of a leaf is the single feature it models.
        self.scope = frozenset({feature_idx})

        # Hard scale bounds (the "clamp"). These constrain a *parameter*, not
        # the support of x, so every admissible value still yields a normalized
        # leaf density -> smoothness/decomposability/Z=1 (tractability) all hold.
        # `sigma_min`/`sigma_max` are absolute backstops; the *effective* floor
        # is data-relative (see `sigma_floor`, set in fit()). A single global
        # constant is wrong for LLM features whose per-dim scales vary by orders
        # of magnitude, so we floor each leaf at `floor_frac` of its OWN
        # data-calibrated scale -> per-feature log-density (hence total NLL)
        # cannot exceed its calibrated value by more than -log(floor_frac)/dim.
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.floor_frac = float(floor_frac)
        # Per-leaf data-relative floor; filled in fit(). Defaults to sigma_min.
        self.register_buffer("sigma_floor", torch.tensor(float(sigma_min)))

        # Learnable Parameters
        self.mu = nn.Parameter(torch.tensor(float(mu_init)))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))
        self.log_nu = nn.Parameter(torch.log(torch.tensor(5.0)))
        self.logits = nn.Parameter(torch.zeros(3))

        # MAP (soft) prior on the scale: a Gaussian on log_sigma anchored at the
        # data-fitted value (set in fit()). Penalizes drift toward collapse
        # *without* re-normalizing the density (unlike the old leaf gate), so it
        # keeps a proper Z=1 NLL. Stored as a buffer: moves with .to(), is
        # saved/loaded, but is not trained.
        self.register_buffer("log_sigma_prior_mean", torch.zeros(()))

    def fit(self, X):
        """Statistical initialization"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        vals = X[:, self.feature_idx].reshape(-1)

        mu = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu)) + 1e-6)
        sigma = mad * 1.4826

        with torch.no_grad():
            self.mu.fill_(mu)
            self.log_sigma.fill_(np.log(sigma))
            self.log_nu.fill_(np.log(5.0))
            self.logits.fill_(0.0)
            # Anchor the MAP prior at the data-estimated scale so the penalty
            # shrinks log_sigma toward the empirical value, not toward sigma=1.
            self.log_sigma_prior_mean.fill_(np.log(sigma))
            # Data-relative hard floor, computed in the SAME (softplus) space the
            # forward pass uses, so it tracks the actual initial density scale
            # regardless of the softplus reparameterization. sigma can't shrink
            # below floor_frac of its calibrated value -> NLL stays bounded near
            # the calibrated level instead of running to -inf.
            s0 = F.softplus(self.log_sigma) + 1e-6
            self.sigma_floor.fill_(max(self.sigma_min, self.floor_frac * s0.item()))

    def forward(self, x, cache=None):
        return self._memoize(x, cache, self._compute)

    def _compute(self, cache):
        # cache may carry the input under a reserved key set by the root.
        x = cache["__input__"]
        vals = x[:, self.feature_idx]

        # softplus keeps sigma > 0; the clamp adds the hard floor/ceiling that
        # makes the normalized NLL bounded below (kills variance collapse). The
        # 1e-6 is only an fp guard. Clamp bounds a parameter, not x, so the leaf
        # stays a valid normalized density -> tractability is untouched. The
        # floor is data-relative (per-leaf sigma_floor); sigma_max is an absolute
        # backstop. torch.maximum/minimum broadcast against the per-sample sigma.
        sigma = F.softplus(self.log_sigma) + 1e-6
        sigma = torch.minimum(sigma, torch.as_tensor(self.sigma_max, device=sigma.device, dtype=sigma.dtype))
        sigma = torch.maximum(sigma, self.sigma_floor)
        nu    = F.softplus(self.log_nu) + 1e-3

        log_gauss = log_gaussian_jit(vals, self.mu, sigma)
        log_lapl  = log_laplace_jit(vals, self.mu, sigma)
        log_stud  = log_student_jit(vals, self.mu, sigma, nu)

        # Determine the target shape dynamically
        # If input is (N, C), vals is (N,). Target: (N, 1, 1)
        # If input is (B, C, H, W), vals is (B, H, W). Target: (B, 1, H, W)
        target_shape = [vals.shape[0], 1] + list(vals.shape[1:])

        log_gauss = log_gauss.view(*target_shape)
        log_lapl = log_lapl.view(*target_shape)
        log_stud = log_stud.view(*target_shape)

        # Vector of size 3 -> (3, 1, 1...) to match stack
        w_shape = [3] + [1] * len(target_shape)
        w = torch.log_softmax(self.logits, dim=0).view(*w_shape)

        stack = torch.stack([log_gauss, log_lapl, log_stud], dim=0)

        # Normalized mixture over {Gaussian, Laplace, Student-t}: a valid density.
        log_mix = torch.logsumexp(w + stack, dim=0)

        return log_mix


# ======================================================
# Sum Node: a mixture. SMOOTHNESS is enforced here.
# ======================================================
class SumNode(AbstractNode):
    def __init__(self, child_nodes, weights=None):
        super().__init__(child_nodes)
        if len(child_nodes) == 0:
            raise ValueError("SumNode requires at least one child.")

        # --- Smoothness (completeness): all children must share one scope ---
        scopes = {c.scope for c in child_nodes}
        if len(scopes) != 1:
            raise ValueError(
                f"SumNode violates smoothness: children have differing scopes {scopes}."
            )
        self.scope = next(iter(scopes))

        n = len(child_nodes)
        if weights is None:
            self.weights = nn.Parameter(torch.full((n,), -np.log(n)))
        else:
            w_t = torch.tensor(weights, dtype=torch.float32)
            self.weights = nn.Parameter(torch.log(w_t / w_t.sum()))

    def forward(self, x, cache=None):
        return self._memoize(x, cache, self._compute)

    def _compute(self, cache):
        x = cache["__input__"]
        child_outputs = [c(x, cache) for c in self.child_nodes]
        stack = torch.stack(child_outputs, dim=0)

        # log_w: (N_children,) -> (N_children, 1, 1...) to broadcast over stack.
        w_shape = [-1] + [1] * (stack.dim() - 1)
        log_w = torch.log_softmax(self.weights, dim=0).view(*w_shape)

        return torch.logsumexp(stack + log_w, dim=0)


# ======================================================
# Product Node: a factorization. DECOMPOSABILITY is enforced here.
# ======================================================
class ProductNode(AbstractNode):
    def __init__(self, child_nodes):
        super().__init__(child_nodes)
        if len(child_nodes) == 0:
            raise ValueError("ProductNode requires at least one child.")

        # --- Decomposability: children scopes must be pairwise disjoint ---
        union = set()
        for c in child_nodes:
            if union & c.scope:
                raise ValueError(
                    "ProductNode violates decomposability: overlapping child scopes "
                    f"({union & c.scope})."
                )
            union |= c.scope
        self.scope = frozenset(union)

    def forward(self, x, cache=None):
        return self._memoize(x, cache, self._compute)

    def _compute(self, cache):
        x = cache["__input__"]
        child_outputs = [c(x, cache) for c in self.child_nodes]
        stack = torch.stack(child_outputs, dim=0)
        # Factorized log-density = sum of disjoint-scope log-densities.
        return torch.sum(stack, dim=0)


# ======================================================
# PCNet: builds a tractable density circuit via a recursive vtree
# ======================================================
class PCNet(nn.Module):
    """A smooth + decomposable probabilistic circuit over flat feature vectors.

    The structure is a region graph induced by a recursively-sampled binary
    vtree: each scope is split into two disjoint halves, sub-circuits are built
    for each half, products combine one node per half (disjoint -> decomposable)
    and sums mix products of identical scope (-> smooth). Recursion always
    descends to singleton scopes, so every input feature is covered and the root
    scope equals the full feature set; `log p(x)` is therefore a proper density.

    Hyperparameters
    ---------------
    n_sum_components : region width (# sum nodes per internal region).
    n_leaf_components: # leaf mixture components per feature.
    n_repetitions    : # independent vtrees mixed at the root (expressiveness).
    max_branching    : back-compat alias; if the *_components args are None they
                       default to this (so existing ablations keep working).
    max_depth        : retained for back-compat with callers; it does NOT cap
                       recursion (capping would drop variables and break
                       coverage). Structure depth is determined by the data
                       dimensionality (~log2 of #features).
    """

    def __init__(self, n_classes=1, max_depth=4, max_branching=3, seed=42,
                 n_sum_components=None, n_leaf_components=None, n_repetitions=1,
                 sigma_min=1e-2, sigma_max=1e2, floor_frac=0.5,
                 sigma_floor_budget=5.0):
        super().__init__()
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.seed = seed
        # Per-leaf scale bounds, threaded into every InputNode (see _build_region).
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        # Dimension-aware floor: total log-density gain from sigma shrinkage is
        # capped at ~sigma_floor_budget nats regardless of #features. The actual
        # per-leaf fraction is recomputed from the feature count in init_network;
        # `floor_frac` here is only the fallback if init_network isn't called.
        self.sigma_floor_budget = sigma_floor_budget
        self.floor_frac = floor_frac
        self.n_sum_components = n_sum_components if n_sum_components is not None else max_branching
        self.n_leaf_components = n_leaf_components if n_leaf_components is not None else max_branching
        self.n_repetitions = n_repetitions
        self.n_features = None
        self.root = None

    def init_network(self, inputs):
        """Builds the tractable PC structure based on input data dimensionality."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        rng = random.Random(self.seed)

        # Feature count C (works for (B, C) and (B, C, H, W)).
        C = inputs.shape[1]
        self.n_features = C
        full_scope = list(range(C))

        # Dimension-aware floor fraction: cap the *total* log-density gain from
        # sigma shrinkage across all C features at `sigma_floor_budget` nats.
        # excess = sum_dim -log(frac) = -C*log(frac)  ->  frac = exp(-budget/C).
        # This binds immediately (per-leaf sigma can only shrink ~budget/C) so
        # NLL stays within `budget` of its calibrated level instead of running
        # off. The residual term (x-mu)^2/sigma^2 only ever lowers log p, so this
        # is a true upper bound on log p regardless of projector/mean dynamics.
        self.floor_frac = float(np.exp(-self.sigma_floor_budget / max(C, 1)))

        # Build n_repetitions independent vtrees, each yielding a full-scope root.
        rep_roots = []
        for _ in range(self.n_repetitions):
            region_nodes = self._build_region(inputs, full_scope, rng)
            rep_root = region_nodes[0] if len(region_nodes) == 1 else SumNode(region_nodes)
            rep_roots.append(rep_root)

        # All repetition roots share the full scope, so mixing them stays smooth.
        self.root = rep_roots[0] if len(rep_roots) == 1 else SumNode(rep_roots)

        # Coverage check: the density must be over ALL features.
        assert self.root.scope == frozenset(full_scope), (
            f"Root scope {sorted(self.root.scope)} != all features 0..{C - 1}"
        )

    def forward(self, x):
        if self.root is None:
            raise RuntimeError("PCNet not initialized. Call init_network(data) first.")
        # Stash the input in the cache so shared sub-circuits read it once.
        cache = {"__input__": x}
        return self.root(x, cache)

    def scale_prior_penalty(self):
        """MAP regularizer: sum over unique leaves of (log_sigma - prior_mean)^2.

        This is the negative log of a Gaussian prior on each leaf's log_sigma
        (anchored at its data-fitted value), summed over the circuit. Multiply
        by `lambda` and add to the training loss to turn MLE into MAP. It only
        regularizes parameters — the density stays normalized (Z=1).

        `modules()` is memo-guarded, so each shared sub-circuit node in the DAG
        is visited exactly once (unlike `children()`, which is why _apply/train
        are overridden but this is not).
        """
        total = None
        for m in self.modules():
            if isinstance(m, InputNode):
                term = (m.log_sigma - m.log_sigma_prior_mean) ** 2
                total = term if total is None else total + term
        if total is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return total

    @torch.no_grad()
    def sigma_stats(self):
        """Diagnostic: post-clamp leaf-sigma summary. Tells us whether the
        collapse is sigma shrinking to the floor (frac_at_floor -> 1, sigma_min
        tiny) vs. something else (frac_at_floor stays ~0)."""
        sig, floors = [], []
        for m in self.modules():
            if isinstance(m, InputNode):
                s = F.softplus(m.log_sigma) + 1e-6
                hi = torch.as_tensor(m.sigma_max, device=s.device, dtype=s.dtype)
                s = torch.maximum(torch.minimum(s, hi), m.sigma_floor)
                sig.append(s.reshape(()))
                floors.append(m.sigma_floor.reshape(()))
        if not sig:
            return {"sigma_min": float("nan"), "sigma_mean": float("nan"), "frac_at_floor": float("nan")}
        sig = torch.stack(sig)
        floors = torch.stack(floors)
        at_floor = (sig <= floors * 1.001).float().mean().item()
        return {"sigma_min": sig.min().item(),
                "sigma_mean": sig.mean().item(),
                "frac_at_floor": at_floor}

    def _apply(self, fn, recurse=True):
        """DAG-safe override of nn.Module._apply (drives .to/.cuda/.half/.float).

        The region graph built by `_build_region` is a DAG: sub-circuits are
        shared across many parents. The default `nn.Module._apply` walks
        `children()` once per incoming edge, so for a shared DAG it re-applies
        `fn` to the same nodes exponentially often (depth ~log2(n_features),
        multiplicity ~n_sum_components*n_leaf_components per level). That makes
        `.to()` appear to hang. The forward pass dodges this via `_memoize`;
        we do the analogous thing here — visit every *unique* module once and
        move only its own tensors (recurse=False reuses torch's own
        param/buffer/grad handling, so semantics match a normal move).
        """
        seen = set()

        def visit(module):
            if id(module) in seen:
                return
            seen.add(id(module))
            if recurse:
                for child in module.children():
                    visit(child)
            nn.Module._apply(module, fn, recurse=False)

        visit(self)
        return self

    def train(self, mode: bool = True):
        """DAG-safe train()/eval() (same rationale as `_apply`).

        `nn.Module.train` recurses through `children()` once per incoming
        edge. The region graph shares sub-circuits across many parents, so the
        default walk re-visits shared nodes exponentially often (multiplicity
        ~n_sum_components*n_leaf_components per level, depth ~log2(n_features))
        and never finishes — the process churns live in `module.train`, GPU
        idle. We visit every *unique* module once instead. `eval()` routes
        through here via `train(False)`, so both modes are covered.
        """
        seen = set()

        def visit(module):
            if id(module) in seen:
                return
            seen.add(id(module))
            module.training = mode
            for child in module.children():
                visit(child)

        visit(self)
        return self

    # ------------------------------------------------------------------
    # Recursive vtree / region-graph builder
    # ------------------------------------------------------------------
    def _build_region(self, inputs, scope, rng):
        """Return a list of nodes (all with scope == frozenset(scope))."""
        # --- Leaf region: a single feature ---
        if len(scope) == 1:
            v = scope[0]
            leaves = []
            for _ in range(self.n_leaf_components):
                leaf = InputNode(v, sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                 floor_frac=self.floor_frac)
                leaf.fit(inputs)  # median/MAD initialization (also anchors MAP prior)
                leaves.append(leaf)
            return leaves

        # --- Internal region: split scope into two disjoint halves (vtree) ---
        s = list(scope)
        rng.shuffle(s)
        mid = len(s) // 2
        left, right = s[:mid], s[mid:]

        left_nodes = self._build_region(inputs, left, rng)
        right_nodes = self._build_region(inputs, right, rng)

        # Products: one node from each half -> disjoint scopes (decomposable).
        products = [ProductNode([l, r]) for l in left_nodes for r in right_nodes]

        # Sums: each mixes the same products -> identical scope (smooth).
        return [SumNode(products) for _ in range(self.n_sum_components)]
