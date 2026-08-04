"""
Access to the reference circuit library `src/probabilistic_circuits.py`.

That file is the specification: its region graphs, structure learners,
validators and object-graph circuits define the semantics this package must
reproduce.  `pcdf.circuits.einsum_pc` re-implements only the *execution* — the
same circuit, evaluated as batched einsums on the GPU — and the equivalence
test in `tests/test_equivalence.py` pins it to the reference numerically.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probabilistic_circuits import (  # noqa: E402,F401
    # region graphs
    RegionNode,
    region_nodes,
    region_graph_from_vtree,
    is_structured_decomposable_rg,
    region_graph_arity,
    flatten_region_graph,
    curvature_region_graph,
    spectral_region_graph,
    learned_region_graph,
    # vtrees + structure learners
    VtreeInternal,
    VtreeLeaf,
    chow_liu_vtree,
    curvature_vtree,
    spectral_vtree,
    learned_vtree,
    random_balanced_vtree,
    vtree_depth,
    # curvature machinery (used for structure diagnostics)
    mutual_information_matrix,
    sparsify_mi_graph,
    ollivier_ricci_curvature,
    forman_curvature,
    curvature_sign_stability,
    # object-graph circuits (reference implementation)
    DensityPC,
    RegionGraphPC,
    SquaredPC,
    GaussianLeaf,
    GaussianMixtureLeaf,
    LeafNode,
    SumNode,
    ProductNode,
    # exact inference + validators
    eval_log_prob,
    eval_log_marginal,
    log_partition,
    circuit_size,
    validate_circuit,
    validate_smoothness,
    validate_decomposability,
    validate_structured_decomposability,
)

__all__ = [n for n in dir() if not n.startswith("_")]
