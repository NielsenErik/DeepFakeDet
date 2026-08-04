# PC deepfake detection

Deepfake detection as **exact density estimation with probabilistic circuits**:
fit a smooth, decomposable, structured-decomposable circuit to REAL faces only,
and score any face by exact likelihood queries — including queries no competing
one-class detector can answer exactly, such as *"how surprising is this patch
given the rest of the face"*.

This is the transfer of PCNET (arXiv:2605.05953 — circuits as tractable density
estimators over LLM residual streams) to vision, rebuilt on real data after the
proof-of-concept in `POC.md` returned an honest negative on globally pooled
features.

## What is here

```
pcdf/
  circuits/     tensorized executor for the reference library's region graphs
                (einsum_pc.py), structure learning (structure.py), leaf layer
  data/         manifests + protocol assertions, face ingestion, SBI, transforms
  features/     CLIP / DINOv2 / SBI-encoder / SRM patch extractors, projector
  models/       the PC detector, one-class baselines, the SBI trained baseline
  eval/         detection metrics, report + pre-registered decision rubric
  explain/      exact per-patch localization and the anomalous-subset query
  cli.py        `pcdf <stage>`; stages.py holds the stage implementations
src/probabilistic_circuits.py   the reference library — the SPECIFICATION
tests/          equivalence with the reference; structure/collapse regressions
scripts/        smoke_pipeline.py — full dry run on synthetic data, CPU, minutes
```

`src/probabilistic_circuits.py` is authoritative for semantics. `pcdf` never
reimplements the model: it reimplements the *execution* and is pinned to the
reference numerically by `tests/test_equivalence.py`.

## The one non-obvious engineering claim

`RegionGraphPC` in the reference library already fixed the parameter blowup
(O(d·K²) rather than O(d·K^depth)). What it did not fix is compute: every unit
is an `nn.Module` and every query is a memoised Python recursion, which is
~10⁵ Python calls per forward pass at the dimensionality this project needs.

`EinsumPC` evaluates the *same circuit* by grouping regions of equal shape and
running each level as one batched einsum. Consequences:

* it trains on a GPU at d = 1024–4096 instead of not finishing;
* marginalization is a mask on the leaf layer, so **all 64 per-patch marginals
  for a batch of images are one forward pass** — which is what makes exact
  localization affordable at dataset scale rather than a demo on eight images.

Verified, not asserted: parameters are copied between the two implementations
and `log p(x)`, exact marginals over random masks, box queries and `log Z` agree
to 2e-4 across Chow-Liu, ORC, multi-partition ORC, Forman, spectral and random
structures.

## Structure

Region graphs come from the reference library's learners, and both learners the
project cares about are first-class:

* **Chow-Liu** — mutual-information max-spanning-tree, weakest-edge recursion.
* **Ollivier-Ricci curvature** — cuts at neighbourhood-aware bottlenecks, with
  exact W₁ per edge; `orc_multi` keeps several cut depths as alternative
  partitions (more expressive, gives up structured decomposability).
* Adversaries that keep the story honest: **spectral** normalized cut, a
  **kd-tree** spatial prior, and a **random** control.

For image features the graph is composed hierarchically — a patch-level graph
over the patch grid, expanded with a channel-level graph inside each patch — so
**every patch is a region of the circuit** and its marginal/conditional are
exact queries on a scope the model already represents.

`tests/test_structure_matters.py` asserts the two things that must hold for any
of this to mean anything: K > 1 beats a product of marginals (7.2 nats on
block-correlated data), and learned structure beats random (4.7 nats). Those are
the regression guards for the silent mixture-collapse bug documented in
`POC.md`.

## Protocol

* FF++ c23, **official identity-disjoint splits** (720/140/140 video pairs).
* The circuit and every one-class baseline are fitted on **real training faces
  only**; `assert_no_fakes_in_train` and `assert_identity_disjoint` enforce it.
* Cross-dataset test sets are scored by the same frozen models.
* Baselines live in the *identical* feature space with the *identical*
  aggregation: Mahalanobis, full-covariance GMM, PatchCore, RealNVP flow, plus
  the SBI-trained EfficientNet-B4 as the supervised real-only competitor.

## Running it

See `STATUS.md` for machine-specific state and the current resume point.

```bash
pcdf manifest --datasets ffpp
pcdf ingest   --dataset ffpp --masks          # videos -> face crops (+ masks)
pcdf features --dataset ffpp                  # crops -> projected patch features
pcdf fit-pc                                   # the circuit, real faces only
pcdf baselines
pcdf evaluate --datasets ffpp
pcdf explain  --n-images 2000                 # exact localization
pcdf ablate-structure                         # ORC vs Chow-Liu vs spectral vs random
pcdf report                                   # -> results/<tag>/REPORT.md + verdict
```

Dry-run everything downstream on synthetic data, on a laptop, in ~1 minute:

```bash
python scripts/smoke_pipeline.py
```

## The decision this repo exists to make

`pcdf/eval/report.py` contains a **pre-registered rubric** written before any
numbers existed: gates for in-dataset detection, cross-dataset generalization,
whether the circuit beats non-circuit one-class baselines on detection *or*
localization, whether structure learning pays, and whether it scales. It emits
one of PURSUE / REFRAME / STOP. The point is to make "is this worth pursuing"
answerable by evidence rather than by attachment to the method.
