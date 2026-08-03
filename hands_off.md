# Hand-off — DeepFakeDet (PC-based deepfake detection)

Last updated: 2026-07-16. Read `POC.md` first for the full story; this file is the
operational handoff: current state, environment, gotchas, and the roadmap.

## Current state

- **Feasibility: proven at POC level.** A `DensityPC` trained by exact NLL on real
  LFW faces only separates pseudo-fakes at AUROC 0.81 (self-blend) / 0.96 (down-up)
  in a 34-d forensic feature space, matching Mahalanobis, close to GMM-full(4)
  (0.84 / 0.99). All circuit properties audited green after every run.
- **Literature gap confirmed (July 2026):** no published PC/SPN + deepfake work.
  Closest: SeeABLE (ICCV 2023), OC-FakeDect (CVPRW 2020) — one-class but
  approximate scores; UnivFD (CVPR 2023) — frozen CLIP features substrate.
- Nothing is committed anywhere — the project directory is **not a git repo** yet.

## Environment

```bash
PY=~/miniconda3/envs/cvad_venv/bin/python   # torch 2.8, torchvision 0.23, cv2, sklearn, scipy
```

- Homebrew `python3` has no ML packages — don't use it.
- LFW auto-downloads to `~/scikit_learn_data/` on first run (~230 MB).
- Feature caches land in the session scratchpad (`POC_CACHE` env var to relocate —
  **default cache path is session-specific and will vanish**; set
  `POC_CACHE=/some/stable/path/features_{forensic|resnet}.npz` for reuse).
- Apple-silicon MPS is used for ResNet feature extraction only; PC training is CPU
  (scalar parameters, Python recursion).

## Reproduce

```bash
POC_FEATURES=forensic POC_WHITEN=1 $PY -u src/poc_deepfake_pc.py   # main result
POC_FEATURES=forensic              $PY -u src/poc_deepfake_pc.py   # no-whitening ablation
POC_FEATURES=resnet                $PY -u src/poc_deepfake_pc.py   # negative control (~0.55)
```

Knobs at the top of `src/poc_deepfake_pc.py`: `N_PCA`, `LEAF_COMPONENTS` (4),
`SUM_COMPONENTS` (K=2), `EPOCHS` (300), `LR`, `JITTER` (0.2). Each PC ≈ 2–4 min.

## Gotchas (each cost real debugging time — don't rediscover them)

1. **Product-of-marginals collapse (library bug, still open).**
   `fit_leaves(jitter=…)` in `src/bak/allinone_probabilistic_circuits.py` jitters
   only scalar-`mu` leaves; `GaussianMixtureLeaf` (`mus`/`log_sigmas`, plural) gets
   none → K sibling subtrees start identical → gradient symmetry → circuit
   degenerates to a product of marginals, silently. **Detector:** train with a
   learned AND a random vtree; byte-identical NLL curves ⇒ collapse. POC
   workaround: `SymBrokenGMLeaf` in `src/poc_deepfake_pc.py`. Proper fix: make
   `_fit_leaves_with_jitter` handle vector-parameter leaves (or jitter inside each
   leaf's `fit()`).
2. **Semantic embeddings are dead on arrival.** Pooled ResNet/ImageNet features →
   every density model at chance. Artifact signal is spectral/noise-level/local.
3. **Whiten before the circuit.** Full-dim PCA whitening (fit on real train) is a
   fixed bijection — properties preserved — and stops the tree from paying for
   linear correlations. Corollary: after whitening the vtree barely matters;
   structure-learning ablations only make sense on unwhitened features.
4. **DensityPC size explodes as K^depth.** `_build` duplicates entire subtrees per
   sum component: nodes ≈ d·K^depth. d=34, K=2 is fine (~30k params); K=4 at
   d≥100 will not be. Scaling needs a vectorized/einsum implementation, not bigger K.
5. **Python stdout buffers when piped to a file** — run long jobs with `python -u`
   or you'll see nothing for minutes.
6. **sklearn LFW min_faces_per_person=20** gives 3023 images / 62 identities;
   identity-disjoint split is done on identity ids, and pseudo-fakes are built from
   TEST identities only. Keep it that way — leakage here invalidates everything.

## Real-deepfake validation (2026-07-16, evening) — read this before anything else

`POC_DATA=openforensics` runs the pipeline on real GAN face swaps (HF
`Hemg/deepfake-and-real-images`, OpenForensics-derived; one parquet shard
cached at the `POC_OF_PARQUET` path, ~366 MB). **Result: chance-level for
every global feature space and every density model** — forensic 0.463, ResNet
0.422, CLIP ViT-L/14 pooled 0.450 (PC), baselines identical; consistently
slightly below 0.5 (fakes are closer to the mode — smooth GAN faces);
two-sided typicality |NLL − median| does NOT recover it (0.44–0.53).

Interpretation: in-context swaps match reals in global statistics by
construction; the signal is local (blend boundary, swapped interior) and
survives in neither pooled embeddings nor global forensic stats. Supervised
CNNs get >95% here, so the signal exists. The LFW pseudo-fake successes were
real but reflect *global* artifacts my generators introduced.

The project's viable directions, in order: (1) per-patch densities over
artifact-amplifying representations — CLIP-ViT PATCH tokens (not pooled),
SRM/NPR residuals, DIRE reconstruction errors — with max/consistency
aggregation; (2) SeeABLE-style pseudo-fake-contrastive embedding shaping, PC
density on top (keeps real-only protocol, adds exact likelihood + marginals);
(3) validation on FaceForensics++ c23 (form-gated; the Kaggle-style mirrors
recompress both classes and erase low-level cues). CLIP runs need
`expllm_env` (transformers 5.3; cv2 was pip-installed into it 2026-07-16).

## Roadmap (ordered)

1. **Real deepfake data.** Swap pseudo-fakes for FaceForensics++ (per-method
   splits: train-free eval on FS/F2F/NT/DF), then Celeb-DF v2 for the hard
   cross-dataset test. Keep the real-only training protocol — that's the selling
   point vs binary classifiers (generalization to unseen generators).
2. **CLIP-ViT patch features.** Replace hand-crafted forensics with frozen
   CLIP-ViT patch tokens (UnivFD substrate). Per-patch PC scores → image score =
   max/mean NLL over patches. This is also the entry to localization.
3. **Localization by exact marginals (the paper's headline).** For a flagged
   image, query `p(z_S)` over feature/patch subsets (Khosravi-style submodular
   selection of the most anomalous subset) → "which region is fake" with exact
   probabilities. No competing one-class detector can do this exactly.
4. **Close the GMM gap / scale.** Vectorized einsum layers (or PyJuice), EM
   init or LVD-style latent supervision, K>2, SquaredPC (SOS) as the expressive
   variant — it's already in the library with exact Z via the pairwise recursion.
5. **Fix the jitter bug upstream** in `allinone_probabilistic_circuits.py` and add
   a regression test: learned-vtree NLL must beat random-vtree NLL on correlated
   synthetic data by a margin.
6. **Robustness protocol.** JPEG compression sweep, resize sweep, cross-dataset —
   forensic/spectral features are notoriously fragile to recompression; this
   determines whether the CLIP route (2) is mandatory rather than optional.

## Explainability (added 2026-07-16, later same day)

`src/poc_explainability.py`: 4x4 patch grid × 4 forensic stats each (64-d,
z-scored, **never whitened** — linear mixing would destroy the patch identity of
coordinates). Two exact queries per patch: marginal `−log p(z_patch)` (one pass)
and conditional `−log p(z_patch | z_rest)` = `log p(z) − log p(z_−i)` (two
passes). Localization vs ground-truth blend masks: AUROC 0.672 (marginal) /
0.648 (conditional); image-level 0.678. All 16 patches × 2846 images scored in
~4 s. Mechanism proven; per-patch features are the weak link → CLIP-ViT patch
tokens are the upgrade path. Overlays: `results_explain/patch_marginal_heatmaps.png`.

Also learned: **500 epochs overfit** (train NLL 38.0→34.3, test self-blend AUROC
0.807→0.796) — add val-NLL early stopping; longer Adam will not close the GMM gap.
The framing note: GMM-full is itself a shallow PC (one sum over multivariate-
Gaussian leaves), so block-multivariate-Gaussian leaves inside DensityPC are the
principled way to inherit its covariance modeling while keeping the deep structure.

## Evidence trail

- POC runs (2026-07-16): resnet ~0.55 everywhere → forensic raw PC 0.684/0.958,
  Mahalanobis 0.735/0.962, GMM 0.836/0.981 → forensic whitened PC 0.807/0.959,
  Mahalanobis 0.807/0.960, GMM 0.842/0.986.
- Structure evidence (raw forensic, post-fix): train NLL 13.98 (chow-liu) vs
  27.51 (random).
- Property audit green in every run: smooth / decomposable / structured /
  log Z = 0 / exact-marginal consistency.
