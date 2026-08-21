# Hand-off — DeepFakeDet

Last updated: **2026-08-22.** The GPU is free of our jobs. Everything is
committed; the branch is **9 ahead / 2 behind `origin/main`** and needs a merge
and a push (§7).

Read order for a new session: **this file → `STATUS.md` (Findings 0–19, newest
first) → `PLAN.md` (now largely obsolete — see §6)**. `POC.md` and the July
appendix are superseded.

---

## 1. Where the project actually is

The POC set out to transfer PCNET (probabilistic circuit as tractable density
estimator) from LLM hallucination detection to deepfakes. **The POC is now
finished and it has an answer.**

The one-sentence state: **two results survive two datasets and a validated
pipeline — the exact log-ratio is nearly domain-invariant, and exact per-region
conditionals give the best localization — and everything else this project
believed for four months was an artefact of a reimplementation that does not
transfer, or of a comparison against the wrong published number.**

### The table that matters (`results/full_picture_*.json`)

Both columns on the **published SBI encoder**, same crops, one process
(`scripts/full_picture.py`). Only the backbone differs from the older numbers.

| stage | FF++ c23 | Celeb-DF-v2 | drop |
|---|---|---|---|
| published SBI (reported, repo table) | — | 0.9287 | — |
| official encoder, end to end | 0.8657 | 0.8921 | +0.026 |
| linear probe (supervised, same features) | **0.9192** | **0.8588** | −0.060 |
| circuit, one-class (best of family) | 0.6944 | 0.6823 | −0.012 |
| **circuit, exact log-ratio** | 0.8411 | 0.8357 | **−0.005** |
| circuit, density − mass | 0.3270 | 0.3908 | — |

**Localization, FF++ only** (`results/official_sbi_g8c16_kd-orc_K8/p0_experiments.json`).
Celeb-DF ships no manipulation masks, so this can never be tested cross-dataset.

| model | patch AUC | per-image | IoU | pointing |
|---|---|---|---|---|
| **PC ratio (conditional)** | **0.7137** | **0.7100** | **0.3321** | **0.3886** |
| mahalanobis | 0.6860 | 0.6541 | 0.2894 | 0.2559 |
| gmm | 0.6838 | 0.6590 | 0.2870 | 0.3870 |
| patchcore | 0.6729 | 0.6444 | 0.2723 | 0.2559 |
| flow | 0.6562 | 0.6280 | 0.2444 | 0.3183 |

---

## 2. The two results that survive

**S1 — the exact log-ratio is nearly domain-invariant.** 0.8411 on FF++ →
0.8357 on Celeb-DF, a drop of **0.005**, where a supervised probe on *identical
coordinates* drops 0.060 and the one-class family drops 0.012. It never sees a
real forgery.

*The caveat that must ship with it:* it never beats the classifier it sits on
(0.8921 on CDF), and that classifier also needs no real forgeries. So the
selling point is **stability, not accuracy**, and a reviewer will note that a
score starting lower has less room to fall. n = 2 datasets cannot separate those.

**S2 — exact per-region conditionals give the best localization.** First on all
four metrics, including `pointing`, which the circuit *lost* on our own encoder
(0.3128 vs PatchCore 0.4392). The only stage that got better under the rerun.

*The caveat:* the margin over the **strongest** baseline fell from +0.0655
(vs patchcore, our encoder) to **+0.0277** (vs mahalanobis, official encoder).
`PLAN.md`'s pre-registered gate is +0.03 over the strongest patch baseline, so
**this narrowly misses the gate.** Against PatchCore alone it is +0.0408 and
clears — quoting only that comparison would clear a gate the result does not.
Do not do this.

Both rest on the same mechanism, and it is the honest version of "only our model
can do this": a smooth, decomposable circuit can **condition on 10³ dimensions
and integrate over a region exactly**. Flows, diffusion models and
full-covariance Gaussians cannot. (A *diagonal* GMM can do a plain global box —
so the exclusivity claim must be about the *conditional* query, not integration
in general.)

---

## 3. What died, and why — do not re-derive these

| claim | status | why |
|---|---|---|
| "our encoder is weak, 0.14 behind SotA" | **false premise** | the 0.9964 target is FF++ **raw** from a raw-trained model (paper Table 2, §4.4). No FF++ c23 in-dataset number is published by anyone. Finding 11 |
| the pipeline is broken | **false** | official weights through our pipeline reproduce CDF **0.9077 vs published 0.9287**, using a different face detector. Finding 12 |
| F1: SBI leaks a global compression cue | **our bug, not SBI's** | official `dynamic_blend` composites onto the array it returns as the real, and its JPEG is in a *shared* transform. Measured: periphery bit-identical beyond one 8×8 block. Finding "F1 does not generalise" |
| F9: forgeries are lower-dimensional (mass beats density) | **artefact of our encoder** | the dimension gap **inverts** on the published encoder on *both* datasets (FF++ 838.5 real vs 884.4 fake; CDF 858.0 vs 876.1). `density − mass` = 0.327 / 0.391, below chance. Finding 16 |
| F14: the circuit beats supervision cross-dataset | **only on the broken encoder** | on official features the probe wins both (0.8588 vs 0.8357). The probe only collapsed on our features because they carried FF++-specific structure. Finding 17 |
| C5 calibration | **falsified** | ECE 0.780 raw, 0.766 after temperature scaling (T=110) |
| crop geometry explains the encoder gap | **tested, rejected** | their crop rule buys +0.0058 on FF++, +0.0156 on CDF — inside the noise |
| hires / SAM / hull / leak-removal fix the encoder | **all failed** | Findings 3, 8; and the ablation **re-ranks** on Celeb-DF (§4) |

### Still unexplained

**`noleak_clean` costs 0.064 in-dataset AND 0.063 cross-dataset.** Removing the
leak cleanly (`pristine_background=True, compress_policy="none"`) makes the
encoder worse *everywhere* — 0.8081 FF++ val, 0.6559 CDF, vs base 0.8716 /
0.7186. The obvious explanations are excluded: it is not a train/test
compression shift (it survives a change of dataset) and not lost compression
exposure (`_augment` still applies JPEG q40–100 at p=0.3 either way). Open since
Aug 6. If F1 is written up at all, this needs an answer first.

---

## 4. The ablation, re-ranked on Celeb-DF

In-dataset val AUC was the wrong metric for every conclusion in Finding 3.
Spearman(FF++ val, CDF) = **0.74**.

| variant | FF++ val | **CDF** |
|---|---|---|
| **hull** | 0.8610 | **0.7370** |
| sam | 0.8747 | 0.7333 |
| base | 0.8716 | 0.7186 |
| all | 0.8543 | 0.6837 |
| hires | 0.7612 | 0.6837 |
| hires_sam | 0.8368 | 0.6801 |
| noleak | 0.8468 | 0.6711 |
| noleak_clean | 0.8081 | 0.6559 |
| all_clean | 0.8333 | 0.6357 |

`hull` looked like a loss in-dataset (−0.011, dismissed as noise) and is the
**best** cross-dataset variant. Randomising the hull so one boundary geometry
cannot be memorised is worth +0.018 where it matters.

Run-to-run noise at this protocol is **≈ ±0.02** (Finding 8, from two runs of
the same variants). Anything smaller is not a result.

---

## 5. Machine, environment, data

```bash
ssh jawa17@192.168.1.8            # RTX 4080 16 GB, Ubuntu
repo   ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
data   ~/deepfake_data            # NOT in the repo
PY     ~/miniconda3/envs/expllm_env/bin/python
```

**Disk: 83 GB free (91% used).** `raw/` is 43 GB, `features/` 11 GB (of which
`combined/` alone is 4.9 GB and can be deleted — see §3, it does not beat its
own better source).

**The `tsp` queue is SHARED with your ExpLLMble project.** At the time of
writing an ExpLLMble ablation is running on the GPU. **Never `tsp -C` or kill
by PID** — check `tsp -l` and only touch jobs you recognise.

### Data state

| artefact | state |
|---|---|
| `crops/` (256px q95, mediapipe, margin 1.3) | complete, 183,723 crops. Everything uses this |
| `crops_hires/` (380px q100 4:4:4) | complete now: train reals 706/720, val all 6 methods 811/840. `hires` failed anyway (Finding 8) |
| `raw/celebdf/` | **Celeb-DF-v2, extracted.** `Celeb-real/`, `YouTube-real/`, `Celeb-synthesis/`, `List_of_testing_videos.txt`. Ingested: 516/518 test videos, 16,059 crops |
| `models/official_sbi/FFc23.tar` | **their released FF++ c23 weights.** 706 tensors, epoch 99 |
| `features/{srm,clip,spectral,sbi,combined,official_sbi}/` | all built |
| `models/pc_official_sbi_*`, `ratio_official_sbi_*`, `baselines_official_sbi_*` | the official arm, fitted |
| `models/sbi_ab_*.pt` | 9 ablation checkpoints |

**All 40+ result JSONs are now committed under `results/`.** They are no longer
workstation-only.

---

## 6. What this does to PLAN.md

`PLAN.md` describes a CVPR 2027 paper with six contributions. **Four are dead**
(C1's novel parts, C4's family-posterior version, C5, and the detection-parity
premise). The 14-week timeline assumed data landing in week 2 and is three weeks
behind with a different destination.

**Rewrite it around S1 and S2.** The honest paper is roughly:

> *What exact tractable inference buys deepfake forensics — and what it doesn't.*
> A domain-invariant one-class detection score that never beats the classifier
> it sits on; the best localization by a margin that just misses our own
> pre-registered gate; and an accounting of five results that looked real and
> were not.

Venue: **UAI / AISTATS / TMLR.** Not CVPR — the detection numbers are not
competitive and the localization margin is small. That last section is not
padding: this project found that four months of conclusions rested on an
unsourced target from the wrong compression level and a reimplementation nobody
had tested cross-dataset. Written up properly it is useful to the field.

---

## 7. Immediate actions

1. **Merge and push.** 9 ahead, 2 behind `origin/main`:
   `git pull --no-rebase && git push`. Nothing is backed up until this happens.
2. **Rewrite `PLAN.md`** per §6 before any further experiments.
3. **Decide on `noleak_clean`** (§3). Either explain it or drop F1 from the
   paper entirely.
4. Optional, cheap: delete `features/combined/` (4.9 GB) and
   `raw/ffpp_zip` if it reappears.

### If more evidence is wanted before writing

* **Seeds.** Every ablation number is single-seed against a ±0.02 noise floor.
  3 seeds on `base` / `hull` / `sam`, scored on **CDF**, would make §4 solid.
* **A third dataset for S1.** DFDCP (85.51%) or FFIW (83.22%) are in the repo's
  reproduction table, so the pipeline can be validated against them the way
  Celeb-DF was. n = 2 is thin for an invariance claim.
  **DFD is not obtainable**: all three FaceForensics mirrors are dead or 403
  (`canis` and `falas` refuse connections, `kaldir` 403s behind its redirect).
* **Localization on a second mask source.** DF40 or FFIW if either ships masks;
  Celeb-DF does not.

---

## 8. Gotchas

1. **`collect_real_items` / `collect_labeled_items` hardcode
   `manifests/ffpp_ingested.csv`.** `crops_dirname` only redirects the directory
   lookup, and a missing directory is **silently skipped**. That is how the
   first `hires` run trained on 14.6% less data and finished without a warning.
2. **The official SBI weights are not drop-in.** `efficientnet_pytorch` (not
   timm), the checkpoint stores the `Detector` wrapper so `net.` must be
   stripped, a 2-way head, and **raw [0,1] input with no ImageNet
   normalization**. Loading them into our timm extractor with `strict=False`
   drops nearly every tensor and does not raise. `OfficialSbiExtractor` refuses
   any mismatch instead.
3. **`log_ball` has a float32 precision floor.** The identity holds to 3e-4 at
   eps=1e-3 and degrades in *both* directions. **Never use eps < 1e-3 in
   float32** — it degrades silently. Pinned by `tests/test_mass.py`.
4. **`patch_surprisal` chunking scales with d now.** `region_log_marginals`
   expands (B, Q) to B·Q rows before the leaf layer; a fixed `chunk_rows=8192`
   never chunked at B=64, P=64, so d=6080 fit in 16 GB and d=7104 OOM'd —
   *after* training finished, inside `calibrate`.
5. **Distrust any unsourced number in these docs.** This has now cost the
   project twice: the `0.860` encoder figure (Aug 6) and the `0.9964` target
   (Aug 4 → Aug 21, four months of misdirection). If a number has no artefact
   behind it, re-measure it before reasoning from it.
6. **In-dataset AUC ranks variants wrongly.** §4. Score on Celeb-DF.
7. `setsid nohup` and `tsp` both survive the parent ssh dying. To stop a `tsp`
   chain, `tsp -r <id>` the queued jobs first — killing the running child just
   starts the next one.

Test suite: **28 passed** (5 new in `tests/test_mass.py`).

---

# Appendix — superseded records

The Aug 6 hand-off and the July POC record are in git history
(`git show 3701bf4:hands_off.md`) and `POC.md`. Two July conclusions still hold:

- **Semantic embeddings are dead on arrival.** Pooled ResNet/ImageNet/CLIP
  features put every density model at chance. Reproduced at scale: CLIP 0.536.
- **The `fit_leaves(jitter=…)` product-of-marginals collapse** in
  `src/bak/allinone_probabilistic_circuits.py` — only scalar-`mu` leaves get
  jitter, so the circuit silently degenerates. `pcdf/circuits/einsum_pc.py` does
  **not** have this bug (`tests/test_equivalence.py` pins it); the old library
  in `src/bak/` still does.
