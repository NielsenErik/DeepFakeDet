# Status — 2026-08-21 (night): the 0.9964 target is the WRONG NUMBER

## Finding 11: two corrections, one of which invalidates the headline gap

### 11a — crop geometry is NOT the explanation (tested, rejected)

Finding 10 proposed that their tighter, aspect-stretched crop explained the
remaining gap. Tested and rejected. `scripts/official_sbi_eval.py --crop-rule
both` re-derives their `crop_face(crop_by_bbox=True, phase='test')` rule inside
our stored crops — same videos, same detector, same frames, only the crop rule
changes:

| crop rule | frame AUC | video AUC |
|---|---|---|
| ours (square, max(w,h)×1.3) | 0.8415 | 0.8496 |
| theirs (bbox + w/8, h/8, stretched to 380) | 0.8483 | **0.8554** |

**+0.0058** — inside the ±0.02 noise floor. The test upsamples from a 256px crop
so it understates the benefit, but not by the 25× that would be needed. Crop
geometry is not where the gap lives.

### 11b — the 0.9964 we have been chasing is FF++ RAW, not c23

From the paper (arXiv:2204.08376), **Table 2**, "Cross-Manipulation Evaluation
on FF++": `EFNB4 + SBIs (Ours)` scores DF 99.99, F2F 99.88, FS 99.91, NT 98.79,
**FF++ 99.64**. Section 4.4 states: *"We use the **raw** version for evaluation
as well as the competitors."*

**We evaluate on FF++ c23.** The paper reports no FF++ c23 in-dataset number,
and neither does the repository — its reproduction table gives only
cross-dataset results for the c23 weights:

| training data | CDF | DFD | DFDC | DFDCP | FFIW |
|---|---|---|---|---|---|
| FF-raw | 93.82 | 97.87 | 73.01 | 85.70 | 84.52 |
| **FF-c23** | **92.87** | **98.16** | **71.96** | **85.51** | **83.22** |

So the `+0.1652` "encoder gap" at the top of `gap_waterfall.json`, and the claim
that 98% of the distance is the encoder, are **measured against a target from a
different compression level and a different model**. The 98% decomposition is
still internally valid — those rows were all measured on our crops — but the
*size* of the gap, and therefore the entire "our encoder is broken" framing, is
not established.

**This is the second unsourced-number incident in this project.** The first cost
a day (the `0.860` encoder figure that no artefact produced, corrected on
Aug 6). This one has been steering the work since Aug 4. The rule from that
correction — *if you find an unsourced number in these docs, distrust it* —
should have been applied to 0.9964 too.

### What is actually established

* Their released c23 weights, on our c23 crops: **0.8610** video AUC
  (mean-over-frames, which is their protocol: 32 frames/video, max over faces
  within a frame, mean over frames — §4.1 and §4.3).
* Our encoder, same crops: **0.8312**.
* Difference: **+0.0298**, just above the noise floor.
* Crop geometry accounts for **+0.0058** of that.
* Whether 0.8610 is a *bad* number for FF++ c23 one-class detection is
  **unknown**, because no published c23 in-dataset figure exists to compare it
  to.

### The only reproducible reference point, and it needs Celeb-DF-v2

There is exactly one published number we can check our pipeline against with the
weights we now hold: **CDF 92.87%** for the FF-c23 model. Run the downloaded
checkpoint through our pipeline on Celeb-DF-v2 and:

* ≈92.9 → the pipeline is correct end to end, 0.86 on FF++ c23 is simply what
  this model does here, and the "encoder gap" narrative is retired.
* ≪92.9 → the defect is in our pipeline and is now localized by a published
  reference rather than guessed at.

Celeb-DF-v2 has been the top-priority blocked item since Aug 6 for Finding 2.
It is now also **the only way to validate the pipeline at all**. Three
independent reasons, one unfilled request form.

Until it runs, the honest statement is: *we do not know how large the gap is,
because we have never had a comparable published number.*

---

# Status — 2026-08-21 (evening): the encoder gap is PREPROCESSING, not weights

## Finding 10: the official SBI weights score 0.8610 on our crops

`scripts/official_sbi_eval.py`, `results/official_sbi_eval.json`. Full FF++
test set, 25,430 crops — the same index the probe and the circuit see, so this
is a difference in encoder and nothing else.

Shiohara & Yamasaki released their trained weights, including one trained on
FF++ c23. The encoder never had to be reproduced; it could be downloaded. All
706 tensors load with **0 unmatched** (epoch 99).

| | FF++ video AUC |
|---|---|
| our encoder, our crops | 0.8312 |
| **official SBI weights, our crops** | **0.8610** |
| official SBI, reported | 0.9964 |

**Their published encoder recovers 18% of the gap on our crops, and +0.0298 over
our own — barely above the ±0.02 noise floor measured in Finding 8.**

This overturns the label on four months of work. `gap_waterfall.json` was right
that 98% of the distance is upstream of the circuit, but "the encoder" was the
wrong name for it. With architecture and weights now held fixed at the published
ones, 82% of the gap is still there. **It is not in the encoder. It is in what
we feed it.**

It also explains Findings 0, 3 and 8 at a stroke: longer training, SAM, hull
variety, leak removal and native-380 crops all failed because **not one of them
touched the crop geometry**. `hires` changed resolution while keeping mediapipe
and margin 1.3, which is why it moved nothing (and, being a different
distribution again, moved it downward).

### The two pipelines, precisely

| | ours (`crop_box`, faces.py:120) | theirs (`crop_face`, crop_by_bbox=True, phase='test') |
|---|---|---|
| detector | mediapipe (bbox derived from dense landmarks) | RetinaFace / dlib 81-landmark |
| extent | **square**, side = max(w,h) × 1.3 | w×1.25 by h×1.25 — bbox + w/8 and h/8 per side |
| aspect | preserved (square crop, square resize) | **not preserved** — a non-square crop is resized to 380×380 |

Two consequences, and the second is probably the larger:

1. **Field of view.** A face bbox is taller than wide, so `max(w,h) × 1.3` takes
   about 1.3·h in *both* directions where theirs takes 1.25·w horizontally. Ours
   is markedly wider and includes much more background.
2. **Aspect distortion.** Their crops are deliberately stretched to square. The
   encoder was trained on stretched faces; we hand it unstretched ones. That is
   a systematic geometric domain shift on *every single image*, which is exactly
   the kind of thing that costs a lot and shows up nowhere in a loss curve.

The docstring on `crop_box` says 1.3 "follows the SBI / DeepfakeBench
convention". Against their released code, it does not follow SBI's.

### Per-method, and why it is not just a constant offset

| method | ours | official weights, our crops |
|---|---|---|
| Deepfakes | 0.921 | **0.9696** |
| FaceSwap | 0.699 | **0.9059** |
| Face2Face | 0.872 | 0.8728 |
| FaceShifter | 0.844 | 0.7851 |
| NeuralTextures | 0.808 | 0.7718 |

Their encoder is much better on Deepfakes and dramatically better on FaceSwap
(+0.21), and *worse* on FaceShifter and NeuralTextures. So the two encoders are
not ranked versions of each other — they have learned different things, and our
crop pipeline is not merely a weaker version of theirs.

### The decisive next test, and an honest alternative hypothesis

The remaining 82% is attributed above to preprocessing, but a second
explanation has not been excluded: **their 0.9964 is measured on their own
protocol** — their detector, their frame sampling, their video list — and our
test crops are a different sample of the same videos.

One experiment separates these, and it is bounded: run **their** inference code
(`src/inference/inference_dataset.py`, RetinaFace + their `crop_face`) on **our**
FF++ c23 videos.

* ≈0.99 → the entire remaining gap is preprocessing. Port `crop_face` and the
  detector, re-ingest, and the detection problem is solved by preprocessing.
  Every downstream result then gets rerun on a competitive backbone.
* ≈0.86 → their published number does not reproduce on our copy of FF++, which
  is a different and considerably more serious story.

Until that runs, "the encoder is weak" should not appear in this file or in the
paper. What is established is narrower and sharper: **with weights and
architecture controlled, the gap survives, so it lives in the data pipeline.**

---

# Status — 2026-08-21 (late): probability MASS fixes the inversion

## Finding 9: scoring by mass instead of density turns 0.21 into 0.79

`scripts/mass_vs_density.py`, `results/sbi_g8c16_kd-orc_K8/mass_vs_density.json`.
Full FF++ test set, 25,430 crops, d = 1024, on the circuit already fitted — no
retraining, no new features, no supervision.

A density is mass per unit volume. Kamkari et al. (ICML 2024, arXiv:2403.18910)
show the likelihood OOD paradox is exactly this confusion: data on a thin,
low-dimensional sheet gets large density and near-zero mass, which is why models
assign OOD inputs high likelihood and never generate them. They must estimate
Local Intrinsic Dimension as a stand-in for volume, because a flow or a
diffusion model cannot integrate its own density. **A smooth decomposable
circuit can**, exactly — `EinsumPC.log_ball`, which was already implemented and
equivalence-tested and had never been used as a score.

| score | FF++ video AUC |
|---|---|
| `density` = −log p(x) | **0.2116** |
| local dimension alone | 0.7258 |
| `density − mass` (Kamkari's dual criterion) | **0.7884** |
| *circuit, one-class NLL family (previously measured)* | *0.8125* |
| *circuit, exact log-ratio (previously measured)* | *0.8283* |

### The inversion is eliminated, on every manipulation

| method | density | LID | density − mass |
|---|---|---|---|
| Deepfakes | 0.1909 | 0.6618 | **0.8091** |
| Face2Face | 0.1722 | 0.7641 | **0.8278** |
| FaceShifter | 0.1799 | 0.7914 | **0.8201** |
| FaceSwap | 0.3127 | 0.6673 | **0.6873** |
| NeuralTextures | 0.2024 | 0.7443 | **0.7976** |
| **pooled** | **0.2116** | 0.7258 | **0.7884** |

Every manipulation was below chance under the likelihood. Every one is above
0.68 under mass. This is the same model, the same features and the same crops —
only the query changed.

### The mechanism, measured

Local dimension read off each adjacent pair of eps (the slope of log P(box)
against log 2ε — an exact LID, not an estimate):

| ε band | d real | d fake | video AUC |
|---|---|---|---|
| 0.003–0.010 | 1024.2 | 1024.0 | 0.6212 |
| 0.010–0.031 | 1024.0 | 1023.7 | 0.6327 |
| 0.031–0.102 | 1019.5 | 1013.9 | 0.7319 |
| 0.102–0.305 | 950.6 | 879.8 | 0.7839 |
| **0.305–1.02** | **657.9** | **415.0** | **0.7894** |
| **1.02–3.05** | **284.5** | **77.2** | 0.7872 |
| 3.05–10.2 | 32.4 | 5.3 | 0.7739 |
| 10.2–30.5 | 0.0 | 0.0 | 0.7401 |
| 30.5–102 | 0.0 | 0.0 | 0.5000 |

**Forgeries occupy roughly a quarter of the effective dimensions real faces do**
(284.5 vs 77.2 at the 1–3 band). Kamkari's prediction, confirmed on deepfakes.

Two sanity checks fall out of the same table. At the smallest ε both classes
read d = 1024, which is correct — every smooth density is full-dimensional at
infinitesimal scale, and it is why the mass score is *identical* to density
there (Spearman ρ = 1.0000 for every ε ≤ 0.03). At the largest ε the box
swallows the support, mass → 1, d → 0 and AUC → 0.500. Normalization confirmed
from both ends.

### The control: mass only helps where there IS a dimension gap

Run on all four fitted circuits. `rel. gap` is the largest relative separation
between real and forged local dimension across the ε sweep; `gain` is the best
mass-based score minus the density it is derived from.

| arm | d | density | LID | density−mass | rel. gap | gain |
|---|---|---|---|---|---|---|
| clip | 1024 | 0.4757 | 0.4421 | 0.5243 | 1.4% | +0.0486 |
| srm | 1024 | 0.4442 | 0.5850 | 0.5558 | 2.4% | +0.1408 |
| spectral | 6080 | 0.4587 | 0.5882 | 0.5413 | 2.9% | +0.1296 |
| combined | 7104 | 0.3172 | 0.6077 | 0.6828 | 5.5% | +0.3656 |
| **sbi** | 1024 | **0.2116** | **0.7258** | **0.7884** | **36.9%** | **+0.5768** |

**The relative dimension gap orders the LID score exactly**: 0.4421, 0.5850,
0.5882, 0.6077, 0.7258 against gaps of 1.4%, 2.4%, 2.9%, 5.5%, 36.9% —
Spearman ρ = 1.000 across five representations. Same shape as F2 (coverage
predicts detectability, ρ = 1.000, n = 5), reached independently. Both are
n = 5 and both need a second dataset before either is a law rather than a
suggestion, but they are now two of them.

### The combined arm shows the gap being DILUTED

`combined` = sbi (16 ch) ⊕ spectral (95 ch), per-channel standardized on train
only, built by `scripts/build_combined_features.py` (the config existed since
Aug 4; `build_extractor` never had a branch for it, and it did not need one —
the two arms are stored over the same crops in the same order, so the arm is a
concatenation, not a re-extraction).

It does **not** beat its better source: 0.6828 against sbi's 0.7884. And the
reason is visible in the gap, which falls from 36.9% to 5.5%. 6,080 of the
7,104 dimensions come from `spectral`, which has essentially no gap of its own,
so the informative subspace is swamped by uninformative coordinates.

That is **H1 dilution** — the first failure mode this project ever identified,
back on Aug 4, where restricting the exact marginal to the top-k discriminative
coordinates bought +0.093 — reappearing in a completely different quantity. The
dimension gap dilutes exactly the way the likelihood did. Concatenating a
representation that carries no signal costs you, and the cost is measurable in
advance from real data alone.

**The gain tracks the gap.** This is the control the method needs: mass is not a
generic statistic that inflates AUC wherever it is applied. On CLIP features
real and forged faces have essentially the same local dimension (499.3 vs 492.5)
and mass buys nothing — 0.5243, chance. The improvement appears only where there
is a thin sheet to find, and its size is ordered by how thin the sheet is.

**And the gap is a property of the REPRESENTATION, not of the images.** The same
forgeries, the same crops, four feature spaces: they collapse onto a
low-dimensional sheet only in the space that was built to expose blending
artifacts. This is Finding "the representation is everything" (Aug 4) arriving
from a completely different direction — and it is a much sharper version of it,
because "how much lower-dimensional are forgeries here" is a single number
computed from real data alone, with no forgery labels.

Note what it does *not* rescue: the spectral (Corvi) arm stays at 0.54. That is
consistent with Corvi et al.'s own caveat — FF++ c23 is H.264-compressed and our
crops add JPEG, and they state that under strong compression "the compression
artifacts dominate the scene and hide completely the generation artifacts." No
fingerprint survives, so there is no dimension gap to find either. The right
test of that mechanism is still fully synthetic images stored losslessly.

### What it does not do, stated plainly

**It does not beat the project's existing scores** — 0.7884 against 0.8125 for
the one-class NLL family and 0.8283 for the log-ratio. As a *detector* this is
not yet the best thing here.

The likely reason is structural and is the obvious next experiment. The mass
score is **global**, over the whole 1024-d joint. The 0.8125 comes from
**per-patch conditional** scores with per-position calibration. The two ideas are
orthogonal: nothing stops a per-patch *mass* score, using
`region_log_marginals` with box bounds instead of point evaluations. That should
beat both, and it is the natural follow-up.

### Why this matters more than the number

- It is the first result where the circuit is **required** rather than
  permitted. A full-covariance GMM — the model that ties the circuit at 0.830 on
  detection — cannot compute box mass at all: it needs the multivariate normal
  CDF, intractable at d = 1024. Flows and diffusion models cannot integrate
  their own density either. This query is exclusive to tractable circuits.
- It answers Le Lan & Dinh (Entropy 2021, arXiv:2012.03808), the deepest
  objection to the whole approach: a density is not reparametrization-invariant,
  so its ordering carries less information than anomaly detection assumes.
  **Probability mass over a region is invariant.** Scoring by mass removes the
  flaw rather than working around it.
- It converts C1 from a diagnosis into a repair. "Fakes are not outliers" was a
  negative result; "fakes are not outliers *in density*, and mass fixes it"
  is a method.

### Cost, risks and where the floor is

`log_ball` inherits a float32 precision floor: the identity
log P(box) → log p(x) + log vol holds to 3e-4 at ε = 1e-3 and degrades in both
directions — truncation above, cancellation in log(Φ(hi) − Φ(lo)) below.
Measured error at ε = 1e-1 … 1e-6: 4.4e-1 / 4.7e-3 / 2.8e-4 / 1.8e-3 / 2.2e-2 /
5.3e-1 (`tests/test_mass.py`, 5 new tests; suite now 28 passed). **Do not use
ε below 1e-3 in float32.**

The risk that the Gaussian-mixture leaves would be too smooth to resolve the
sheet is **real but survivable**: below ε ≈ 1 the mass score is a monotone
function of density and ranks identically. The signal lives at ε ∈ [0.1, 10],
and the sweep is what makes that visible instead of assumed.

---

# Status — 2026-08-21

Three things landed today: the F1 validation that `hands_off.md` §5 ranked
first, two ablation rows that had actually finished on Aug 6 after the hand-off
was written and sat unread for two weeks, and the `hires` runs that close the
last open recipe hypothesis.

**The headline: F1 does not generalise (it is our bug), `hires` fails by
−0.110, and every recipe-level explanation for the encoder gap is now
exhausted.** Detection parity is formally off the table; the paper is a
diagnosis.

## F1 does NOT generalise to the official SBI — it is our bug

`scripts/official_sbi_symmetry.py`, `results/official_sbi_symmetry.json`.
Checked against Shiohara & Yamasaki's released code
(`github.com/mapooon/SelfBlendedImages`), by measurement, not by reading:

| test | result |
|---|---|
| A — does `additional_targets` share sampled parameters? | **yes**, max diff 0 over 200 trials |
| B — periphery diff after `dynamic_blend` | **0** (bit-identical) |
| B — periphery diff after the shared transforms | 50, but see C |
| C — that difference, beyond 8 px from the mask | **0**, zero pixels changed |

The official recipe is periphery-symmetric by construction, in two places we
diverged from:

1. **The background.** `dynamic_blend` computes
   `mask*source + (1-mask)*target`, and the array SBI returns as the *real* is
   `target` itself. Both branches of its `p=0.5` coin perturb `img` **before**
   blending and return that same perturbed `img` as the real, so real and fake
   share a bit-identical background. Ours composites onto `tgt`, a separately
   perturbed copy, while training compares against the raw `img`.
2. **The compression.** Official SBI has no post-blend re-encode at all. Its
   `alb.ImageCompression(40,100,p=0.5)` sits in the shared
   `Compose(additional_targets={'image1':'image'})` and therefore hits fake and
   real at the *same* quality. Ours applies `match_source_pipeline` (q88–96) to
   the blend only.

The residual 50 is ordinary JPEG block-locality at the blend boundary and dies
within one 8×8 block; `periphery_blocks()` dilates by 24 px, so the audit
discards it already. **The official pipeline would score the null, 0.500, in
`shortcut_audit.py --tests T1`.**

**Consequence: F1 is demoted from a field-wide leakage audit to a
reproducibility note about our `self_blend`.** It is still true, still explains
our own saturation, and is still worth a paragraph — but it is not a claim about
SBI, FSBI, BlenD or anything else, and the paper must not present it as one.
This was the cheap test that decided how big the result is, and it decided
*small*.

## The two ablation rows that were already on disk

`queue_ablation2.sh` completed at 19:08 on Aug 6, minutes after the hand-off was
written. Full table (`scripts/show_ablation.py`):

| variant | val AUC | vs base | loss |
|---|---|---|---|
| sam | 0.8747 | +0.0030 | 0.0120 |
| base | 0.8716 | — | 0.0155 |
| hull | 0.8610 | −0.0106 | 0.0267 |
| all | 0.8543 | −0.0173 | 0.0306 |
| **all_clean** | **0.8333** | **−0.0384** | 0.0243 |
| noleak | 0.8468 | −0.0249 | 0.0197 |
| **noleak_clean** | **0.8081** | **−0.0636** | 0.0173 |

Two corrections to Finding 3 follow.

**(a) The convergence confound is largely dead.** Finding 3 says "val AUC falls
monotonically with final training loss" and could not separate "harder
pseudo-task transfers worse" from "less converged at 20 epochs". Across all
seven variants that rank correlation drops from **ρ = −0.70 to ρ = −0.36**:
`noleak_clean` has the third-lowest loss and the *worst* AUC, `hull` the
second-highest loss and the third-best. Convergence does not explain the
ordering. F3 gets stronger.

**(b) `noleak_clean` is the largest drop in the ablation, and it is unexplained.**
Removing the leak cleanly costs **6.4 AUC points on real forgeries** — but the
leak is chance (0.477) on real forgeries, so deleting it should have cost
nothing there. It is not a compression-exposure artefact either: `_augment`
still applies JPEG q40–100 at p=0.3 to both classes regardless of
`compress_policy`, so `noleak_clean` sees compression variation. **A reviewer
will find this in five seconds; F1 cannot be written up until it has an
answer.**

Note what (b) does to the encoder story: `noleak_clean` is the variant closest
to the official recipe on the periphery axis, and it is our *worst* result at
0.8081 against published SBI's ~0.99. So the leak was never what separated us
from published SBI — removing it makes us worse. **The encoder gap has a cause
we have not identified yet.**

## Finding 8: high-resolution crops fail too — the last recipe hypothesis is closed

`hires` was the one input-side variable never tested: every crop the encoder had
ever seen was stored at 256px JPEG q95 4:2:0 and upsampled to 380.
`crops_hires` is the same faces, same detector, same margin, at native 380px
q100 4:4:4. Run twice, because the first run was confounded.

| variant | val AUC | vs base | loss |
|---|---|---|---|
| base | 0.8716 | — | 0.0155 |
| sam | 0.8747 | +0.0030 | 0.0120 |
| **hires** | **0.7612** | **−0.1104** | 0.0183 |
| **hires_sam** | **0.8368** | **−0.0348** | 0.0095 |
| *hires_partialdata* | *0.7683* | *−0.1033* | *0.0164* |
| *hires_sam_partialdata* | *0.8136* | *−0.0580* | *0.0125* |

**Higher-fidelity input makes the encoder worse, decisively.** Two confounds
were checked and both eliminated before this was believed:

1. **Leak amplification — refuted by measurement.** `crops_hires` is stored q100
   4:4:4 while `match_source_pipeline` re-encodes the blend at q88–96 with
   cv2's default 4:2:0, so the fake differs from the real by a chroma-subsampling
   change *plus* a quality drop — a bigger asymmetry than base. If that were the
   cause the periphery leak would be much larger. It is not:
   `results/shortcut_audit_crops_hires.json` gives **0.9486** against base's
   0.937, with `pristine_bg_no_reencode` at exactly **0.5000** on both. The
   F1 null reproduces on an independent crop set, which is a free check on the
   audit's machinery.
2. **Training-set size — eliminated by re-running.** The first `hires` run used
   588 of 706 real videos (9,550 crops vs base's 11,187, −14.6%): the Aug-6
   hires ingest was killed after writing 604 directories and `hands_off.md`
   recorded train as "complete" when it was not. The train ingest was finished
   (`706/720 videos, 1.7 min`) and both runs now draw on **exactly 11,187
   crops**. The verdict did not move.

### The re-run also gives a noise floor, for free

`hires` and `hires_sam` have each now been trained twice, differing only by
14.6% of the training data:

| variant | partial data | full data | Δ |
|---|---|---|---|
| hires | 0.7683 | 0.7612 | −0.0071 |
| hires_sam | 0.8136 | 0.8368 | **+0.0232** |

The two move in *opposite* directions, so the data difference is not systematic
and what is left is run-to-run variance of roughly **±0.02 val AUC** at this
protocol. That is a retroactive correction to Finding 3, and it is the cheap
version of the 3-seed check `hands_off.md` §5 item 5 asked for:

- **`sam` +0.0030 and `hull` −0.0106 are inside the noise band.** "SAM buys
  noise" is confirmed; "hull hurts" is not supported and must not be stated.
- **Only the large effects survive**: `noleak_clean` −0.0636, `all_clean`
  −0.0384, `hires` −0.1104.
- Any ablation number quoted in the paper below ~0.02 needs real seeds behind
  it or has to go.

### What it closes

`hands_off.md` §5 item 2: *"`hires` is the only untested cause of the encoder
gap. If it also fails, stop trying to reach detection parity."* **It failed.**
Every recipe-level axis — optimizer, leak, hull geometry, input resolution — has
now been tested and none recovers the encoder gap. Finding 0 is complete and the
decision it was gating is made: **the paper is purely diagnostic, and detection
parity is not a goal.**

The shape of the failure is worth one line, because it is the same shape as
everything else here: `hires_sam` reaches the *lowest training loss in the whole
ablation* (0.0095) and still lands 0.035 below base. The pseudo-task gets easier
and transfer gets worse. That is Finding 6 again, from a fourth direction.

## Where that points

Reading the official code to run the F1 test also produced a list of concrete,
still-unported deviations, which is the first new lead on the encoder gap since
Finding 0:

- **Blend alpha.** Official draws from `[0.25,0.5,0.75,1,1,1]` — full strength
  half the time. Our `deform_mask` ends with `mask * rng.uniform(0.25,1.0)`,
  which is full strength essentially never.
- **Paired batches.** Official returns `(img_f, img_r)` for *every* item, so
  each batch is exactly balanced and each fake sits next to its own source.
  Ours flips a per-sample coin and returns one or the other.
- **Colour augmentation.** Official's shared compose has RGBShift,
  HueSaturationValue and RandomBrightnessContrast. Ours has hflip, JPEG and
  downscale only.
- **Elastic on the source.** Official's `randaffine` applies
  `alb.ElasticTransform(alpha=50,sigma=7)` to the donor *image*; our
  `_random_affine` warps the donor affinely and the elastic term is applied to
  the *mask* instead.

A faithful port is bounded work and is now the ONLY untested lead, `hires`
having since failed (Finding 8). It is not a reason to reopen the
detection-parity goal; it is the reason the paper can say *why* our
reimplementation underperforms rather than leaving it as "weaker in ways not
captured".

## Also true as of today

- `crops_hires` is now complete: train reals 706/720 and val all-six-methods
  811/840, both at the config's `n_frames_test=32` so the frames match
  `crops/` exactly. Note that training never reads
  `ffpp_ingested_crops_hires.csv` at all — `collect_real_items` and
  `collect_labeled_items` both hardcode `ffpp_ingested.csv` and use
  `crops_dirname` only to redirect the directory lookup, silently skipping
  any directory that is missing. That silent skip is what hid the 14.6%
  shortfall in the first `hires` run.
- Disk on the workstation is now **101 GB free**. `raw/ffpp_zip` (17 GB) is
  still redundant.
- Every result JSON is now committed under `results/` (29 files, 5.4 MB).
  They previously existed only on the workstation.

---

# Status — 2026-08-06

# WHY THIS IS NOT A STATE-OF-THE-ART DETECTOR — the investigation

## Finding 0: the encoder plateau is real, and it kills the compute excuse

The 100-epoch SBI run was killed at epoch 48 by the Aug-5 08:14 reboot (the same
reboot that cleared the driver blocker). Its checkpoint survived:
**best val video AUC 0.8722, reached at epoch 14.** Thirty-four further epochs
never beat it, oscillating 0.83–0.86. Against the 40-epoch run's 0.8653 that is
**+0.007 for 2.5× the compute**.

STATUS previously argued the encoder was weak "by construction — 40 epochs,
16 frames/video … so most of that shortfall is compute, not method". That
hypothesis is now falsified. Reproduced again on 2026-08-06 at 16 frames/video,
20 epochs: **0.8716**, i.e. the same ceiling from a quarter of the budget.

## Finding 1: the gap is 98% upstream of the circuit — now measured

The `0.860` encoder figure that appeared in this file since Aug 4 had **no
artefact behind it** — no log, no result JSON produced it. `scripts/gap_waterfall.py`
measures every stage on exactly the same crops (it reads the `path` field of the
feature index, so the encoder is scored on the images whose projected features
the probe and the circuit see). Measured:

| stage | FF++ video AUC | lost here |
|---|---|---|
| published SBI on FF++ c23 (reported) | 0.9964 | — |
| **our encoder, end to end** | **0.8312** | **+0.1652** |
| linear probe on projected patch features (1024 d) | 0.8591 | −0.0280 |
| circuit, one-class NLL | 0.8125 | +0.0467 |
| circuit, exact log-ratio | 0.8283 | −0.0158 |
| | **total 0.1681** | **encoder = 98%** |

The real encoder number is **0.8312**, not 0.860 — and its val AUC was 0.8722,
so there is a 0.041 val→test drop that model selection on 4 frames/video was
hiding. Two consequences:

* **The projection is not a bottleneck — it is not even lossy.** The probe on
  16 channels × 64 patches reaches 0.8591, *above* the encoder's own 0.8312.
  (The probe is supervised on val real forgeries, so it is an upper bound on
  extractable signal rather than an information-preservation proof — but it does
  establish both that PCA-16 discards nothing usable and that the encoder's own
  head leaves 0.028 on the table.) `sota_push.sh` was queued to sweep
  `grid=12, out_dim=32` on the theory that squeezing 1792 dims into 16 was
  costing us. **That sweep is cancelled.**
* Everything this project is about — density estimation, exact inference,
  structure learning — is competing over the last 0.047.

## Finding 2: the pseudo-task leaks a global cue that real forgeries do not have

This is the substantive result, and it is a statement about the SBI *generator*,
not about our training.

`scripts/shortcut_audit.py` classifies real vs self-blend using **only 8×8 JPEG
blocks lying entirely outside the (generously dilated) blending mask** — pixels
the forgery never touched — from DCT coefficient histograms and blockiness
alone. Splits are grouped by image so a no-signal condition reads as exactly
0.5.

| self-blend recipe | leak AUC outside the mask |
|---|---|
| **what this project has been training on** | **0.937** |
| drop the post-blend JPEG re-encode | 0.812 |
| composite onto the unperturbed image (pristine background) | 0.917 |
| **pristine background + no re-encode** | **0.500** |
| **pristine background + symmetric re-encode** | **0.500** |

Two independent mechanisms, neither sufficient alone, jointly worth 0.44 AUC:

1. `self_blend` ends with `match_source_pipeline`, a JPEG q88–96 re-encode
   applied to the blend and **not** to the real image it is contrasted against.
2. `source_target_pair` perturbs *both* copies and swaps them with p=0.5, so
   about half the time the entire context — not the donor — is the
   resolution-jittered, re-compressed copy. The pseudo-fake is globally marked.

And the cue does not transfer. The identical features on **real FF++ forgeries**
vs real faces:

| | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures | pooled |
|---|---|---|---|---|---|---|
| periphery AUC | 0.491 | 0.480 | 0.528 | 0.474 | 0.512 | **0.477** |

Chance. In a real swap the context *is* the original video frame, so no such
signature exists — which is exactly what the pristine-background fix restores.

**This explains the saturation directly.** The training loss reaches 0.008 and
real-vs-blend AUC 0.9996 because ~94% of the pseudo-task is decidable without
looking at the face at all; the network is not required to learn forgery
evidence, so it does not, and 0.86 on real forgeries is what is left.

Verified rather than assumed: for the pristine + no-reencode recipe the
periphery pixels are **bit-identical** to the real crop (max difference 0 over
40 images, `scripts/_check_periphery.py`), so 0.500 is a true null and not a
weak classifier.

### How this differs from the published position

"The Alpha Blending Hypothesis" (arXiv 2605.10334) argues detectors act as
*alpha blending boundary searchers* — its experiments manipulate boundary
hardness and photometric mismatch, and it explicitly treats SBI as a generic
blending heuristic rather than dissecting the generator. Our measurement is
disjoint from that: the signal we find is **not at the boundary and not in the
manipulated region at all**, and it comes from implementation choices in the
pseudo-fake pipeline. If it generalises to the reference SBI implementation, it
affects the whole SBI-derived line (SBI, FSBI, BlenD, …) as a *leakage audit
that any pseudo-fake generator should have to pass*.

## Finding 3: no recipe-level fix recovers the encoder gap

Five controlled retrains, 20 epochs, 16 frames/video, identical protocol:

| variant | val AUC | vs base | train loss |
|---|---|---|---|
| `sam` (SAM instead of AdamW) | 0.8747 | **+0.0030** | 0.0120 |
| `base` (current recipe) | 0.8716 | — | 0.0155 |
| `hull` (randomised hull type) | 0.8610 | −0.0106 | 0.0267 |
| `all` | 0.8543 | −0.0173 | 0.0306 |
| `noleak` (pristine bg + symmetric re-encode) | 0.8468 | −0.0249 | 0.0197 |

**Every hypothesis fails.** SAM — the difference from published SBI that looked
most likely to matter — buys +0.003, which is noise. Everything else hurts.

Note the pattern: val AUC falls monotonically with final training loss. Two
readings, and we cannot yet separate them: either harder pseudo-tasks genuinely
transfer worse, or they are simply less converged at a fixed 20-epoch budget.
The second is the more likely and the more boring, and it is the reason these
are ranked rather than treated as final.

`noleak` additionally carries a confound of our own making:
`compress_policy=symmetric` re-encodes every *training* image while val and test
stay untouched. `noleak_clean` (pristine background + **no** re-encode, equally
leak-free at 0.500) removes that side effect and is queued.

## Finding 4: coverage under the pseudo-fake density predicts detectability

This is the strongest positive result of the investigation, and it is the kind of
statement only an exactly-normalized density can make.

For each manipulation, how far do real forgeries sit from the pseudo-fakes the
model was actually fitted on? With exact densities that is measurable in nats:

| manipulation | coverage under `p_mix` | mean log-ratio | detection AUC |
|---|---|---|---|
| Deepfakes | 0.733 | −1000 | 0.921 |
| Face2Face | 0.563 | −1564 | 0.872 |
| FaceShifter | 0.506 | −1846 | 0.844 |
| NeuralTextures | 0.456 | −2300 | 0.808 |
| FaceSwap | 0.297 | −3845 | 0.699 |
| *(our own pseudo-fakes)* | 0.950 | +1427 | 1.000 |
| *(real test faces)* | 0.132 | −7617 | — |

**Spearman ρ = 1.000, Pearson r = 0.998** between mean log-ratio and
per-manipulation AUC. The ordering is not approximately right, it is exactly
right across all five manipulations.

So the domain gap is not a hand-wave — it is a measurable quantity that
*predicts* per-manipulation performance. (n = 5, so this is a strong
suggestion, not an established law; it needs Celeb-DF-v2 and DF40 to become a
claim.) It also reframes contribution C1: the diagnosis is no longer only
"likelihood is the wrong statistic" but "**detectability is a function of
pseudo-fake coverage, and coverage is computable**".

## Finding 5: the mechanism mixture buys nothing (honest negative)

`p_mix = Σ_f π_f p_f` over four forgery families, one circuit each, shared region
graph, everything exactly normalized (verified: log Z ≈ 0 for all five circuits,
`tests/test_family_mixture.py`).

| scorer | FF++ video AUC |
|---|---|
| mixture over 4 families | 0.8286 |
| `overshoot` alone | 0.8306 |
| `render` alone | 0.8300 |
| `blend` alone | 0.8285 |
| `statistical` alone | 0.8274 |

The mixture equals its best component to four decimals, and every component —
including families designed to deviate in *opposite* directions (rougher vs
smoother than real) — produces a near-identical per-method breakdown. At
log-ratios of thousands of nats one component dominates the logsumexp entirely,
so the mixture degenerates to a max.

The **exact family posterior** `P(f | z)` is likewise uninformative: real faces
get `blend` 0.599 and every manipulation gets `blend` 0.49–0.66. The hoped-for
result (Face2Face → `render`) does not appear, at the image level or per region.

Why: the four families are distinguishable in pixels but not after an encoder
that was itself trained to make self-blends maximally separable from real. The
representation collapses the mechanisms it was never asked to distinguish.

## Finding 6: the saturation is caused by the encoder, not the leak

A falsifiable prediction was made and **failed**. The hybrid λ sweep was flat
because the pseudo-task was saturated (BCE 0.0000, real-vs-blend 0.9996); if
~94% of that separability was the compression leak, leak-free blends should
restore gradient. Rerun on `blend-blendP`:

| λ | FF++ AUC | localization patch AUC | real-vs-blend |
|---|---|---|---|
| 1.0 | 0.8250 | 0.678 | **1.0000** |
| 0.3 | 0.8268 | 0.673 | **1.0000** |
| 0.0 | 0.8132 | 0.462 | **1.0000** |

Real-vs-blend is 1.0000 — *more* saturated, not less. Removing the leak does not
make self-blends harder to tell from real faces **in the SBI feature space**,
because that space was built by training on exactly this discrimination. The
representation and the pseudo-fakes are the same construction, so the density
task on top is degenerate by design.

(λ=0 still destroys localization, 0.678 → 0.462 — the earlier finding that a
purely discriminative fit costs the density semantics reproduces.)

## Finding 7: C5 (calibration) is not supported

PLAN.md claims the ratio is a log-odds, so "thresholds transfer, abstention is
principled". Measured on FF++:

| | ECE | MCE |
|---|---|---|
| raw `sigmoid(s)` | 0.780 | 0.916 |
| after temperature scaling (T = 110, fitted on val reals + own pseudo-fakes) | 0.766 | 0.890 |

Risk–coverage barely moves: 0.780 at full coverage → 0.588 at 24%. The ranking
is fine (AUC 0.83) but the probabilities are meaningless — scores run to −7617
nats on real faces against +1427 on pseudo-fakes, so `sigmoid` saturates and no
single temperature repairs it.

This fails for the *same* reason as everything else: both densities are
badly misspecified for real forgeries. **C5 should be dropped from the
contribution list, or rewritten as a negative result** — "an exact log-ratio is
a calibrated log-odds only under the pseudo-fake distribution it was fitted on,
and that is precisely what does not transfer".

## What is left, and the one untested hypothesis

Everything recipe-level has now been tested and failed. What has **never** been
varied is the input: every crop the encoder has seen was stored at 256px JPEG
q95 with 4:2:0 chroma subsampling and then upsampled to 380 — two lossy steps
attacking exactly the high-frequency and colour detail a blending artifact lives
in, applied before the network sees anything. `crops_hires` (same faces, same
detector, same margin, native 380px q100 4:4:4) is ingested and the `hires` /
`hires_sam` variants are queued (`scripts/queue_hires.sh`).

If `hires` does not move it either, the conclusion is that our encoder is simply
a weaker SBI reimplementation in ways not captured by these four axes, and the
paper should stop treating detection parity as reachable and lean entirely on
the diagnosis (Findings 1, 4, 6) — which is where the defensible contribution
now clearly sits.

## In flight

* `scripts/queue_ablation2.sh` → `noleak_clean`, `all_clean`.
* `scripts/queue_hires.sh` → `hires`, `hires_sam`.

---

# Status — 2026-08-04

## P0 RESULTS — the circuit finally wins something no one else can do

### P0.1 Localization by per-patch RATIO — **WIN, +0.066 over PatchCore**

Score = `log p_blend(z_p | z_−p) − log p_real(z_p | z_−p)`, exact, per patch.
It inherits the ratio's nuisance cancellation AND the circuit's exact
conditioning on context — a combination no competitor can form.

| model | patch AUC | per-image | IoU | pointing |
|---|---|---|---|---|
| **PC ratio (conditional)** | **0.7366** | **0.7309** | **0.3178** | 0.3128 |
| PatchCore | 0.6711 | 0.6721 | 0.2885 | **0.4392** |
| GMM | 0.5424 | 0.5305 | 0.1843 | 0.2046 |
| Mahalanobis | 0.5404 | 0.5145 | 0.1841 | 0.0814 |
| flow | 0.5113 | 0.5066 | 0.1660 | 0.1422 |

Clears the pre-registered 0.03 localization gate. Every manipulation above
chance (Deepfakes 0.841, Face2Face 0.722, FaceShifter 0.722, FaceSwap 0.692,
NeuralTextures 0.639). PatchCore keeps a sharper single peak (pointing 0.44);
the circuit wins on region overlap, which is what a forensic user needs.

### P0.2 Occlusion robustness — **no advantage** (honest negative)

Exact marginalization 0.8297 → 0.8287 as 0-50% of patches are hidden; a GMM
with mean imputation 0.8318 → 0.8301. Both flat: the image score averages over
patches, so an imputed patch contributes a benign near-average value and the
circuit's exactness never gets to matter. A max-aggregated score might separate
them; as measured, this is not a selling point.

### P0.3 Region-wise discriminative information — **works, and is unique**

`D_R = E_real[log p_real(z_R) − log p_blend(z_R)]` per patch, both marginals
exact (the region is a node of the shared graph), expectation on held-out
reals. Top regions: patches 36, 37, 35 (row 4, cols 3-5), 28, 19, 20, 27 —
**the centre of the face**, 115-159 nats. The model identified where swaps
manipulate without ever being told where a face is. No baseline can produce a
model-level explanation at all.

## Hybrid discriminative training — flat, and the reason matters

| λ | FF++ AUC | localization | real-vs-blend |
|---|---|---|---|
| 1.0 (generative) | 0.8274 | 0.7500 | 0.9996 |
| 0.3 | 0.8274 | 0.7559 | 0.9996 |
| 0.1 | 0.8269 | **0.7600** | 0.9996 |
| 0.0 (discriminative) | 0.8214 | 0.7191 | 0.9992 |

Detection does not move. The diagnostic is in the last column: at λ=0 the
discriminative loss reaches **0.0000** and real-vs-blend AUC is **0.9996** —
the pseudo-task is SATURATED. There is no gradient left because the model
already separates our blends perfectly; what it cannot do is generalise to real
forgeries.

**Conclusion: the bottleneck is neither the model, the objective, nor the
scoring rule — it is the pseudo-fake distribution.** Same conclusion as "The
Alpha Blending Hypothesis" (2026), reached independently from our own numbers.
Consequently the next levers are (a) the encoder, (b) the projection ceiling
(probe caps at 0.859 because we keep 16 of 1792 dims and pool 12×12 to 8×8),
and (c) pseudo-fake DIVERSITY — now implemented as four forgery families
(`blend` / `render` / `overshoot` / `statistical`) covering both the
rougher-than-real and smoother-than-real directions, which a circuit can hold
natively as a mixture.


## RESULT SUMMARY — the representation is everything

Four representations, identical circuit, identical protocol (real faces only,
official identity-disjoint FF++ splits), identical baselines:

| arm | PC video AUC | supervised probe on the SAME features | gap |
|---|---|---|---|
| SRM (hand-built, radial spectrum) | 0.624 | — | — |
| CLIP ViT-L/14 patch tokens | 0.536 | 0.775 | **0.239** |
| **SBI-shaped encoder** | **0.812** | 0.859 | **0.047** |
| spectral residual (Corvi et al.) | 0.554 | 0.802 | **0.248** |

**The probe-minus-one-class gap is the headline.** It measures how much of the
extractable signal density estimation actually recovers. Shaping the
representation with self-blends closes it from 0.24 to 0.047: in an
artifact-tuned space, one-class density detection recovers nearly everything
supervision can get from the same coordinates. That is the transfer working —
but only once the representation makes forgery *atypical*.

SBI arm, all models in that feature space:

| model | FF++ video AUC |
|---|---|
| SBI classifier (supervised, real-only protocol) | 0.860 |
| **PC (patch_cond_lowmax)** | **0.812** |
| RealNVP flow | 0.804 |
| GMM-full | 0.788 |
| PatchCore | 0.461 |
| Mahalanobis | 0.386 |

The circuit leads every one-class competitor, but by +0.009 over the flow —
under the 0.02 gate. G1 fails at 0.812 < 0.90, so the rubric still says STOP;
the encoder is however weak by construction (val AUC 0.865 where published SBI
reaches ~0.99: 40 epochs, 16 frames/video, AdamW instead of SAM), so most of
that shortfall is compute, not method.

**Localization loses to PatchCore** — PC 0.568 pooled patch AUC / 0.218
pointing vs PatchCore 0.670 / 0.438. The exact-marginal machinery does not
currently buy better localization ACCURACY; its claim has to rest on exactness
and probabilistic semantics, not on beating a memory bank.

Note which score keeps winning in every arm: `patch_cond_lowmax`, the
*two-sided* branch. Forgeries are consistently found in HIGH-density regions,
even in the SBI space.

### Why the spectral (Corvi et al.) arm did not rescue it

Its features are good — probe 0.802, and unusually UNIFORM across manipulations
(0.73-0.84 per method, versus CLIP's 0.59-0.92), which is the cross-generator
robustness that paper argues for, obtained with no learning at all. But the
one-class gap stays at 0.248, and there are two structural reasons it should:

* Corvi et al. analyse **fully synthetic** images, where the entire frame comes
  from a generator and therefore lacks a sensor-noise floor. An FF++ face swap
  re-renders a REAL face into a REAL video: both classes carry camera noise, so
  the "missing noise floor" signature largely does not apply.
* FF++ c23 is H.264-compressed and our crops add JPEG q95 — the paper itself
  warns that compression and resampling destroy these fingerprints. Testing the
  hypothesis on its home ground needs fully synthetic images (GenImage / DF40
  diffusion subsets) stored losslessly, not re-encoded video.

So this is not evidence against the paper; it is evidence that FF++ c23 is the
wrong dataset for its mechanism, and that the failure is once again the
typicality assumption, not the features.

## DIAGNOSIS: the score was wrong, not the circuit (2026-08-04, late)

The "everything is at chance" result was challenged rather than accepted, and
the challenge was right. `pcdf/eval/diagnose.py` runs the FITTED circuit through
three competing explanations on the spectral arm (d = 6080):

| hypothesis | test | outcome |
|---|---|---|
| H1 dilution | exact marginal restricted to the top-k discriminative coordinates (selected on VAL) | **CONFIRMED: +0.093 AUC** over the full joint (0.461 → 0.554) |
| H2 inversion | fraction of forgeries above the real median likelihood | **CONFIRMED: 55.4%** — fakes are MORE likely |
| H3 estimation | circuit val NLL vs full-covariance Gaussian | **RULED OUT** — the Gaussian is singular at d = 6080; the circuit fits fine |

So the density is good and the features carry signal (probe 0.802); what fails
is using `log p(z)` as the statistic. It sums surprisal over 6080 coordinates,
so evidence carried by a few dozen is outvoted, and the residual direction
points the wrong way anyway. This is the textbook failure of likelihood-based
OOD detection (Ren et al., NeurIPS 2019): `p(x)` is dominated by background
statistics that both classes share.

### The fix: an exact likelihood RATIO of two circuits

`pcdf/models/ratio.py` — two circuits over ONE shared region graph:

    p_real    fitted on real training faces
    p_blend   fitted on SELF-BLENDS of those same faces (SBI protocol,
              no real forgery is ever seen)
    score     s(z) = log p_blend(z) − log p_real(z)

Both are exactly normalized, so this is a true log-likelihood ratio — a
generative classifier, Bayes-optimal when both densities are right — and the
per-patch version

    s_p(z) = log p_blend(z_p | z_−p) − log p_real(z_p | z_−p)

is exact as well: a localization map whose values mean "this region is k nats
better explained by blending than by the camera". Neither a flow (no marginals)
nor a memory bank (no probabilities) can produce that. This is the first
construction in the project where the circuit's exactness is load-bearing for
DETECTION rather than only for explanation.

## THE RATIO RESULT, AND THE FAIRNESS CHECK THAT QUALIFIES IT

The two-circuit ratio was validated, then tested, then challenged:

1. **Construction verified.** On real vs the SELF-BLENDS it was trained on
   (held-out val, spectral features), the exact log-ratio reaches **0.953**
   from two densities that alone score 0.585 and 0.643. The machinery is
   sound — a circuit does detect a clear distribution shift, decisively.
2. **On FF++ spectral features it fails (0.554→0.549)** and the per-method
   pattern says why: the neural manipulations are detected (Deepfakes 0.68,
   FaceShifter 0.67, NeuralTextures 0.59) while the GRAPHICS-based ones are
   INVERTED (Face2Face 0.45, FaceSwap 0.36). Rendered faces are *smoother*
   than real; our self-blends are *rougher*. The pseudo-fakes deviate in the
   opposite direction from half the real forgeries. Matching the compression
   history (JPEG re-encode after blending) did not fix it.
3. **On SBI features it works: 0.828**, every method above chance
   (Deepfakes 0.920, Face2Face 0.870, FaceShifter 0.842, NeuralTextures 0.808,
   FaceSwap 0.702) — because that encoder was trained end to end to make
   self-blends and real forgeries look alike, so the domain gap is small.
4. **But the gain is the RATIO, not the circuit.** Given the same two-density
   construction:

   | model | one-class | with ratio |
   |---|---|---|
   | RealNVP flow | 0.200 | 0.828 |
   | **GMM-full** | 0.225 | **0.830** |
   | Mahalanobis | 0.286 | 0.814 |
   | **PC (circuit)** | 0.812 | **0.828** |

   A GMM with the ratio matches — marginally beats — the circuit. G3
   (circuit beats the best non-circuit baseline by ≥0.02) **FAILS** under the
   fair comparison. The circuit's per-patch CONDITIONAL ratio (0.822-0.826)
   also did not beat its own joint ratio (0.828), so conditioning on context
   bought nothing here either.

The honest summary: the project discovered a real and transferable result —
**likelihood-ratio scoring against a self-blend density rescues every density
model on this task, converting a 0.20-0.29 inverted score into 0.81-0.83** —
and that result is model-agnostic. It is not evidence for probabilistic
circuits specifically.

## What this means for "is it worth pursuing"

Three things are established and would survive review:

1. **The exact-inference engineering works.** Equivalence-tested against the
   reference library, 44.5× faster, scales to d = 7104 on one 16 GB card, log Z
   exact to 1e-6 after training, all four circuit properties audited per run.
2. **Structure learning works, and curvature beats Chow-Liu decisively** —
   Forman −169 nats, ORC −46, Chow-Liu −5 versus random, reproduced on three
   different representations. This is a result about probabilistic circuits
   that does not depend on deepfakes at all.
3. **Detection does not follow likelihood.** Across four representations, large
   NLL improvements produced no AUC improvement, and the winning score was
   always the "too typical" branch. Density-based one-class detection fails on
   deepfakes unless the representation is discriminatively shaped, because a
   forgery is not an outlier — it is an ordinary-looking face.

The honest framing is therefore a **methodological/negative contribution with a
precise diagnostic** (the probe-minus-one-class gap), plus a genuine systems
contribution (tensorized exact circuits) and a structure-learning result — not
a state-of-the-art detection paper. Whether that is worth pursuing is a
judgement call about venue and appetite, and it is the user's to make; the
evidence is now in `results/*/REPORT.md` with a pre-registered rubric attached.

The one open lever on the detection axis is the SBI encoder: ours reached val
0.865 against ~0.99 published, and the arm built on it is the only one that
closed the gap. A properly trained encoder (100 epochs, 32 frames/video) is
running to settle whether G1 is reachable at all.

## THE CENTRAL FINDING (CLIP arm + probe)

The CLIP patch-token arm ran end to end on the GPU. It is at chance — and the
diagnostic that follows is the most important result so far, because it says
*where* the failure is:

| model | FF++ video AUC | winning score |
|---|---|---|
| **PC** | **0.536** | `patch_cond_lowmax` |
| RealNVP flow | 0.536 | `patch_max` |
| GMM-full | 0.531 | `patch_lowmax` |
| Mahalanobis | 0.519 | `patch_lowmax` |
| PatchCore | 0.517 | `patch_lowmax` |

**A supervised linear probe on the SAME projected features reaches 0.775 video
AUC** (Deepfakes 0.92, FaceSwap 0.87, FaceShifter 0.77, Face2Face 0.72,
NeuralTextures 0.59) — `pcdf/eval/probe.py`, trained on the val split, never on
test. So the signal is present in exactly the coordinates the circuit sees, and
one-class density scoring recovers almost none of it: a 0.24 AUC gap.

Note *which* score wins for nearly every model: `patch_lowmax`, the two-sided
branch. Fakes sit in **higher**-density regions than reals. This is not a
tuning problem and not a circuit problem — it falsifies, for this
representation, the assumption the whole PCNET transfer rests on:

> anomalies of a generative process live in LOW-density regions of a
> well-chosen representation

For deepfakes in a semantically organised space, a manipulated face is
*more typical* than a real one — generated skin is smoother, more average, more
"face-like" than camera output. Density is the wrong statistic there; the
signal is discriminative, not typicality-based.

### Structure learning, on the other hand, works

Same features, same budget, only the region graph changes:

| structure | val NLL | vs random | detection AUC |
|---|---|---|---|
| random/random | 1255.70 | — | 0.533 |
| kd/chow_liu | 1250.47 | −5.2 | 0.532 |
| chow_liu/orc | 1229.93 | −25.8 | 0.530 |
| orc/orc | 1235.73 | −20.0 | 0.529 |
| kd/orc | 1209.72 | −46.0 | 0.525 |
| **kd/forman** | **1086.26** | **−169.4** | 0.519 |

Curvature-guided structure beats Chow-Liu by a wide margin on held-out
likelihood (Forman −169 nats, ORC −46, Chow-Liu only −5 vs random), so gate G4
passes decisively. And detection does not follow the likelihood at all — which
is the same finding from the other direction: a better density model of real
faces is not a better deepfake detector.

Gates for the CLIP arm: G1 FAIL (0.536), G3 FAIL both halves, **G4 PASS
(169.4 nats)**, G5 PASS (37.7 s per fit). Verdict: STOP for this arm.

### Engineering claim, measured

`pcdf bench --with-reference`: the tensorized circuit is **44.5× faster** than
the reference object-graph implementation (0.0034 vs 0.152 s/step), and on the
GPU a full fit takes 1.2 s/epoch versus 11.7 s/epoch on CPU. Structure ablation
variants dropped from 225 s to ~25 s each.

## First real FF++ result (SRM arm, CPU) — honest negative

The hand-crafted forensic arm (`backbone=srm`, 8×8 patches × 16 whitened dims,
d = 1024, K = 8) ran end to end on all 183,723 crops. **It does not work, and
neither does anything else in that feature space:**

| model                         | FF++ video AUC  | localization patch AUC |
| ----------------------------- | --------------- | ---------------------- |
| **PC (patch_cond_max)** | **0.624** | 0.519                  |
| Mahalanobis                   | 0.628           | 0.563                  |
| PatchCore                     | 0.627           | 0.528                  |
| GMM-full                      | 0.531           | 0.525                  |

Per manipulation the circuit gets FaceSwap 0.69 and Face2Face 0.65, but
Deepfakes 0.45 and FaceShifter 0.34 — *below* chance, i.e. those forgeries are
**more typical** than real faces under the density. That is the same
likelihood-OOD pathology the POC hit, now reproduced on real FF++ data.

Rubric verdict for this arm: **STOP — the detector does not work in-dataset;
fix the representation first.** Which is exactly what this arm is for: SRM is
the no-learned-component control, and its failure isolates the representation
as the deciding variable rather than the density model. Every model being
within 0.005 AUC of every other is the signature of a representation that
carries no usable signal — not of a bad circuit.

The circuit itself is healthy in that run: val NLL 1899 → 715 over 40 epochs,
`log Z = -1.1e-06`, exact-marginal consistency to 1.8e-06, structured
decomposable, 2047 regions / 622k parameters at d = 1024, 11.7 s/epoch on CPU.

**The real tests (CLIP patch tokens, then the SBI-tuned encoder) are blocked on
the GPU.** See the blocker below.

## Device handling (added 2026-08-04)

Nothing hardcodes CUDA any more. `pcdf/device.py` resolves `device: auto` once
in `main()` and every stage inherits the concrete string:

* CUDA is accepted only after a real allocation succeeds — `is_available()`
  returned True throughout the driver breakage below while every allocation
  failed, so the flag alone is not evidence;
* an EXPLICIT `device=cuda` that cannot run raises instead of downgrading
  silently (a six-hour run at CPU speed is worse than an error);
* MPS is now fully supported: `torch.special.log_ndtr` is missing there, which
  would have cost the interval/box query, so `pcdf.circuits.leaves.log_ndtr`
  supplies an erfc + asymptotic-tail fallback that matches the native op
  exactly, and MPS agrees with CPU to 1.9e-06 on log_prob, marginals, box
  queries and log Z;
* AMP is selected per device type rather than assuming `"cuda"`;
* fitted baselines pickle their tensors on CPU and move at score time, so a
  model fitted on GPU can be scored on CPU and vice versa.

`tests/test_device.py` pins all of this. Practical effect: the same commands
run unchanged on the Mac (MPS), on the workstation now (CPU fallback), and on
the workstation after the reboot (CUDA) — no `-s device=…` overrides needed.

## BLOCKER: NVIDIA driver mismatch

An unattended `apt` upgrade replaced the driver (580.159.03 → 580.173.02) while
the old kernel module stays loaded; `nvidia-smi` reports *"Driver/library
version mismatch"* and torch raises CUDA error 804. A reboot did not clear it:
DKMS has built 580.173.02 for the running kernel (`modinfo -F version nvidia`
confirms) but the initramfs still carries the old module. Needs sudo:

```bash
sudo update-initramfs -u -k $(uname -r) && sudo reboot
```

Everything below assumes that is done.

# Original status (paused, resumable)

Rebuild of the PC deepfake work on real data and real baselines. `POC.md` and
`hands_off.md` remain the historical record of the LFW/OpenForensics POC; this
file is the current state.

## Where the work stands

| stage                                        | state                                                             |
| -------------------------------------------- | ----------------------------------------------------------------- |
| Tensorized circuit (`pcdf/circuits/`)      | **done, verified** — 9/9 equivalence tests pass            |
| FF++ c23 data on the workstation             | **done** — 17.9 GB, unzipped, official splits fetched      |
| Face ingestion                               | **done** — 183,723 crops from 5,874/6,000 videos (4.5 GB)  |
| CLIP feature extraction                      | **not finished** — killed mid-run, must be rerun (~35 min) |
| PC fit / baselines / eval / explain / report | written,**not yet run**                                     |
| SBI encoder + baseline                       | written,**not yet trained** (~5 h)                          |

## What is verified, not just written

* `EinsumPC` computes **exactly** what the reference `RegionGraphPC` in
  `src/probabilistic_circuits.py` computes — parameters are copied between the
  two and `log p(x)`, exact marginals over random observation masks, box
  queries and `log Z` all agree to 2e-4 across six structure families
  (Chow-Liu, ORC, multi-partition ORC, Forman, spectral, random).
  `tests/test_equivalence.py`, run with `expllm_env`.
* Scaling: d = 1024 (8×8 patches × 16 channels), K = 8, 622k parameters,
  fwd+bwd on a CPU in 0.33 s/step; `log Z = -5.3e-06` after training, i.e.
  normalization survives optimisation.
* Ingestion produces derived localization masks whose manipulated area is
  ~16% of the crop on FF++ Deepfakes — the right order for a face swap.

## Machine

```
ssh jawa17@192.168.1.8          # RTX 4080 16 GB, 62 GB RAM, 24 cores
repo   ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
data   ~/deepfake_data          # NOT in the repo
python ~/miniconda3/envs/expllm_env/bin/python   # torch 2.10+cu128, timm, mediapipe
```

**Disk is the binding constraint: 69 GB free.** The FF++ zip
(`~/deepfake_data/raw/ffpp_zip/FaceForensics++_C23.zip`, 17 GB) is redundant
now that the videos are unzipped and the crops are extracted — deleting it is
the easiest 17 GB back. The raw videos (17 GB) are still needed only if crops
are ever re-extracted at a different margin/size.

## Resume commands (in order)

**Step 0 — re-sync first.** The workstation went offline before the last sync
landed, so it is running code from a few commits' worth of edits ago (it is
missing the robustness perturbations, `ablate-structure`, `train-sbi`, the
report/rubric module and this file):

```bash
# from the Mac
cd ~/Documents/UNITN/PHD/MAIN_Project/DeepFakeDet
rsync -az --exclude __pycache__ --exclude .DS_Store pcdf tests configs STATUS.md \
  jawa17@192.168.1.8:~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet/
```

```bash
cd ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
PY=~/miniconda3/envs/expllm_env/bin/python

# 1. features — MUST start clean, a partial run leaves a bad projector behind
rm -rf ~/deepfake_data/features/clip
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml features --dataset ffpp   # ~35 min

# 2. the circuit and its competitors, all fitted on real faces only
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml fit-pc
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml baselines

# 3. numbers
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml evaluate --datasets ffpp
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml explain --n-images 2000
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml ablate-structure --epochs 25
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml bench --with-reference
$PY -u -m pcdf.cli -c configs/ffpp_clip.yaml report        # -> results/<tag>/REPORT.md

# 4. the arm with a real shot at SotA (long)
$PY -u -m pcdf.cli -c configs/ffpp_sbi.yaml train-sbi --epochs 30   # ~5 h
$PY -u -m pcdf.cli -c configs/ffpp_sbi.yaml features --dataset ffpp
#   …then fit-pc / baselines / evaluate / explain / report with ffpp_sbi.yaml
```

Long jobs: launch with `setsid nohup … > log 2>&1 < /dev/null &` in its own ssh
call. Combining `pkill` and a launch in one ssh command kills the launch too —
that cost two restarts today.

## Operational gotchas found today (do not rediscover)

1. **ONNX Runtime has no CUDA provider in this env** (`onnxruntime-gpu` 1.28
   reports only Azure/CPU), so insightface runs on CPU and is slow. MediaPipe
   `FaceLandmarker` (tasks API — 0.10.35 has no `mp.solutions`) is the default
   detector: ~2 s/video, 478 landmarks, good enough for SBI masks.
2. **Worker affinity decides ingestion throughput by 5×.** Unpinned, N workers
   each grab 24 cores and thrash (0.2 vid/s). Pinned to one core each,
   MediaPipe's internal threading is wasted (0.6 vid/s). One *block* of
   `24/workers` cores per worker: 4.7 vid/s. `data.workers=8` is the sweet spot.
3. **Seek, don't decode.** Reading every frame to keep 32 was the original
   bottleneck; `CAP_PROP_POS_FRAMES` per sampled frame is far cheaper, and it
   halves again for fakes (the reference real video is seeked too, for masks).
4. **A partial `features` run poisons the next one.** The projector is written
   first and reused if present, and completed splits are skipped — so an
   interrupted run must be deleted, not resumed.
5. The FF++ mirror ships **no official mask videos**. Masks are derived from
   `|fake_t − real_t|` on the frame-aligned real counterpart and are labelled
   `derived_frame_diff` everywhere they are used.
6. `DeepFakeDetection` (the DFD actor subset) is excluded by default: its real
   counterparts are not in this distribution, so its fakes would be compared
   against reals from a different source population.

## Open questions for tomorrow

* Celeb-DF-v2 still has to come from the official form — cross-dataset
  generalization (the G2 gate) cannot be measured without it. DF40 mirrors are
  on HF (74 GB / 160 GB) but only fakes, which forces the flagged
  `mixed_reals=True` caveat.
* Whether the CLIP arm clears the G1 gate at all. The POC's negative result
  says pooled CLIP is at chance on real swaps; the bet here is that *patch*
  tokens plus per-patch conditionals recover the local signal. If G1 fails on
  CLIP, the SBI arm is the only remaining path and should be started first.
