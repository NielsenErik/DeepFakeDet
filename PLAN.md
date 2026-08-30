# Plan — *What exact tractable inference buys deepfake forensics*

*Rewritten 2026-08-21, replacing the 2026-08-04 CVPR plan (`git show 83b855d:PLAN.md`).
Four of that plan's six contributions are dead; see §2. This rewrite is the action
required by `hands_off.md` §7.2, and it is built on the two results that survived
(`hands_off.md` §2) plus a literature survey done 2026-08-21 (§4–§6).*

**Read order:** `hands_off.md` → `STATUS.md` (Findings 0–19) → this file.

---

## 0. The situation in four numbers

| | FF++ c23 | Celeb-DF-v2 |
|---|---|---|
| circuit, exact log-ratio (**S1**) | 0.8411 | **0.8357** — drops 0.005 |
| linear probe, supervised, identical features | 0.9192 | 0.8588 — drops 0.060 |
| official SBI encoder, end to end | 0.8657 | 0.8921 |
| PC ratio localization, patch AUC (**S2**) | **0.7137** | n/a — CDF ships no masks |

Everything below is about closing the gap between 0.8357 and a number that
survives a vision-venue reviewer, without inventing a fifth result that later
turns out to be an artefact.

---

## 1. The bar in 2026, measured — not assumed

This project has twice reasoned from an unsourced number (`hands_off.md` §8.5).
The 0.9287 SBI figure the plan has treated as "SotA" is three years old and is
no longer the reference point:

| system | CDF-v2 | protocol / note |
|---|---|---|
| SBI (2022), the encoder we sit on | 0.9287 | FF++ reals only |
| **GenD** ([arXiv 2508.06248](https://arxiv.org/html/2508.06248v4)) | **0.946** | CLIP ViT-L, LayerNorm-only tuning (0.03% of params); 0.912 avg over 14 benchmarks |
| **BlenD** ([arXiv 2605.10334](https://arxiv.org/abs/2605.10334)) | ~0.913 avg over 15 datasets | real-only faces + SBI, no generated fakes used |
| **BlenD + FS-VFM, ensembled** | **0.940 AUROC** | two *complementary* detectors |
| ours, circuit exact log-ratio | 0.8357 | one-class, never sees a forgery |

**Every number in this table is second-hand and must be re-measured or quoted as
"reported" before it enters a draft.** The GenD row in particular came from an
HTML fetch, not from a run.

Two consequences drive the whole plan:

1. 0.8357 is not a competitive detection number and no framing makes it one.
2. **No headline number in 2026 comes from a single model.** The two strongest
   rows are (a) a foundation backbone with almost no tuning and (b) an ensemble
   of two detectors that fail differently. Both routes are open to us and
   neither requires abandoning the circuit.

---

## 2. Contributions — the 2026-08-04 list, audited

| # | 2026-08-04 claim | status now | evidence |
|---|---|---|---|
| **C1** | Diagnosis: dilution + inversion | **dead as stated** | the dimension gap *inverts* on the published encoder, on both datasets (Finding 16). It described our reimplementation, not forgeries |
| **C2** | Repair: exact log-ratio fixes one-class density | **survives → S1** | 0.6944 → 0.8411 in-dataset, 0.6823 → 0.8357 cross-dataset; drop of 0.005 vs 0.060 for supervision (Findings 17, 18) |
| **C3** | Localization by exact per-region conditionals | **survives → S2** | first on all four metrics; +0.0277 over strongest baseline (Finding 19) — **narrowly misses our own +0.03 gate** |
| **C4** | Region-wise divergence, family posterior | **the family-posterior version is dead** | the region-divergence map itself still holds |
| **C5** | Calibration | **falsified** | ECE 0.780 raw, 0.766 after temperature scaling (T = 110) |
| **C6** | Systems: tensorized exact circuits | **survives** | 44.5×, d ≈ 10⁴ on 16 GB, equivalence-tested, 28 tests pass |

**The honest paper that exists today**, with no further experiments:

> *What exact tractable inference buys deepfake forensics — and what it doesn't.*
> A domain-invariant one-class detection score that never beats the classifier it
> sits on; the best localization by a margin that just misses our own
> pre-registered gate; and an accounting of five results that looked real and
> were not.

Venue for that version: **UAI / AISTATS / TMLR**. §3 is about whether the
stronger version is reachable in time.

---

## 3. The gate: which paper we are writing

Run **L1 (§4.1) and L2 (§4.2) first**, then decide. Nothing else in this plan is
worth starting before that decision.

| if | then | venue |
|---|---|---|
| fused CDF ≥ **0.93** *and* the circuit is a load-bearing ensemble member | detection is a real contribution; S1 becomes "a forgery-free component that improves a SotA detector under shift" | **CVPR 2027** (deadline ~mid-Nov 2026 — *verify on the official site*), ~12 weeks |
| fused CDF lands 0.90–0.93, or the gain needs the circuit only marginally | S1 stays a stability result, S2 carries the novelty | **TMLR / UAI**, no deadline pressure |
| fusion buys nothing (scores highly correlated) | write §2's honest paper as-is | **TMLR**, submit ~6 weeks |

The failure mode this project has repeatedly hit is running one more encoder
instead of deciding. **The gate is a week-1 decision, on evidence, in writing.**

---

## 4. The five levers, ranked by (expected gain) / (cost on one RTX 4080)

### L1 — Fuse the ratio with the classifier it sits on
**Cost: days. The data is already on disk. Do this first.**

Yermakov, Čech, Fritz & Matas ([arXiv 2605.10334](https://arxiv.org/abs/2605.10334),
May 2026) show that "explicit blending searchers" (the SBI family) and models
"resilient to blending shortcuts" are **highly complementary**: BlenD 91.3 +
FS-VFM 90.0 → **94.0 AUROC** ensembled. That is the mechanism this project has
circled for months without naming.

Finding 17 is the precondition for a good ensemble member: the ratio is *stable*
(−0.005) where the discriminative stage is *accurate but brittle* (−0.060).
Stability and accuracy failing on **different examples** is what makes fusion work.

Steps:
1. From `results/full_picture_{ffpp,celebdf}.json`, take per-video scores for
   `circuit, exact log-ratio` and `official encoder, end to end`. Compute
   Spearman ρ and error-disagreement (videos one gets right and the other wrong).
   **Low ρ with both above chance is the result.**
2. Fuse two ways: (a) rank-average — no fitting, cannot be accused of tuning on
   the target domain; (b) logistic stacking fitted **on FF++ only**, evaluated on
   CDF. Report both.
3. Target claim, and it is honest and strong:
   *the exact log-ratio adds Δ AUC to a SotA discriminative detector at zero
   forgery supervision, and the gain grows with domain shift.*
   If fused CDF clears 0.9287 we have beaten published SBI with a component that
   never saw a forgery.

Risk: high ρ ⇒ fusion buys nothing. That costs an afternoon and settles §3.

### L2 — Change the backbone (and it is the ablation reviewers will demand anyway)
**Cost: ~1 week.**

The pipeline is backbone-agnostic (features → projector → circuit), and Findings
16–18 already proved *the conclusions change with the encoder*. Today that is a
weakness in the writeup — "your results depend on which encoder you chose". A
**third** encoder converts it into the paper's strongest methodological claim:
S1's invariance either holds across three representations or it does not.

Candidates, both with public weights:

- **FSFM / FS-VFM** — CVPR 2025 + extension [arXiv 2510.10663](https://arxiv.org/abs/2510.10663),
  code [wolo-wolo/FSFM-CVPR25](https://github.com/wolo-wolo/FSFM-CVPR25).
  Self-supervised on **real faces only** (masked image modelling + instance
  discrimination). Philosophically the right partner for a one-class density
  model: the representation itself never sees a forgery, so the system becomes
  forgery-free end to end. Also the exact model BlenD found complementary.
- **CLIP ViT-L/14 + LayerNorm-only tuning** (GenD recipe). 0.946 CDF at 0.03%
  trainable params — cheap here. Two of their findings bear directly on us:
  (i) **paired real/fake from the same source video** is worth +4.7pp against
  shortcut learning; (ii) older-but-diverse data (FF++) beats recent-only data.

Caveat to test, not assume (InsightFace, May 2026 review): frozen foundation
backbones stay strong on full-face synthesis but degrade on **localized editing**
— precisely the regime S2 lives in.

### L3 — Feed the circuit far more real faces
**Cost: I/O, not research risk. It is *our* contribution that benefits.**

Every real-only system that works in 2026 trains on a large real corpus, not on
FF++'s 720 real videos:

- BlenD: "large-scale, diverse dataset of real-only facial images augmented with SBI".
- µFlow: FFHQ, 70k faces.

The literature names our constraint explicitly: *"the FF++ dataset contains only
1,000 real videos with imbalanced facial attribute distributions."* The circuit
estimates p(real) — it is the component most starved by that. FFHQ + CelebA-HQ
(+ VGGFace2) through the same 256px / mediapipe / margin-1.3 pipeline strengthens
the density estimate that every other score in the system is a ratio against.

Watch disk: 83 GB free, 91% used (`hands_off.md` §5). Delete `features/combined/`
(4.9 GB) first — it does not beat its own better source.

### L4 — µFlow's averaged-feature base distribution
**Cost: ~2 days. Must be cited regardless — it is the nearest competitor.**

**Read this one first.** µFlow ([arXiv 2606.30528](https://arxiv.org/html/2606.30528),
June 2026) is one-class, real-only, normalizing-flow, log-likelihood-as-fakeness —
the same slot as S1. Its diagnosis *is* our Finding 18: *"mapping features into a
normal distribution results in high likelihoods for both classes, limiting current
methods' performance."* Plain density scoring sits near chance; that is our 0.6944.

Their repair differs from ours and is transplantable: average K real faces, fit a
GMM in that averaged-feature space, then train the flow to map *single* real
images into that discriminative distribution instead of a standard Gaussian.
Averaging amplifies generator traces and suppresses identity.

Our repair is the exact log-ratio of two circuits. **They compose** — fit the
circuit's target in the averaged space. If it works, a number; if not, a
principled head-to-head against the only other real-only detector in this space.

**Protocol warning:** µFlow trains on FFHQ and tests on CelebA-HQ + WILD (19
unseen generators); it does *not* evaluate on FF++ or CDF. Its 96.8 is **not**
comparable to our 0.8357. Do not put them in one table without re-running one side.

### L5 — More datasets: what turns S1 from a claim into a result
**Cost: 1–2 weeks, mostly ingestion.**

n = 2 cannot support an invariance claim and a reviewer will say so.
`hands_off.md` §7 already lists **DFDCP (85.51)** and **FFIW (83.22)** as
validated targets in the SBI reproduction table, so both can be checked the way
Celeb-DF was (Finding 12). Add:

- **DF40** ([NeurIPS 2024](https://github.com/YZY-stack/DF40)) — 40 forgery
  methods including diffusion. **Image-level labels only, no masks.** This is
  where "invariant across *generator families*, not just two datasets" is earned.
- **WILD** (via µFlow) — 19 unseen generators, GAN + open/closed diffusion.

The strongest form of S1: *the exact log-ratio drops < 0.02 across k datasets and
g generator families, while a supervised probe on identical features drops
0.06–0.15.* That is a real result even at 0.83 absolute — the y-axis is stability,
not accuracy — **but only at k ≥ 4.**

DFD remains unobtainable (all three FaceForensics mirrors dead or 403).

---

## 5. Two papers that change the plan, not just the numbers

### 5.1 The Alpha Blending Hypothesis explains `noleak_clean`

[arXiv 2605.10334](https://arxiv.org/abs/2605.10334) — Yermakov, Čech, Fritz, Matas, 11 May 2026.

*This paper is already in the old plan as risk R4. What is new is that it is not
only an attack — it is an explanation and a route.*

Thesis: SotA frame-based detectors "primarily function as alpha blending
searchers" — they localize low-level compositing artifacts, not semantic
anomalies or generative fingerprints.

That is `hands_off.md` §3's unexplained result from the other side. Removing the
leak cleanly (`pristine_background=True, compress_policy="none"`) costs 0.064
in-dataset **and** 0.063 cross-dataset because the compositing artifact is not a
shortcut *contaminating* the signal — under this hypothesis it substantially
**is** the signal the SBI family detects. Both our datasets are compositional, so
removing it hurts on both. That is exactly why the train/test-compression-shift
explanation was excluded and nothing replaced it.

Consequences:
- **F1 is unblocked for the writeup.** It stops being the open question that
  gates publication (`hands_off.md` §7.3) and becomes a corroborated finding with
  an independent citation.
- **Verify before relying on it.** Their claim is about *detectors*; ours is
  about a *generator ablation*. The test: does `noleak_clean` degrade specifically
  on **compositional** forgeries and not on fully-generated ones? Rerun that
  ablation row on DF40's diffusion / EFS subsets. If the loss vanishes there, the
  mechanism is confirmed on our own data.
- **It re-motivates S2.** If detection is blending-search, then a model that can
  *integrate exactly over an arbitrary region* is the natural instrument for it.
  That is a better motivation than the one in the old plan.

### 5.2 µFlow is the paper that could scoop S1

[arXiv 2606.30528](https://arxiv.org/html/2606.30528), June 2026. Same slot,
different protocol, already public, and it beats real-only baselines by 11–15
points on its own benchmark. It does not evaluate on FF++/CDF and does not do
localization — that is our room. But a reviewer who knows it will ask why our
one-class number is 0.83 when theirs is 0.968. **The answer is protocol, and the
paper must say so explicitly and early**, not in a footnote.

---

## 6. Localization: the missed gate, and how to retire it

S2's margin is **+0.0277** over Mahalanobis against a pre-registered gate of
**+0.03**. Against PatchCore alone it is +0.0408 and clears — *quoting only that
comparison would clear a gate the result does not clear, and must not happen.*

Three routes, strongest first:

1. **A second mask source.** "Wins on two datasets by +0.02" is stronger than
   "wins on one by +0.03", and it retires the gate argument instead of litigating
   it. **DDL** ([arXiv 2506.23292](https://arxiv.org/abs/2506.23292)) is the
   target: **1.18M+ spatial masks, up to 80 methods**, built precisely because
   DF40 has labels but no masks. **Dolos** is the smaller single-face,
   local-manipulation-mask option and is faster to stand up. FFIW is multi-face.
   **DF40 is unusable here** — binary labels only. Celeb-DF has no masks and
   never will.
2. **State the novelty plainly.** A search of the SPN / PC literature (surveys,
   `awesome-spn`, the mammography anomaly-detection work) finds **no application
   of probabilistic circuits to face forgery localization**. Exact per-region
   conditionals in a 1000+-dimensional feature space is an untaken position.
   *That*, not a 0.03 margin, is the contribution.
3. **Raise the ceiling with L2/L3.** The baselines gained far more than we did
   from a better encoder (Mahalanobis 0.5404 → 0.6860). Finding 19's honest
   reading — "most of the localization signal was the representation, not the
   circuit's exactness" — is itself a publishable negative about the whole
   patch-anomaly literature and deserves its own subsection.

Related work we will be asked about: **LAA-Net** (CVPR 2024,
[arXiv 2401.13856](https://arxiv.org/html/2401.13856v2)) and **LAA-X**
(arXiv 2604.04086) on localized-artifact attention; **MFVLR** and **DiffusionFF**
(arXiv 2508.01873) on diffusion-face localization; **SeeABLE**
(arXiv 2211.11296), the one-class self-supervised localization ancestor.

---

## 7. Also cite

- **Ren et al., Likelihood Ratios for OOD Detection** ([arXiv 1906.02845](https://ar5iv.labs.arxiv.org/html/1906.02845))
  — the background/semantic decomposition our two-circuit ratio is an *exact*
  instance of. Our delta: exactness, and the conditional region query.
- **Effort / Orthogonal Subspace Decomposition** (ICML 2025 spotlight, arXiv
  2411.15633) and **Low-rank Orthogonal Subspace Intervention** (arXiv 2601.11915)
  — the current "preserve the foundation model, add a forgery subspace" line.
- **Forgery-aware Layer Masking + multi-artifact subspace decomposition** (arXiv 2601.01041).
- **NTIRE 2026 Robust Deepfake Detection Challenge report**
  ([arXiv 2604.24163](https://arxiv.org/pdf/2604.24163)) — the current competitive
  protocol. Its 4th-place entry is a calibrated ensemble of complementary
  pathways, i.e. L1 with more members.
- **PyJuice / Scaling Tractable PCs** (arXiv 2406.00766) and **Scaling PCs via
  Data Partitioning** (arXiv 2503.08141) — if L3 makes circuit fitting the bottleneck.

---

## 8. Schedule

| week | action | lever | why now |
|---|---|---|---|
| **1** | complementarity + fusion on existing scores | L1 | free; decides whether the paper has a detection number at all |
| **1** | read µFlow; write the protocol-difference paragraph | L4 | scooping risk |
| **1 (end)** | **§3 GATE: choose the paper and the venue, in writing** | — | the decision this project keeps deferring |
| **2** | FSFM/FS-VFM features through the existing pipeline | L2 | third encoder; makes or breaks S1 |
| **2–3** | FFHQ + CelebA-HQ reals ingested; circuit refit | L3 | strengthens p(real), the actual contribution |
| **3** | DFDCP + FFIW for S1; DF40 if disk allows | L5 | n = 2 → n = 4 |
| **3** | `noleak_clean` rerun on DF40 diffusion subset | §5.1 | confirms or kills the Alpha-Blending explanation on our data |
| **3–4** | Dolos or DDL masks for S2 | §6.1 | retires the gate |
| **4** | µFlow averaged-feature base distribution in the circuit | L4 | cheap upside |
| **5** | 3 seeds on `base` / `hull` / `sam`, scored on CDF | — | every ablation number is single-seed against a ±0.02 floor |
| **6** | **Results freeze.** Figures only after this | — | the "one more representation" failure mode |

---

## 9. Standing rules (violating these is how the last four months happened)

1. **Score on Celeb-DF, never on in-dataset val.** Spearman(FF++ val, CDF) = 0.74
   and the ablation *re-ranks* (`hands_off.md` §4).
2. **The noise floor is ±0.02.** Anything smaller is not a result.
3. **Re-measure any number with no artefact behind it** — including every number
   in §1 of this file, which is second-hand.
4. **Never quote the favourable baseline comparison alone** (§6).
5. **Single-seed numbers are provisional** until week 5.
6. Watch the gotchas in `hands_off.md` §8 — the hardcoded manifest path, the
   non-drop-in official weights, the float32 `log_ball` floor at eps ≥ 1e-3, and
   the `patch_surprisal` chunking that OOMs *after* training.

---

## 10. What we know that most papers in this area do not

Kept visible while drafting — these are the sentences that make the paper
interesting rather than merely competent. **Revised**: the 2026-08-04 version of
this list was four-fifths artefact.

- **A better density is not a better detector.** Forman structure improves
  held-out likelihood by 169 nats and changes AUC by 0.00.
- **The score function mattered far more than the model class.** Every density
  model converges to ≈ 0.83 with the ratio, from as low as 0.20.
- **The one-class family needs a *worse* encoder to work** (0.8125 on ours vs
  0.6944 on the published one) — density models exploit dataset-specific
  structure, and a clean representation removes the crutch. Finding 18.
- **Most of the localization signal is the representation, not the model.**
  Mahalanobis goes 0.5404 → 0.6860 on a better encoder while the circuit gains
  little. Finding 19.
- **Five published-looking results in this project were artefacts**, traced to a
  reimplementation that does not transfer, a bug in our own pseudo-fake generator,
  and a comparison against a number from the wrong compression level. Written up
  properly, that accounting is useful to the field.
