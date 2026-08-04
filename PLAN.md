# CVPR 2027 plan — *Tractable Forensic Reasoning*

*Written 2026-08-04. Deadline assumed **~mid-November 2026** (CVPR has opened in
November for the last three cycles — **verify on the official site before
committing to the schedule below**). That gives roughly **14 weeks**.*

---

## 1. The paper in one sentence

> Deepfake detectors built on likelihood fail for two measurable reasons, an
> exact likelihood-ratio between two probabilistic circuits repairs them, and
> because both circuits are exactly normalized the same model localizes the
> manipulation with calibrated per-region probabilities — beating memory-bank
> and flow baselines at localization while matching them at detection.

**Title candidates**
1. *Tractable Forensic Reasoning: Exact Likelihood-Ratio Circuits for Deepfake
   Detection and Localization*
2. *Fakes Are Not Outliers: Diagnosing and Repairing Likelihood-Based Deepfake
   Detection*
3. *What Makes a Face Fake? Exact Region-Level Evidence from Probabilistic
   Circuits*

(2) is the most honest framing of the strongest result and reads well as a
CVPR title; (1) is the safer, more conventional choice. Decide after the
cross-dataset numbers land — if detection is competitive, use (1); if detection
stays mid-pack, (2) leads with the part reviewers cannot dispute.

---

## 2. Contributions, with evidence status

| # | contribution | status | evidence |
|---|---|---|---|
| **C1** | **Diagnosis.** One-class likelihood fails on deepfakes for two separable reasons — *dilution* (evidence spread over thousands of nuisance dimensions) and *inversion* (forgeries are MORE likely than real faces) — established with a reusable probe-gap protocol | ✅ **have** | dilution +0.093 via exact marginals; 55.4% of fakes above real median; probe-minus-one-class gap 0.239 (CLIP) → 0.047 (SBI) |
| **C2** | **Repair.** An exact log-ratio against a self-blend density fixes it for *every* density model, without seeing a real forgery | ✅ **have** | flow 0.200→0.828, GMM 0.225→0.830, Mahalanobis 0.286→0.814, circuit 0.812→0.828; 0.953 on matched data |
| **C3** | **Localization.** Exact per-region conditional ratios beat the strongest patch baselines | ✅ **have**, needs official masks | patch-AUC 0.737 vs PatchCore 0.671, IoU 0.318 vs 0.289 |
| **C4** | **Model-level explanation.** Region-wise divergence between the two circuits — which regions carry discriminative information *in general* | ✅ **have** | face-centre regions at 115–159 nats, recovered with no face supervision |
| **C5** | **Calibration.** The ratio is a log-odds; thresholds transfer, abstention is principled | ⚠️ **need numbers** | ECE / risk–coverage not yet measured |
| **C6** | **Systems.** Tensorized exact circuits: 44.5× over the reference, d ≈ 10⁴ on one 16 GB GPU, equivalence-tested | ✅ **have** | 0.0034 vs 0.152 s/step; log Z = −1.1e-06; 14/14 tests |

**Deliberately NOT claimed:** detection state of the art. We say plainly that a
GMM with the same ratio ties the circuit on detection (0.830 vs 0.828). Owning
this pre-empts the obvious attack and buys credibility for C3–C4.

---

## 3. Story arc (section outline)

1. **Intro.** One-class deepfake detection is attractive (generalizes to unseen
   generators) but underperforms. We show *why*, fix it, and get exact
   localization for free.
2. **Motivating measurement (Fig. 1).** The probe gap: a linear probe on the
   *same* features reaches 0.775–0.859 while every density model sits at
   0.52–0.55. The signal is present; likelihood does not use it.
3. **Diagnosis (§3, Fig. 2).** Dilution and inversion, each measured through the
   circuit's own exact marginals. Estimation ruled out.
4. **Method (§4, Fig. 3).** Two compatible circuits, exact log-ratio, per-region
   conditional ratio, region-wise divergence. Show that NLL is the special case
   with a uniform alternative.
5. **Experiments (§5).** Detection (in- and cross-dataset), localization,
   calibration, robustness, ablations.
6. **Analysis (§6).** What the ratio does and does not fix; the pseudo-fake
   distribution as the residual bottleneck (with the per-family evidence).
7. **Limitations.** Detection not SotA; localization depends on pseudo-fake
   realism; derived vs official masks.

---

## 4. Figures and tables (the visual spine)

| id | content | status |
|---|---|---|
| **Fig. 1** | Probe gap bar chart across 4 representations — the motivating measurement | data ✅, figure ✗ |
| **Fig. 2** | (a) likelihood histograms real vs fake showing inversion; (b) AUC vs #coordinates kept, showing dilution | data ✅, figure ✗ |
| **Fig. 3** | Method schematic: two circuits, shared region graph, ratio, per-patch conditional | ✗ |
| **Fig. 4** | Qualitative localization: image / GT mask / PC-ratio heat map / PatchCore heat map, ~6 examples across manipulations | partial (overlay code exists) |
| **Fig. 5** | Region-wise divergence heat map over the face grid (C4) | data ✅, figure ✗ |
| **Fig. 6** | Reliability diagram + risk–coverage curve (C5) | ✗ |
| **Tab. 1** | Detection: FF++ per-method + CDF2 + DFDC-P/DF40, vs one-class baselines and published real-only detectors | needs CDF2 |
| **Tab. 2** | Localization vs PatchCore/GMM/flow + a supervised self-blend segmentation head | needs official masks |
| **Tab. 3** | Score-family ablation: NLL / two-sided / ratio / per-patch ratio | ✅ have |
| **Tab. 4** | Structure ablation: random / kd / Chow-Liu / ORC / Forman / spectral, NLL *and* AUC | ✅ have |
| **Tab. 5** | Pseudo-fake family ablation (blend / render / overshoot / statistical / all) | running |
| **Tab. 6** | Efficiency: throughput, parameters, query types supported per model | ✅ have |

---

## 5. Experiment matrix

### Have (needs only re-running at final settings)
- FF++ c23, official splits, 183,723 crops, 5 manipulations.
- Four representations (SRM / CLIP / spectral / SBI) × {NLL, two-sided, ratio}.
- Baselines: Mahalanobis, GMM, PatchCore, RealNVP — each with and without ratio.
- Structure ablation on three representations.
- Hybrid λ sweep.
- Diagnosis suite (`diagnose.py`, `probe.py`).
- Localization vs derived masks.

### Must acquire (blocking, see §8)
- **Celeb-DF-v2** — the cross-dataset headline. Without it Tab. 1 is incomplete
  and no reviewer will accept a generalization claim.
- **Official FF++ masks** — makes Tab. 2 comparable to published numbers.
- **DFDC-P** and/or a **DF40 subset** — second and third cross-dataset points;
  DF40 also covers diffusion-era forgeries.

### Must add (ours to build)
- **Calibration** (C5): ECE, reliability diagram, risk–coverage, and
  cross-dataset threshold transfer.
- **Supervised localization baseline**: a small segmentation head trained on
  self-blend masks — *fair*, since it uses no real forgery, and the strongest
  honest competitor for Tab. 2.
- **Seeds**: 3 seeds for every headline number, report mean ± std.
- **Robustness sweep** at final settings (JPEG 70/50/30, resize, blur, noise).
- **Video-level protocol** stated explicitly (mean over 32 frames) with a
  frame-level appendix.

---

## 6. Timeline (14 weeks)

| weeks | milestone | exit criterion |
|---|---|---|
| **1** (Aug 5–11) | Encoder + projection + pseudo-family results land; pick the final configuration | one config frozen; detection ≥ 0.90 in-dataset or documented why not |
| **2** | CDF2 + FF++ masks ingested (assuming the requests land); DFDC-P/DF40 subset | Tab. 1 and Tab. 2 skeletons filled |
| **3–4** | Calibration suite + supervised localization baseline + 3-seed reruns | C5 numbers exist; Tab. 2 complete |
| **5** | Robustness sweep; final ablations (Tab. 3–5) | all tables final |
| **6** | **Internal freeze on results.** Nothing new after this except rebuttal material | results/ frozen, tagged in git |
| **7–8** | Figures 1–6; first full draft | draft with every claim tied to a number |
| **9–10** | Rewrite for the diagnosis-first narrative; related work | co-author-readable draft |
| **11** | External read (advisor + one skeptical colleague); pre-empt §7 attacks | written responses to every §7 risk |
| **12** | Polish, supplementary, code release prep | anonymized repo builds from scratch |
| **13** | Buffer for the experiment that will inevitably be requested | — |
| **14** | Submission | — |

**Hard rule:** week 6 freeze. The failure mode for this project is endless
"one more representation" — we have already run four.

---

## 7. Risk register (what reviewers will attack)

| # | attack | severity | mitigation |
|---|---|---|---|
| R1 | *"Detection is not SotA."* | **high** | Own it in the abstract. Claim is tractable reasoning, not accuracy. Show we beat every *comparable* (one-class, real-only, density) method and are within X of supervised detectors. Requires competitive-not-leading numbers to be credible — if in-dataset stays below ~0.9, this paper is in trouble at CVPR. |
| R2 | *"A GMM with your ratio ties you."* | **high** | We report it ourselves, prominently. The contribution is the diagnosis + the query class, and Tab. 6 shows the GMM cannot produce C3/C4 at all. |
| R3 | *"The ratio is just SPQN / Ren et al."* | medium | Cite both. Novelty is (a) the diagnosis that motivates it for forensics, (b) exact per-REGION conditional ratios, (c) model-level divergence. |
| R4 | *"Self-blends are a compositing shortcut"* (Alpha Blending Hypothesis, 2026) | **high** | Our own per-method table shows the failure (graphics forgeries invert). Answer with the multi-family ablation (Tab. 5) and cross-dataset results. This is a strength if we present it as analysis rather than hide it. |
| R5 | *"Derived masks, not official."* | medium | Get official masks (§8). Keep derived-mask numbers in the appendix as a consistency check. |
| R6 | *"Only FF++."* | **high** | CDF2 + DFDC-P/DF40. Non-negotiable. |
| R7 | *"Circuits are unnecessary machinery."* | medium | Tab. 6 (query types × models) + the localization win. If C3 evaporates at final settings, the paper has no CVPR-strength claim → go to plan B. |
| R8 | *"No statistical rigour."* | low | 3 seeds, mean ± std, on every headline number. |

---

## 8. Blocked on data (highest-value actions, not ours to do)

1. **Celeb-DF-v2** — request form on the official repo. Ingestion already
   written (`build_celebdf_manifest`), so it is one command on arrival.
2. **Official FF++ masks** — same EULA as the dataset; needed for Tab. 2 to be
   comparable to published localization numbers.
3. **DFDC-P / DF40 subset** — DF40 mirrors are public (74 GB / 160 GB, and
   fake-only, which forces a documented real-side caveat).

Disk on the workstation is at **64 GB free**; ingest one dataset at a time with
crop-then-purge, and delete the FF++ zip (17 GB, redundant) first.

---

## 9. Plan B, decided in advance

If, at the week-1 freeze, detection stays materially below ~0.90 in-dataset
**and** the localization win does not survive official masks:

- **B1 — venue shift.** The diagnosis (C1) + repair (C2) + tractability (C6) is
  a strong **UAI / AISTATS / TMLR** paper, where "a GMM ties us on accuracy" is
  a finding rather than a rejection. The tractable-models community is the
  natural audience for C3–C4 anyway.
- **B2 — split.** A short forensics paper on C1+C2 (model-agnostic, immediately
  useful) and a circuits paper on C3+C4+C6.
- **B3 — reframe as a benchmark/analysis paper.** "Why one-class deepfake
  detection fails": four representations, five density models, two failure
  modes, one repair — with the code as the contribution. Viable at a CVPR
  workshop or as a dataset/benchmark track submission.

Plan B is not failure. C1 and C2 are true, useful and reproducible regardless
of where the detection number lands.

---

## 10. What we already know that most papers in this area do not

Worth keeping visible while drafting, because these are the sentences that make
the paper interesting rather than merely competent:

- Fakes are **more typical** than real faces — the assumption the entire
  one-class literature rests on is inverted, and we measured it (55.4%).
- A better density is **not** a better detector: Forman structure improves
  held-out likelihood by 169 nats and changes AUC by 0.00.
- The discriminative objective **saturates** on self-blends (loss 0.0000, AUC
  0.9996) while real-forgery AUC stays at 0.827 — proof that the bottleneck is
  the pseudo-fake distribution, not the model or the objective.
- Every density model converges to ~0.83 with the ratio, from as low as 0.20 —
  the score function mattered far more than the model class.
