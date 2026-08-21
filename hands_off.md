# Hand-off — DeepFakeDet

Last updated: **2026-08-06, 19:10 CEST.** All jobs stopped by request; the GPU is
idle. Nothing is committed — the working tree has ~20 modified/new files (see
§7).

Read order for a new session: **this file → `STATUS.md` (the findings, in full)
→ `PLAN.md` (the CVPR plan, now partly invalidated — see §6)**. `POC.md` and the
historical appendix at the bottom are the July record and are superseded.

---

## 1. Where the project actually is

The project set out to transfer PCNET (probabilistic circuit as tractable
density estimator, NLL as anomaly score) from LLM hallucination detection to
deepfakes. **As a detector, it does not work, and as of today we know precisely
why, with numbers.** As a *diagnosis*, it now has a genuinely novel result and a
quantitative law that no discriminative method could have produced.

The one-sentence state: **98% of the distance to a state-of-the-art detector is
in the encoder, not the circuit; every recipe-level fix for the encoder has been
tested and failed; and the binding constraint is the pseudo-fake distribution,
whose coverage of a manipulation predicts that manipulation's detectability with
Spearman ρ = 1.000.**

### The measured gap (`results/sbi_g8c16_kd-orc_K8/gap_waterfall.json`)

Every row measured on the *same crops* — the script scores the encoder on
exactly the images whose projected features the probe and circuit see.

| stage | FF++ video AUC | lost |
|---|---|---|
| published SBI on FF++ c23 (reported) | 0.9964 | — |
| **our encoder, end to end** | **0.8312** | **+0.1652** |
| linear probe on projected features (1024 d) | 0.8591 | −0.0280 |
| circuit, one-class NLL | 0.8125 | +0.0467 |
| circuit, exact log-ratio | 0.8283 | −0.0158 |
| | total **0.1681** | encoder = **98%** |

The `0.860` encoder number that sat in `STATUS.md` from Aug 4 to Aug 6 was
**never measured** — no log or JSON produced it. The real value is 0.8312.
If you find an unsourced number in these docs, distrust it; that one cost a day
of reasoning built on a wrong anchor.

---

## 2. The four findings worth keeping

**F1 — The pseudo-task leaks a global cue that real forgeries do not have.**
(`scripts/shortcut_audit.py`, `results/shortcut_audit.json`.) Classifying real
vs self-blend using *only* 8×8 JPEG blocks entirely outside the dilated blending
mask — pixels the forgery never touches:

| self-blend recipe | leak AUC outside the mask |
|---|---|
| what the project trained on | **0.937** |
| no post-blend re-encode | 0.812 |
| pristine background only | 0.917 |
| **pristine background + no re-encode** | **0.500** |
| **pristine background + symmetric re-encode** | **0.500** |

Two mechanisms, neither sufficient alone: `self_blend` ends with a JPEG q88–96
re-encode the real image never gets, and `source_target_pair` perturbs *both*
copies then swaps them with p=0.5, so ~half the time the whole context is the
degraded copy. Same features on **real FF++ forgeries**: pooled **0.477**
(per-method 0.47–0.53) — chance, because in a real swap the context *is* the
original frame. Verified the null is real: under the clean recipe the periphery
pixels are **bit-identical** (max diff 0 over 40 images,
`scripts/_check_periphery.py`).

This is **distinct from the published position.** "The Alpha Blending
Hypothesis" (arXiv 2605.10334) argues detectors are *boundary* searchers and
explicitly treats SBI as a generic heuristic without dissecting the generator.
Our signal is not at the boundary and not in the manipulated region at all. If
it reproduces on the reference SBI implementation it affects the whole
SBI-derived line (SBI, FSBI, BlenD, …) as a **leakage audit any pseudo-fake
generator should have to pass**. *This is the most publishable thing here and it
has not been checked against the official SBI code — do that first (§5).*

**F2 — Coverage under the pseudo-fake density predicts detectability.**
(`results/sbi_g8c16_kd-orc_K8/family_mixture.json`.)

| manipulation | coverage | mean log-ratio | AUC |
|---|---|---|---|
| Deepfakes | 0.733 | −1000 | 0.921 |
| Face2Face | 0.563 | −1564 | 0.872 |
| FaceShifter | 0.506 | −1846 | 0.844 |
| NeuralTextures | 0.456 | −2300 | 0.808 |
| FaceSwap | 0.297 | −3845 | 0.699 |

**Spearman ρ = 1.000, Pearson r = 0.998** (mean log-ratio vs AUC). n = 5, so
this is a strong suggestion, not a law — it needs Celeb-DF-v2 / DF40 to become a
claim. This is the result that *requires* an exactly-normalized density: it is a
statement in nats about a distribution, not a score.

**F3 — No recipe-level fix recovers the encoder gap.**
(`results/encoder_ablation.json`, 20 epochs, 16 frames/video, identical protocol.)

| variant | val AUC | vs base | train loss |
|---|---|---|---|
| `sam` | 0.8747 | **+0.0030** | 0.0120 |
| `base` | 0.8716 | — | 0.0155 |
| `hull` | 0.8610 | −0.0106 | 0.0267 |
| `all` | 0.8543 | −0.0173 | 0.0306 |
| `noleak` | 0.8468 | −0.0249 | 0.0197 |

SAM — the most plausible difference from published SBI — buys noise.
**Caveat:** val AUC falls monotonically with training loss, and we cannot yet
separate "harder pseudo-task transfers worse" from "harder task is less
converged at a fixed 20-epoch budget". Also `noleak` is confounded:
`compress_policy=symmetric` re-encodes training images while val/test stay
untouched. `noleak_clean` exists to fix that and **has not run**.

**F4 — The saturation is the encoder's fault, not the leak's.** A falsifiable
prediction was made and **failed**: on leak-free blends the hybrid λ sweep is
still fully saturated (real-vs-blend **1.0000**, λ moves FF++ only 0.825→0.813).
Removing the leak does not make self-blends harder to separate *in SBI feature
space*, because that space was built by training on exactly this
discrimination. The representation and the pseudo-fakes are the same
construction, so the density task on top is degenerate by design.
(λ=0 still destroys localization, 0.678 → 0.462.)

### Negatives that close doors (don't redo these)

- **The projection is not a bottleneck and is not even lossy.** Probe on 16×64
  = 0.8591 > encoder 0.8312. **`sota_push.sh`'s `grid=12, out_dim=32` sweep is
  cancelled** — it was chasing 0.001.
- **The mechanism mixture buys nothing.** `p_mix` over 4 families = 0.8286;
  every single family alone = 0.827–0.831, with near-identical per-method
  breakdowns even for families designed to deviate in opposite directions. At
  log-ratios of thousands of nats one component dominates the logsumexp, so the
  mixture degenerates to a max.
- **The exact family posterior is uninformative.** Real faces get `blend` 0.599;
  every manipulation gets `blend` 0.49–0.66. No Face2Face→`render` structure, at
  image or region level. The encoder collapses mechanisms it was never asked to
  distinguish.
- **C5 (calibration) is not supported.** ECE 0.780 raw, 0.766 after temperature
  scaling (T = 110); risk–coverage 0.780 → 0.588 at 24% coverage. The ranking is
  fine, the probabilities are meaningless. **Drop C5 from PLAN.md's contribution
  list or rewrite it as a negative.**

---

## 3. Machine, environment, data

```bash
ssh jawa17@192.168.1.8            # RTX 4080 16 GB, 62 GB RAM, 24 cores, Ubuntu
repo   ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
data   ~/deepfake_data            # NOT in the repo
PY     ~/miniconda3/envs/expllm_env/bin/python    # torch 2.10+cu128, timm, mediapipe
```

**Disk is the binding constraint: 57 GB free.** `~/deepfake_data/raw/ffpp_zip`
(17 GB) is redundant — the videos are unzipped at `raw/ffpp` (17 GB) and the
crops are extracted. Deleting the zip is the easiest 17 GB back.

Sync from the Mac (there is no shared checkout; the workstation is a copy):

```bash
cd ~/Documents/UNITN/PHD/MAIN_Project/DeepFakeDet
rsync -az --exclude __pycache__ --exclude .DS_Store \
  pcdf scripts tests configs STATUS.md hands_off.md \
  jawa17@192.168.1.8:~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet/
```

### Data state (exact, as of the stop)

| artefact | state |
|---|---|
| `crops/` (256px q95) | complete, 183,723 crops — everything published so far uses this |
| `crops_hires/` (380px q100 4:4:4) | **TRAIN reals only** (696 dirs, 2.1 GB). Manifest `manifests/ffpp_ingested_crops_hires.csv` lists train only. **VAL was killed mid-ingest and must be redone.** |
| `features/{srm,clip,spectral,sbi}/` | complete |
| `features/sbi/*_blend-{blend,render,overshoot,statistical}P.*` | complete (pristine-background per-family sets) |
| `models/sbi_effnetb4.pt` | the encoder everything uses — val 0.8722, **test 0.8312** |
| `models/sbi_ab_{base,noleak,sam,hull,all}.pt` | the ablation checkpoints |

Results JSONs on the workstation under `~/deepfake_data/results/`:
`shortcut_audit.json`, `encoder_ablation.json`, and in
`sbi_g8c16_kd-orc_K8/`: `gap_waterfall.json`, `family_mixture.json`,
`hybrid_sweep_noleak.json`, `probe.json`, plus the Aug-4 files.
**None of these are in the repo — pull them before the machine is wiped.**

---

## 4. Resume commands

```bash
cd ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
PY=~/miniconda3/envs/expllm_env/bin/python
```

**(a) Finish the hires ingest** — required before the hires variants can train:

```bash
$PY -u -m pcdf.cli -c configs/ffpp_sbi.yaml \
  -s data.crop_size=380 -s data.jpeg_quality=100 -s data.workers=6 \
  -s data.n_frames_test=8 \
  ingest --dataset ffpp --splits val --crops-dir crops_hires   # ~6 min
```

**(b) The four variants that never ran** (`encoder_ablation.py` skips anything
already in the results file, so this only runs what is missing):

```bash
$PY -u scripts/encoder_ablation.py --epochs 20 --frames-per-video 16 --workers 8 \
    --variants noleak_clean all_clean          # ~45 min — the unconfounded leak test
$PY -u scripts/encoder_ablation.py --epochs 20 --frames-per-video 16 --workers 8 \
    --variants hires hires_sam                 # ~75 min — the last untested hypothesis
$PY scripts/show_ablation.py                   # table
```

Long jobs: `setsid nohup … > log 2>&1 < /dev/null &` in its **own** ssh call.
Combining `pkill` and a launch in one ssh invocation kills the launch too.

**(c) Inspect what already exists:**

```bash
$PY scripts/show_ablation.py
$PY scripts/show_mixture.py
```

---

## 5. What to do next, in priority order

1. **Validate F1 against the official SBI implementation.** Everything about the
   leak is currently a statement about *our* reimplementation of `self_blend`.
   Clone Shiohara & Yamasaki's released code, generate blends with it, and run
   `scripts/shortcut_audit.py --tests T1` on those. If the leak is there too,
   F1 is a contribution to the whole field and the paper writes itself. If it is
   not, F1 is a bug report about our code and must be reported that way. **This
   is cheap (CPU, ~20 min) and it decides how big the result is. Do it first.**
2. **Run (a)+(b) above.** `hires` is the only untested cause of the encoder gap.
   If it also fails, stop trying to reach detection parity.
3. **Get Celeb-DF-v2.** F2 (ρ = 1.000 on n = 5) is the strongest positive result
   and needs cross-dataset points to be a claim rather than an anecdote.
   Ingestion is already written (`build_celebdf_manifest`); it is one command on
   arrival. Official request form — not something a session can do for you.
4. **Rewrite `PLAN.md`.** See §6.
5. Optional, cheap: 3 seeds on the `base` / `sam` / `hires` variants — the
   0.003–0.03 differences in F3 are currently single-seed and some are noise.

---

## 6. What this does to the CVPR plan

`PLAN.md` §2 lists six contributions. After today:

- **C1 (diagnosis)** — *stronger*. Now dilution + inversion **+ F1 (leakage) +
  F2 (coverage predicts detectability)**. This is the paper.
- **C2 (repair, the ratio)** — unchanged and still model-agnostic (GMM 0.830 vs
  circuit 0.828). The mixture extension **failed** to improve it.
- **C3 (localization)** — unchanged, still the circuit's only exclusive win
  (0.737 vs PatchCore 0.671). Untouched today.
- **C4 (model-level explanation)** — the *family posterior* version **failed**.
  The region-divergence version from Aug 4 still stands.
- **C5 (calibration)** — **falsified. Remove or rewrite as a negative.**
- **C6 (systems)** — unchanged and solid.

The week-1 exit criterion was "detection ≥ 0.90 in-dataset **or documented why
not**". It is now documented why not, in five independent ways. **Plan B (§9 of
PLAN.md) is the live branch**, and title candidate (2), *"Fakes Are Not Outliers:
Diagnosing and Repairing Likelihood-Based Deepfake Detection"*, is now clearly
the right framing — with F1 and F2 as the new spine. The venue argument for
UAI/AISTATS/TMLR is stronger than it was this morning.

---

## 7. Uncommitted work (nothing is in git)

Modified: `STATUS.md`, `pcdf/cli.py`, `pcdf/data/faces.py`, `pcdf/data/sbi.py`,
`pcdf/models/supervised.py`, `pcdf/stages.py`, `scripts/hybrid_sweep.py`.

New: `pcdf/models/family_mixture.py`, `tests/test_family_mixture.py`,
`scripts/{shortcut_audit,encoder_ablation,family_mixture_experiment,gap_waterfall,show_ablation,show_mixture,_check_periphery}.py`,
`scripts/queue_{mixture,waterfall,ablation2,hires}.sh`.

Test suite: **23 passed** (9 new, covering shared region graph, log Z ≈ 0 per
mixture component, posterior normalisation, ratio = logsumexp identity).

### New capability added today

- `self_blend(..., pristine_background=, hull_variety=)`, `multi_family_blend(...)`
  likewise; `SbiConfig` gained `compress_policy` / `pristine_background` /
  `hull_variety` / `optimizer` (SAM implemented) / `crops_dirname` / `tag`.
- `pcdf features --pseudo-family NAME --pristine-background`.
- `pcdf ingest --splits ... --crops-dir NAME` (a named crop set writes a
  **named manifest** — a partial re-ingest must never clobber
  `ffpp_ingested.csv`, which is the index the whole pipeline reads).
- Crops at `jpeg_quality >= 98` now also disable 4:2:0 chroma subsampling.

---

## 8. Gotchas (new today; the July ones are in the appendix)

1. **Pseudo-blend feature extraction used to blend the forged videos too**, then
   discard them via the `label==0` filter downstream — 6× the work for nothing.
   Fixed in `cmd_features`; if you touch that path, keep the filter.
2. **`compress_policy=symmetric` is not a clean control.** It re-encodes
   training images only, so it swaps a leak for a train/test shift. Use
   `pristine_background=True, compress_policy="none"` (`noleak_clean`) — the
   leak sweep puts both at exactly 0.500.
3. **Group your splits when auditing paired data.** A random row split put each
   real and its bit-identical blend twin on opposite sides; the model memorised
   the vector with the wrong label and a true null reported AUC 0.35 instead of
   0.500. `auc_lr(..., g0=, g1=)` in `shortcut_audit.py` handles it.
4. **Restrict "periphery-only" features to periphery blocks — all of them.** The
   first version averaged blockiness over the whole frame, letting the
   manipulated region leak into a measurement whose entire point was to exclude
   it.
5. **`setsid nohup` survives everything**, including the parent ssh being
   killed — verified today when four local wrappers were terminated and every
   remote job kept running. To actually stop a chain, **kill the parent `bash
   scripts/queue_*.sh` first**, then the python children, or the script simply
   launches the next stage.
6. **Model selection on 4 frames/video is optimistic.** The encoder's val AUC is
   0.8722 and its test AUC is 0.8312 — a 0.041 gap that hid for two days.

---

# Appendix — July 2026 POC record (superseded, kept for provenance)

The LFW/OpenForensics POC and its findings are in `POC.md`. The two conclusions
that still matter:

- **Semantic embeddings are dead on arrival.** Pooled ResNet/ImageNet/CLIP
  features put every density model at chance on real swaps. Artifact signal is
  local and low-level. (Reproduced at scale on FF++: the SRM arm, 0.624.)
- **The `fit_leaves(jitter=…)` product-of-marginals collapse** in
  `src/bak/allinone_probabilistic_circuits.py`: only scalar-`mu` leaves get
  jitter, so `GaussianMixtureLeaf` starts with K identical sibling subtrees and
  the circuit silently degenerates. Detector: learned and random vtrees give
  byte-identical NLL. The tensorized `pcdf/circuits/einsum_pc.py` used
  throughout the current work does **not** have this bug (`weight_jitter` is
  applied per component and `tests/test_equivalence.py` pins it), but the old
  library in `src/bak/` still does.
