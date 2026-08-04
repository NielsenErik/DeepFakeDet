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
