# Status — 2026-08-04

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
