#!/bin/bash
# Queued behind the encoder ablation: per-family pseudo-forgery features, then
# the exact family-mixture ratio (domain gap + mechanism posterior + C5).
#
# The blends are extracted with `--pristine-background`, which is not cosmetic:
# scripts/shortcut_audit.py measured that without it a classifier reading only
# blocks OUTSIDE the blend mask separates real from self-blend at AUC 0.94,
# while the same features on real FF++ forgeries score 0.48.  p_blend was
# therefore partly modelling a global compression signature that no real
# forgery carries.
set -u
cd ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
PY=~/miniconda3/envs/expllm_env/bin/python
CFG=configs/ffpp_sbi.yaml

# wait for the ablation to release the GPU
until grep -q "=== SUMMARY ===" ~/deepfake_data/logs/encoder_ablation.log 2>/dev/null; do
  sleep 120
done
echo "=== encoder ablation done, starting mixture pipeline ==="
date

for FAM in blend render overshoot statistical; do
  echo "=== features: pseudo-family $FAM ==="
  $PY -u -m pcdf.cli -c $CFG features --dataset ffpp \
      --pseudo-family $FAM --pristine-background || exit 1
done

echo "=== family mixture experiment ==="
$PY -u -m pcdf.cli -c $CFG probe || true
$PY -u scripts/family_mixture_experiment.py -c $CFG || exit 1
echo "=== done ==="
date
