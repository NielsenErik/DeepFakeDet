#!/bin/bash
# Chained behind the mixture pipeline: the gap waterfall, measured on exactly
# the crops every other stage is measured on.
set -u
cd ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
PY=~/miniconda3/envs/expllm_env/bin/python

until grep -q "^=== done ===" ~/deepfake_data/logs/queue_mixture.log 2>/dev/null; do
  sleep 120
done
echo "=== mixture pipeline done, running gap waterfall ==="
date
$PY -u scripts/gap_waterfall.py -c configs/ffpp_sbi.yaml

# The leakage finding makes a falsifiable prediction about the hybrid sweep.
# With the leaky blends the discriminative term had NO gradient left — BCE
# 0.0000, real-vs-blend AUC 0.9996 — so lambda did nothing (0.8274 -> 0.8214
# across the whole range).  ~94% of that separability was a global compression
# cue.  On leak-free blends the pseudo-task should be genuinely harder, the
# discriminative term should have gradient again, and lambda should MOVE.  If it
# still does not, the saturation was never the reason and that is worth knowing
# just as much.
echo "=== hybrid sweep on leak-free blends ==="
$PY -u scripts/hybrid_sweep.py --config configs/ffpp_sbi.yaml \
    --blend-suffix blend-blendP --out-name hybrid_sweep_noleak.json \
    --epochs 40
echo "=== waterfall done ==="
date
