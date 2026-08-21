#!/bin/bash
# Second ablation pass, chained last: the two variants that close the leak
# WITHOUT the train-only compression shift that `symmetric` introduces.
# encoder_ablation.py skips variants already present in the results file, so
# this runs only `noleak_clean` and `all_clean`.
set -u
cd ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
PY=~/miniconda3/envs/expllm_env/bin/python

until grep -q "^=== waterfall done ===" ~/deepfake_data/logs/queue_waterfall.log 2>/dev/null; do
  sleep 120
done
echo "=== running clean leak-free encoder variants ==="
date
$PY -u scripts/encoder_ablation.py --epochs 20 --frames-per-video 16 --workers 8 \
    --variants noleak_clean all_clean
echo "=== ablation2 done ==="
date
