#!/bin/bash
# Last in the chain: the crop-resolution hypothesis.
#
# Every other recipe-level explanation for the encoder's 0.136 shortfall has now
# been tested and failed (SAM +0.003, leak-free -0.025, hull -0.011, all -0.017).
# What has never been varied is the input itself: every crop the encoder has
# seen was stored at 256px JPEG q95 with 4:2:0 chroma subsampling and then
# upsampled to 380.  `crops_hires` is the same faces, same detector, same
# margin, at native 380px q100 4:4:4 — one variable, the one nobody changed.
set -u
cd ~/Documents/Unitn/PhD/Main-Project/GitHub/DeepFakeDet
PY=~/miniconda3/envs/expllm_env/bin/python

until grep -q "^=== ablation2 done ===" ~/deepfake_data/logs/queue_ablation2.log 2>/dev/null; do
  sleep 120
done
until grep -q "^=== hires ingest done ===" ~/deepfake_data/logs/ingest_hires.log 2>/dev/null; do
  sleep 60
done
echo "=== high-resolution crop variants ==="
date
$PY -u scripts/encoder_ablation.py --epochs 20 --frames-per-video 16 --workers 8 \
    --variants hires hires_sam
echo "=== hires variants done ==="
$PY scripts/show_ablation.py
date
