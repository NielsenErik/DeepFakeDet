#!/usr/bin/env bash
# Full experiment pipeline on the RTX 4080 workstation.
#
# Stages are resumable and each one skips work that already exists on disk —
# with ONE exception noted below.  Run under tmux; the SBI arm takes hours.
#
#   scripts/run_all.sh clip      # frozen CLIP patch tokens (pure one-class arm)
#   scripts/run_all.sh sbi       # SBI-tuned encoder (the arm aiming at SotA)
#   scripts/run_all.sh both
set -euo pipefail

PY=${PY:-$HOME/miniconda3/envs/expllm_env/bin/python}
ROOT=${ROOT:-$HOME/deepfake_data}
ARM=${1:-clip}
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

run() { echo -e "\n=== $* ==="; "$PY" -u -m pcdf.cli "$@"; }

prepare() {
  # one-off: manifests + face crops.  ~40 min for FF++ with data.workers=8.
  [ -f "$ROOT/manifests/ffpp_ingested.csv" ] || {
    run -c configs/ffpp_clip.yaml manifest --datasets ffpp
    run -c configs/ffpp_clip.yaml ingest --dataset ffpp --masks
  }
}

arm() {
  local cfg=$1 backbone=$2
  # A PARTIAL features run must never be resumed: the projector is written
  # first and reused if present, and finished splits are skipped, so an
  # interrupted run silently poisons everything downstream.  Delete and redo.
  if [ ! -f "$ROOT/features/$backbone/ffpp_test_index.json" ]; then
    rm -rf "${ROOT:?}/features/$backbone"
    run -c "$cfg" features --dataset ffpp
  fi
  run -c "$cfg" fit-pc
  run -c "$cfg" baselines
  run -c "$cfg" evaluate --datasets ffpp
  run -c "$cfg" explain --n-images 2000
  run -c "$cfg" ablate-structure --epochs 25
  run -c "$cfg" report

  # robustness: test-time perturbations only, models stay frozen
  for p in jpeg70 jpeg50 resize0.5 blur; do
    run -c "$cfg" features --dataset ffpp --perturb "$p"
  done
  run -c "$cfg" evaluate --datasets ffpp --robustness
  run -c "$cfg" report
}

prepare
run -c configs/ffpp_clip.yaml bench --with-reference

case "$ARM" in
  clip) arm configs/ffpp_clip.yaml clip ;;
  sbi)
    [ -f "$ROOT/models/sbi_effnetb4.pt" ] || \
      run -c configs/ffpp_sbi.yaml train-sbi --epochs 30
    arm configs/ffpp_sbi.yaml sbi
    ;;
  both)
    arm configs/ffpp_clip.yaml clip
    [ -f "$ROOT/models/sbi_effnetb4.pt" ] || \
      run -c configs/ffpp_sbi.yaml train-sbi --epochs 30
    arm configs/ffpp_sbi.yaml sbi
    ;;
  *) echo "usage: $0 {clip|sbi|both}" >&2; exit 2 ;;
esac

echo -e "\nReports:"
ls -1 "$ROOT"/results/*/REPORT.md
