"""Print the encoder ablation as a table."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1
         else Path.home() / "deepfake_data/results/encoder_ablation.json")
d = json.load(open(p))
base = d.get("base", {}).get("best_val_auc_video")

print(f"{'variant':>14} {'val AUC':>8} {'vs base':>8} {'loss':>7} {'ep':>3} {'min':>4}  overrides")
for k, v in d.items():
    delta = "    —   " if k == "base" or base is None else f"{v['best_val_auc_video'] - base:+8.4f}"
    ov = ", ".join(f"{a}={b}" for a, b in v["overrides"].items()) or "control"
    print(f"{k:>14} {v['best_val_auc_video']:8.4f} {delta} "
          f"{v['final_train_loss']:7.4f} {v['best_epoch']:3d} {v['minutes']:4.0f}  {ov}")
