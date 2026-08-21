"""
Where do the 0.136 AUC between our SBI encoder and the published one live?

THE MEASUREMENT THAT MOTIVATES THIS
-----------------------------------
The gap to a state-of-the-art detector is almost entirely upstream of the
circuit.  Decomposed on FF++ c23, video AUC:

    published SBI                     ~0.996
    our encoder, end to end            0.860     <- 0.136 lost here
    linear probe on its projected
      patch features (1024 dims)       0.859     <- 0.001 lost in projection
    circuit, one-class                 0.812
    circuit, exact likelihood ratio    0.828     <- 0.031 lost by density scoring

So the representation is not the bottleneck the project assumed it was (the
probe on 16 channels x 64 patches recovers everything the encoder itself has),
and neither is the projection width.  The encoder is ~80% of the shortfall, and
this script attributes it.

VARIANTS (each differs from `base` in one thing, except the last)
----------------------------------------------------------------
base      the recipe the project has been using
noleak    pristine background + symmetric compression.  `scripts/shortcut_audit`
          measured that real-vs-self-blend is decidable at AUC 0.95 from JPEG
          statistics in blocks the blend never touched, while the same features
          on real FF++ forgeries score 0.505 — the pseudo-task hands over a
          global cue that does not exist in real forgeries, and the network can
          drive its loss to 0.008 on it without learning forgery evidence.
sam       SAM instead of AdamW.  SBI trains with it, we did not.  It matters
          here specifically because the pseudo-task is saturated: what
          transfers is decided by the flatness of the solution, not the loss.
hull      randomised hull type, so one boundary shape cannot be memorised.
all       noleak + sam + hull.

Kept deliberately short (fewer epochs, fewer frames per video) so that five
variants rank in one session; the winner is then rerun at full length.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdf.cli import load_config  # noqa: E402
from pcdf.device import resolve_device  # noqa: E402
from pcdf.models.supervised import SbiConfig, train_sbi  # noqa: E402


VARIANTS = {
    "base":   {},
    # `symmetric` re-encodes every TRAINING image while val/test stay untouched,
    # so it closes the leak but introduces a train/test compression shift of its
    # own — a confound, not a control.  `noleak_clean` removes the extra JPEG
    # from both classes instead; the leak sweep puts both recipes at exactly
    # 0.500, so this is the same experiment without the side effect, and it is
    # the one to read.
    "noleak": {"pristine_background": True, "compress_policy": "symmetric"},
    "noleak_clean": {"pristine_background": True, "compress_policy": "none"},
    "sam":    {"optimizer": "sam"},
    "hull":   {"hull_variety": True},
    # kept as first run so its result stays comparable; carries the same
    # `symmetric` confound as `noleak`
    "all":    {"pristine_background": True, "compress_policy": "symmetric",
               "optimizer": "sam", "hull_variety": True},
    "all_clean": {"pristine_background": True, "compress_policy": "none",
                  "optimizer": "sam", "hull_variety": True},
    # The last untested hypothesis, and after the others failed, the leading
    # one.  Every crop the encoder has ever seen was stored at 256px JPEG q95
    # with 4:2:0 chroma subsampling and then UPSAMPLED to 380 — two lossy steps,
    # both of which attack exactly the high-frequency and colour detail a
    # blending artifact lives in, applied before the network sees anything.
    # `crops_hires` is the same crops at native 380px, q100, 4:4:4.
    "hires":     {"crops_dirname": "crops_hires"},
    "hires_sam": {"crops_dirname": "crops_hires", "optimizer": "sam"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/ffpp_sbi.yaml")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--frames-per-video", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config, [])
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    root = Path(cfg["root"])
    out_path = Path(a.out or root / "results" / "encoder_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name in a.variants:
        if name in results:
            print(f"[ablate-enc] {name} already done, skipping", flush=True)
            continue
        overrides = VARIANTS[name]
        scfg = SbiConfig(device=cfg["device"], seed=cfg["seed"],
                         epochs=a.epochs, batch_size=a.batch_size,
                         max_frames_per_video=a.frames_per_video,
                         workers=a.workers, tag=f"sbi_ab_{name}",
                         **overrides)
        print(f"\n{'=' * 70}\n[ablate-enc] {name}: {overrides or 'control'}\n"
              f"{'=' * 70}", flush=True)
        t0 = time.time()
        ckpt = train_sbi(cfg, scfg)

        import torch

        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        hist = blob.get("history", [])
        results[name] = {
            "overrides": overrides,
            "best_val_auc_video": blob.get("best_val_auc_video"),
            "best_epoch": int(max(range(len(hist)),
                                  key=lambda i: hist[i]["val_auc_video"]))
                          if hist else None,
            "final_train_loss": hist[-1]["loss"] if hist else None,
            "epochs_run": len(hist),
            "minutes": (time.time() - t0) / 60,
            "checkpoint": str(ckpt),
            "history": hist,
        }
        out_path.write_text(json.dumps(results, indent=2, default=float))
        print(f"[ablate-enc] {name}: best val video AUC "
              f"{results[name]['best_val_auc_video']:.4f} "
              f"(epoch {results[name]['best_epoch']}, "
              f"train loss {results[name]['final_train_loss']:.4f}, "
              f"{results[name]['minutes']:.0f} min)", flush=True)

    print("\n=== SUMMARY ===")
    base = results.get("base", {}).get("best_val_auc_video")
    for name, r in results.items():
        d = (f"{r['best_val_auc_video'] - base:+.4f}"
             if base and name != "base" else "  —   ")
        print(f"{name:>8}  val video AUC {r['best_val_auc_video']:.4f}  {d}  "
              f"train loss {r['final_train_loss']:.4f}")
    print(f"\n[ablate-enc] wrote {out_path}")


if __name__ == "__main__":
    main()
