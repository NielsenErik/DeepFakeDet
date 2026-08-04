"""
Does a MORE EXPRESSIVE circuit fix the detection gap, or is the statistic wrong?

Runs, on the same features and the same selected scope:

  monotone PC     EinsumPC (the tensorized workhorse)
  squared PC      SquaredPC from the reference library — subtractive mixtures
                  with real-valued sum weights, normalized exactly via the
                  squared-circuit pairwise construction (Loconte et al., AAAI
                  2025).  Strictly more expressive at equal size.
  ratio           two monotone circuits, p_blend / p_real

crossed with the structure learners the project cares about (Chow-Liu, ORC,
Forman, spectral, random).

The scope is restricted to the top-k coordinates ranked on the VALIDATION split,
for two reasons: SquaredPC is an object-graph implementation and does not scale
to d = 6080, and the diagnosis already showed that the full joint dilutes the
signal — so this is also the cleanest setting in which to ask whether
expressiveness or the score function is the binding constraint.

    python scripts/sos_experiment.py --config configs/ffpp_spectral.yaml --top-k 128
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdf.circuits.einsum_pc import EinsumPC  # noqa: E402
from pcdf.circuits.structure import build_structure  # noqa: E402
from pcdf.cli import load_config  # noqa: E402
from pcdf.eval.diagnose import discriminative_coordinates  # noqa: E402
from pcdf.eval.metrics import video_keys, video_level  # noqa: E402
from pcdf.pclib import GaussianMixtureLeaf, SquaredPC  # noqa: E402
from pcdf.stages import _load_split  # noqa: E402


def fit_einsum(rg, Xtr, Xval, K, epochs, dev, seed=0):
    pc = EinsumPC(rg, n_sum_components=K, n_input_components=K,
                  leaf_components=4, seed=seed).to(dev)
    pc.fit_leaves(Xtr[:20000], seed=seed)
    opt = torch.optim.Adam(pc.parameters(), lr=5e-3)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            xb = Xtr[perm[i:i + 256]].to(dev)
            opt.zero_grad(set_to_none=True)
            (-pc.log_prob(xb).mean()).backward()
            opt.step()
        with torch.no_grad():
            v = float(-pc.log_prob(Xval.to(dev), chunk=256).mean())
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in pc.state_dict().items()}
    pc.load_state_dict(best_state)
    pc.eval()
    return pc, best


def fit_squared(rg, Xtr, Xval, K, epochs, seed=0):
    """SquaredPC is the reference object-graph implementation: CPU, small d."""
    pc = SquaredPC(rg, n_sum_components=K,
                   leaf_factory=lambda i: GaussianMixtureLeaf(i, 4), seed=seed)
    pc.fit_leaves(Xtr[:4000], jitter=0.2)
    opt = torch.optim.Adam(pc.parameters(), lr=5e-3)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))[:4096]
        for i in range(0, len(perm), 512):
            xb = Xtr[perm[i:i + 512]]
            opt.zero_grad(set_to_none=True)
            (-pc.log_prob(xb).mean()).backward()
            opt.step()
        with torch.no_grad():
            v = float(-pc.log_prob(Xval[:2000]).mean())
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in pc.state_dict().items()}
    pc.load_state_dict(best_state)
    return pc, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ffpp_spectral.yaml")
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--sos-epochs", type=int, default=15)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=8000)
    ap.add_argument("--structures", nargs="*",
                    default=["chow_liu", "orc", "forman", "spectral", "random"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    from pcdf.device import resolve_device

    dev = cfg["device"] = resolve_device(cfg.get("device", "auto"))
    rng = np.random.default_rng(cfg["seed"])

    Ztr, _ = _load_split(cfg, "train", label=0)
    Zval, ival = _load_split(cfg, "val")
    Zte, ite = _load_split(cfg, "test")
    if len(Zte) > args.n_test:
        sel = np.sort(rng.choice(len(Zte), args.n_test, replace=False))
        Zte, ite = Zte[sel], {k: v[sel] for k, v in ite.items()}

    keep = np.sort(discriminative_coordinates(Zval, ival["label"].astype(int),
                                              args.top_k))
    print(f"[sos] scope: {len(keep)} of {Ztr.shape[1] * Ztr.shape[2]} coordinates, "
          f"selected on VAL only")

    flat = lambda Z: Z.reshape(len(Z), -1)[:, keep].astype(np.float32)
    Xtr = torch.from_numpy(flat(Ztr))
    Zval_real = Zval[ival["label"] == 0]
    Xval = torch.from_numpy(flat(Zval_real)[:4000])
    Xte = torch.from_numpy(flat(Zte))
    y = ite["label"].astype(int)
    vkey = video_keys(ite)

    # blended training set for the ratio arm, if it exists
    Xbl = None
    try:
        Zbl, _ = _load_split(cfg, "train", label=0, perturb="blend")
        Xbl = torch.from_numpy(flat(Zbl))
        print(f"[sos] self-blended training set available: {len(Xbl)}")
    except FileNotFoundError:
        print("[sos] no self-blended features; ratio arm skipped")

    from sklearn.metrics import roc_auc_score

    def auc_of(scores: np.ndarray) -> float:
        vs, vy = video_level(scores, vkey, y)
        return float(roc_auc_score(vy, vs))

    results = {}
    for method in args.structures:
        t0 = time.time()
        rg = build_structure(flat(Ztr)[:8000], method=method, seed=cfg["seed"])
        entry = {"structure_seconds": time.time() - t0}

        pc, nll = fit_einsum(rg, Xtr, Xval, args.K, args.epochs, dev)
        with torch.no_grad():
            lp_real = pc.log_prob(Xte.to(dev), chunk=256).cpu().numpy()
        entry["monotone"] = {"val_nll": nll, "auc_nll": auc_of(-lp_real),
                             "auc_nll_inverted": auc_of(lp_real)}

        if Xbl is not None:
            pcb, nllb = fit_einsum(rg, Xbl, Xval, args.K, args.epochs, dev, seed=1)
            with torch.no_grad():
                lp_bl = pcb.log_prob(Xte.to(dev), chunk=256).cpu().numpy()
            entry["ratio"] = {"val_nll_blend": nllb,
                              "auc_ratio": auc_of(lp_bl - lp_real)}

        try:
            sq, snll = fit_squared(rg, Xtr, Xval, max(2, args.K // 2), args.sos_epochs)
            with torch.no_grad():
                s_lp = sq.log_prob(Xte[:4000]).cpu().numpy()
            entry["squared"] = {
                "val_nll": snll,
                "auc_nll": auc_of_subset(s_lp, y[:4000], vkey[:4000], invert=False),
                "auc_nll_inverted": auc_of_subset(s_lp, y[:4000], vkey[:4000], invert=True),
                "log_partition": float(sq.log_partition()),
            }
        except Exception as exc:                      # noqa: BLE001
            entry["squared"] = {"error": f"{type(exc).__name__}: {exc}"}

        results[method] = entry
        print(f"[sos] {method}: {json.dumps(entry, default=float)}", flush=True)

    out = Path(args.out or (Path(cfg["root"]) / "results" / "sos_experiment.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[sos] wrote {out}")
    return 0


def auc_of_subset(scores, y, vkey, invert: bool) -> float:
    from sklearn.metrics import roc_auc_score

    s = scores if invert else -scores
    vs, vy = video_level(s, vkey, y)
    return float(roc_auc_score(vy, vs))


if __name__ == "__main__":
    sys.exit(main())
