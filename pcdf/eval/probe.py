"""
Representation diagnostic: how much forgery signal survives the projection AT ALL?

This is the experiment that decides where a failure lives.  A one-class density
model can fail for two completely different reasons:

  (a) the features carry the signal, but the density model cannot exploit it
      (fakes sit in HIGH-density regions — the likelihood-OOD pathology), or
  (b) the features do not carry the signal at all, in which case no density
      model, ours or anyone's, can ever work and the fix is upstream.

A supervised linear probe separates the two.  It is trained on the VALIDATION
split (which has both classes) and evaluated on TEST, so it never sees test
data, and it is the *upper bound* on what any method could extract from these
exact coordinates.  Reading:

    probe AUC ≈ 0.5    -> (b): the projection destroyed the artifact signal.
                          Fix the representation: more dims, a different layer,
                          no patch pooling, or a variance criterion that is not
                          PCA (artifacts live in LOW-variance directions).
    probe AUC >> PC AUC -> (a): the signal is there and the one-class score is
                          the bottleneck — which is a research finding about
                          density-based detection, not a bug.

Both outcomes are publishable; confusing them is not.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _pooled(Z: np.ndarray, how: str) -> np.ndarray:
    if how == "mean":
        return Z.mean(1)
    if how == "flat":
        return Z.reshape(len(Z), -1)
    raise KeyError(how)


def run_probe(cfg: Dict, max_train: int = 40000, seed: int = 0) -> Dict:
    """
    Linear probes on the projected features, at three granularities:

      image-mean   the global summary (what the POC used, expected to fail)
      image-flat   every patch × channel as its own coordinate — a linear model
                   over the same input the circuit sees
      patch-level  one probe over patch vectors, image score = max over patches;
                   this is the granularity the artifact actually lives at
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from ..stages import _load_split
    from .metrics import video_keys, video_level

    Ztr, itr = _load_split(cfg, "val")            # both classes, never test
    Zte, ite = _load_split(cfg, "test")
    rng = np.random.default_rng(seed)
    if len(Ztr) > max_train:
        sel = np.sort(rng.choice(len(Ztr), max_train, replace=False))
        Ztr, itr = Ztr[sel], {k: v[sel] for k, v in itr.items()}

    ytr, yte = itr["label"].astype(int), ite["label"].astype(int)
    vkey = video_keys(ite)
    out: Dict[str, Dict[str, float]] = {}

    for how in ("mean", "flat"):
        Xtr, Xte = _pooled(Ztr, how), _pooled(Zte, how)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        s = clf.decision_function(Xte)
        vs, vy = video_level(s, vkey, yte)
        out[f"image_{how}"] = {"auc_frame": float(roc_auc_score(yte, s)),
                               "auc_video": float(roc_auc_score(vy, vs)),
                               "n_features": int(Xtr.shape[1])}

    # patch-level probe: every patch of every image is a training example
    P, c = Ztr.shape[1], Ztr.shape[2]
    Xp = Ztr.reshape(-1, c)
    yp = np.repeat(ytr, P)
    keep = rng.choice(len(Xp), min(len(Xp), max_train * 4), replace=False)
    clf = LogisticRegression(max_iter=2000).fit(Xp[keep], yp[keep])
    sp = clf.decision_function(Zte.reshape(-1, c)).reshape(len(Zte), P)
    for agg, s in (("max", sp.max(1)), ("mean", sp.mean(1))):
        vs, vy = video_level(s, vkey, yte)
        out[f"patch_{agg}"] = {"auc_frame": float(roc_auc_score(yte, s)),
                               "auc_video": float(roc_auc_score(vy, vs)),
                               "n_features": int(c)}

    # per-method breakdown of the strongest probe, so a single average cannot
    # hide that one manipulation carries everything
    best = max(out, key=lambda k: out[k]["auc_video"])
    per_method = {}
    if best.startswith("image"):
        Xte_b = _pooled(Zte, best.split("_")[1])
        clf = LogisticRegression(max_iter=2000).fit(_pooled(Ztr, best.split("_")[1]), ytr)
        s_best = clf.decision_function(Xte_b)
    else:
        s_best = sp.max(1) if best.endswith("max") else sp.mean(1)
    for meth in sorted(set(ite["method"][yte == 1].tolist())):
        m = (yte == 0) | (ite["method"] == meth)
        vs, vy = video_level(s_best[m], vkey[m], yte[m])
        per_method[meth] = float(roc_auc_score(vy, vs))

    result = {"probes": out, "best": best, "per_method": per_method,
              "interpretation": _interpret(out[best]["auc_video"])}
    return result


def _interpret(auc: float) -> str:
    if auc < 0.6:
        return ("REPRESENTATION FAILURE: even a supervised linear probe cannot "
                "separate real from fake in these coordinates, so no one-class "
                "score could. Fix the features (more dims / different layer / "
                "no patch pooling / non-PCA reduction) before touching the model.")
    if auc < 0.8:
        return ("WEAK REPRESENTATION: some signal survives but not much; the "
                "one-class gap to the probe is the interesting quantity.")
    return ("SIGNAL IS PRESENT: the features separate the classes under "
            "supervision, so a one-class model failing here is a statement "
            "about density-based detection, not about the representation.")


def cmd_probe(cfg: Dict, args) -> None:
    res = run_probe(cfg, max_train=args.max_train, seed=cfg["seed"])
    from ..stages import _tag

    out = Path(cfg["root"]) / "results" / _tag(cfg) / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    print(f"[probe] wrote {out}")
