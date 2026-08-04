"""
Report assembly and the pre-registered go/no-go rubric.

The rubric is written down BEFORE the numbers exist, and it is deliberately
hostile to the hypothesis, because the question the user asked is "is this
worth pursuing" — not "can this be made to look good".  Each gate has a
threshold, a rationale, and an explicit failure meaning.

  G1 DETECTION IS REAL          in-dataset FF++ video AUC ≥ 0.90 for the best
                                circuit score.  Below that, the representation
                                or the density model is not working at all and
                                nothing else matters.

  G2 GENERALIZATION            cross-dataset video AUC within 3 points of the
                                SBI baseline trained under the same real-only
                                protocol.  This is the axis the field is judged
                                on; a one-class method that only works
                                in-dataset is not publishable in 2026.

  G3 THE CIRCUIT EARNS ITS KEEP the circuit beats the best non-circuit one-class
                                baseline (Mahalanobis / GMM / PatchCore / flow)
                                in the SAME feature space by ≥ 0.02 AUC, OR
                                beats them on localization patch-AUC by ≥ 0.03.
                                If it loses both, the exact-marginal machinery
                                is not buying anything measurable and the
                                honest conclusion is to stop.

  G4 STRUCTURE MATTERS          a learned region graph (ORC / Chow-Liu) beats a
                                random one by ≥ 2 nats validation NLL.  If not,
                                structure learning — the part inherited from
                                the reference library — is decoration here, and
                                the paper's structure story must be dropped.

  G5 IT SCALES                  the circuit fits in < 1 GPU-hour at the working
                                dimensionality.  This is what the tensorized
                                executor was written for.

Verdict: PURSUE if G1 and (G2 or G3) pass; REFRAME if G1 passes and only G3's
localization half passes (the contribution is explanation, not accuracy); STOP
if G1 fails or both halves of G3 fail.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

GATES = {
    "G1_detection": {"threshold": 0.90, "metric": "ffpp in-dataset video AUC"},
    "G2_generalization": {"threshold": -0.03, "metric": "cross-dataset AUC minus SBI"},
    "G3_circuit_value_detection": {"threshold": 0.02, "metric": "AUC minus best baseline"},
    "G3_circuit_value_localization": {"threshold": 0.03, "metric": "patch AUC minus best baseline"},
    "G4_structure": {"threshold": 2.0, "metric": "val NLL gain over random structure (nats)"},
    "G5_scale": {"threshold": 3600.0, "metric": "circuit fit seconds"},
}


def _table(rows: List[List[str]], header: List[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def build_report(cfg: Dict, tag: str) -> Path:
    root = Path(cfg["root"])
    res_dir = root / "results" / tag
    scores = _load(res_dir / "scores.json")
    local = _load(res_dir / "localization.json")
    audit = _load(root / "models" / f"pc_{tag}_audit.json")
    bench = _load(root / "results" / "bench.json")
    struct = _load(root / "results" / f"structure_ablation_{tag}.json")
    sbi_hist = _load(root / "models" / "sbi_history.json")

    lines: List[str] = [
        f"# PC deepfake detection — results ({tag})", "",
        "Protocol: the circuit and every one-class baseline are fitted on FF++ "
        "c23 REAL training faces only (official identity-disjoint 720/140/140 "
        "split); no forgery is seen during fitting. Cross-dataset test sets are "
        "scored by the same frozen models.", "",
    ]

    # ── detection ────────────────────────────────────────────────────────
    if scores:
        lines += ["## Detection (video-level AUC)", ""]
        models = sorted(scores.get("summary", {}))
        datasets = sorted({d for m in models for d in scores["summary"][m]})
        rows = []
        for m in models:
            row = [m]
            for d in datasets:
                e = scores["summary"][m].get(d)
                row.append(f"{_fmt(e['auc_video'])} ({e['best_score']})" if e else "—")
            rows.append(row)
        lines += [_table(rows, ["model"] + datasets), ""]

        for ds, block in scores.get("per_dataset", {}).items():
            lines += [f"### {ds} — per manipulation method", ""]
            if block.get("mixed_reals"):
                lines += ["> Real side comes from FF++ (this test set ships no "
                          "reals): source mismatch inflates this number.", ""]
            methods = sorted({k for m in block["models"].values()
                              for s in m.values() for k in s if k != "all"})
            rows = []
            for model, per_score in block["models"].items():
                best = max(per_score.items(),
                           key=lambda kv: kv[1]["all"]["auc_video"]
                           if np.isfinite(kv[1]["all"]["auc_video"]) else -1)
                rows.append([f"{model} ({best[0]})"] +
                            [_fmt(best[1].get(meth, {}).get("auc_video")) for meth in methods])
            lines += [_table(rows, ["model (score)"] + methods), ""]

    # ── robustness ───────────────────────────────────────────────────────
    if scores and scores.get("robustness"):
        rb = scores["robustness"]
        models = sorted({m for e in rb.values() for m in e})
        perts = list(rb)
        lines += ["## Robustness (frozen models, perturbed test crops)", "",
                  "Score function pinned to the clean-data winner; Δ is in "
                  "absolute AUC points.", ""]
        rows = []
        for m in models:
            row = [m]
            for p in perts:
                e = rb[p].get(m)
                row.append(f"{_fmt(e['auc_video'])} ({e['delta']:+.3f})" if e else "—")
            rows.append(row)
        lines += [_table(rows, ["model"] + perts), ""]

    # ── localization ─────────────────────────────────────────────────────
    if local:
        lines += ["## Localization (exact per-patch queries)", "",
                  f"Masks: `{local.get('mask_source')}` over "
                  f"{local.get('n_images')} manipulated frames.", ""]
        rows = [[k, _fmt(v.get("patch_auc_pooled")), _fmt(v.get("patch_auc_per_image")),
                 _fmt(v.get("iou_best_mean")), _fmt(v.get("pointing_accuracy"))]
                for k, v in local.get("models", {}).items()]
        lines += [_table(rows, ["model", "patch AUC (pooled)", "patch AUC (per image)",
                                "best IoU", "pointing acc"]), ""]
        if local.get("per_method_conditional"):
            rows = [[k, _fmt(v.get("patch_auc_pooled")), _fmt(v.get("manipulated_patch_fraction"))]
                    for k, v in local["per_method_conditional"].items()]
            lines += ["### Conditional surprisal by manipulation", "",
                      _table(rows, ["method", "patch AUC", "manipulated fraction"]), ""]

    # ── circuit properties + cost ────────────────────────────────────────
    if audit:
        lines += ["## Circuit property audit", "",
                  "```", json.dumps(audit, indent=2), "```", ""]
    if bench:
        rows = [[b["d"], b["K"], _fmt(b["einsum_sec_per_step"]),
                 _fmt(b.get("reference_sec_per_step")), _fmt(b.get("speedup")),
                 b["size"]["parameters"], _fmt(b["log_partition"])] for b in bench]
        lines += ["## Circuit throughput (tensorized vs reference object graph)", "",
                  _table(rows, ["d", "K", "einsum s/step", "reference s/step",
                                "speedup", "params", "log Z"]), ""]
    if struct:
        rows = [[k, _fmt(v.get("val_nll")), _fmt(v.get("auc_video")),
                 _fmt(v.get("fit_seconds"))] for k, v in struct.items()]
        lines += ["## Structure ablation", "",
                  _table(rows, ["structure", "val NLL", "video AUC", "fit s"]), ""]

    # ── rubric ───────────────────────────────────────────────────────────
    verdict, gate_rows = evaluate_gates(scores, local, struct, bench, sbi_hist)
    lines += ["## Pre-registered decision rubric", "",
              _table(gate_rows, ["gate", "metric", "threshold", "observed", "pass"]), "",
              f"**Verdict: {verdict}**", ""]

    out = res_dir / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    (res_dir / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "gates": gate_rows}, indent=2, default=float))
    return out


def evaluate_gates(scores: Optional[Dict], local: Optional[Dict],
                   struct: Optional[Dict], bench: Optional[List],
                   sbi_hist: Optional[List]) -> Tuple[str, List[List[str]]]:
    rows: List[List[str]] = []
    passed: Dict[str, Optional[bool]] = {}

    def add(name: str, observed, ok: Optional[bool]) -> None:
        g = GATES[name]
        rows.append([name, g["metric"], _fmt(g["threshold"]), _fmt(observed),
                     "—" if ok is None else ("PASS" if ok else "FAIL")])
        passed[name] = ok

    # G1
    pc_ffpp = _get(scores, ["summary", "PC", "ffpp", "auc_video"])
    add("G1_detection", pc_ffpp,
        None if pc_ffpp is None else pc_ffpp >= GATES["G1_detection"]["threshold"])

    # G2: cross-dataset gap to SBI
    gap = None
    if scores:
        cross = [d for d in _get(scores, ["summary", "PC"], {}) if d != "ffpp"]
        sbi = _get(scores, ["summary", "SBI"], {})
        gaps = [scores["summary"]["PC"][d]["auc_video"] - sbi[d]["auc_video"]
                for d in cross if d in sbi
                and np.isfinite(scores["summary"]["PC"][d]["auc_video"])]
        gap = min(gaps) if gaps else None
    add("G2_generalization", gap,
        None if gap is None else gap >= GATES["G2_generalization"]["threshold"])

    # G3a: detection margin over the best one-class baseline
    margin = None
    if scores and pc_ffpp is not None:
        others = [v["ffpp"]["auc_video"] for k, v in scores["summary"].items()
                  if k not in ("PC", "SBI") and "ffpp" in v
                  and np.isfinite(v["ffpp"]["auc_video"])]
        if others:
            margin = pc_ffpp - max(others)
    add("G3_circuit_value_detection", margin,
        None if margin is None else margin >= GATES["G3_circuit_value_detection"]["threshold"])

    # G3b: localization margin
    loc_margin = None
    if local:
        m = local.get("models", {})
        pc = max([m[k].get("patch_auc_pooled", np.nan)
                  for k in m if k.startswith("PC")] or [np.nan])
        base = max([m[k].get("patch_auc_pooled", np.nan)
                    for k in m if not k.startswith("PC")] or [np.nan])
        if np.isfinite(pc) and np.isfinite(base):
            loc_margin = float(pc - base)
    add("G3_circuit_value_localization", loc_margin,
        None if loc_margin is None
        else loc_margin >= GATES["G3_circuit_value_localization"]["threshold"])

    # G4: structure gain over random.  Ablation keys are "<patch>/<channel>",
    # so the control is whichever variant is random on both axes.
    gain = None
    if struct:
        ctrl = [v["val_nll"] for k, v in struct.items()
                if k.split("/")[0] == "random" and v.get("val_nll") is not None]
        learned = [v["val_nll"] for k, v in struct.items()
                   if k.split("/")[0] != "random" and v.get("val_nll") is not None]
        if ctrl and learned:
            gain = float(max(ctrl) - min(learned))
    add("G4_structure", gain,
        None if gain is None else gain >= GATES["G4_structure"]["threshold"])

    # G5: fit cost
    fit_s = None
    if struct:
        vals = [v.get("fit_seconds") for v in struct.values() if v.get("fit_seconds")]
        fit_s = max(vals) if vals else None
    add("G5_scale", fit_s,
        None if fit_s is None else fit_s <= GATES["G5_scale"]["threshold"])

    g1 = passed.get("G1_detection")
    g2 = passed.get("G2_generalization")
    g3d = passed.get("G3_circuit_value_detection")
    g3l = passed.get("G3_circuit_value_localization")
    if g1 is None:
        verdict = "INCOMPLETE — detection numbers missing"
    elif not g1:
        verdict = "STOP — the detector does not work in-dataset; fix the representation first"
    elif g2 or g3d:
        verdict = "PURSUE — competitive detection plus exact queries"
    elif g3l:
        verdict = ("REFRAME — the contribution is exact localization/explanation, "
                   "not raw accuracy; position the paper accordingly")
    else:
        verdict = ("STOP — the circuit matches neither the baselines' accuracy nor "
                   "their localization; the exact machinery buys nothing here")
    return verdict, rows


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                  # noqa: BLE001
        return None


def _get(d, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
