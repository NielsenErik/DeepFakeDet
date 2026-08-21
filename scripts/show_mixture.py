"""Print the family-mixture experiment as readable tables."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else
         Path.home() / "deepfake_data/results/sbi_g8c16_kd-orc_K8/family_mixture.json")
d = json.load(open(p))
fams = d["families"]
r3 = lambda m: {k: round(v, 3) for k, v in m.items()}  # noqa: E731

print("=== Q2  DETECTION ===")
for k, v in d["Q2_detection"]["mixture"].items():
    print("  %-22s video %.4f  frame %.4f" % (k, v["auc_video"], v["auc_frame"]))
print("  own pseudo-fakes (sanity) : %.4f"
      % d["Q2_detection"]["own_pseudo_fake_auc"])
print("  mixture per-method        :", r3(d["Q2_detection"]["per_method"]))
print()
print("  each family alone, as its own ratio vs p_real:")
for f, v in d["Q2_detection"]["single_family"].items():
    print("   %-12s %.4f  %s" % (f, v["auc_video"], r3(v["per_method"])))

print()
print("=== Q1  DOMAIN GAP (coverage under the pseudo-fake density) ===")
q1 = d["Q1_domain_gap"]
print("  log p_mix 5th pct of pseudo-fakes: %.1f" %
      q1["log_p_mixture_q05_of_pseudo_fakes"])
print("  %-22s %10s %14s" % ("class", "coverage", "mean log-ratio"))
for k, v in q1["per_method"].items():
    print("  %-22s %10.3f %14.1f" % (k, v["coverage_at_q05"], v["mean_log_ratio"]))

print()
print("=== Q3  EXACT FAMILY POSTERIOR  P(mechanism | z) ===")
print("  %-16s %s" % ("manipulation", "  ".join("%10s" % f for f in fams)))
for k, v in d["Q3_family_posterior"]["mean_posterior"].items():
    print("  %-16s %s" % (k, "  ".join("%10.3f" % v[f] for f in fams)))

pr = d["Q3_family_posterior"].get("per_region")
if pr:
    print()
    print("  dominant family per region (%d-grid), by manipulation:" % pr["grid"])
    g = pr["grid"]
    for meth, v in pr["by_method"].items():
        dom = v["dominant_family_per_region"]
        uniq = sorted(set(dom), key=dom.index)
        print("   %-16s %s" % (meth, {u: dom.count(u) for u in uniq}))
elif "per_region_error" in d["Q3_family_posterior"]:
    print("  per-region FAILED:", d["Q3_family_posterior"]["per_region_error"])

print()
print("=== C5  CALIBRATION ===")
c = d["C5_calibration"]
print("  raw            ECE %.4f  MCE %.4f" % (c["raw"]["ece"], c["raw"]["mce"]))
print("  temperature    T = %.3f" % c["temperature"])
print("  after scaling  ECE %.4f  MCE %.4f" % (c["scaled"]["ece"], c["scaled"]["mce"]))
rc = c["risk_coverage"]
print("  risk-coverage: ", " ".join("%.0f%%:%.3f" % (x["coverage"] * 100, x["risk"])
                                    for x in rc[::4]))
