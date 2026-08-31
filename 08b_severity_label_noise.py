#!/usr/bin/env python3
"""
Stage 8b — Severity-to-label noise bound (audit fix for rho disclosure)

Question: the rule-based reference labels (Stage 9) gate two tools on severity
(n1_security: required iff S>0.0526; opf: required iff has_overload and S>0.1053).
Severity_est is now computed from WLS state-estimate voltages (08, corrected), so
estimation uncertainty can flip tiers. This script recomputes the full tier table
with severity_true vs severity_est and reports:
  - per-tool tier flip counts (any change)
  - scoring-relevant flips: membership in {required, strongly_appropriate} changes,
    which is the set the agent tool-selection correctness / H-TOOL tags are judged
    against. This is the quantitative bound on labeling noise injected into the
    pilot's tool-accuracy and hallucination-rate numbers (Table IV).
"""
from pathlib import Path
import json
import pandas as pd
import importlib.util

OUTPUT_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)

# reuse Stage 9 assign_tier directly so rules cannot drift
spec = importlib.util.spec_from_file_location("stage9", Path("09_reference_labels.py"))
stage9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage9)

ACCEPTED = ("required", "strongly_appropriate")

def main():
    summary = {}
    for case in ["ieee14", "case39", "case118"]:
        sev = pd.read_csv(OUTPUT_DIR / f"{case}_violation_severity.csv")
        scen = pd.read_csv(OUTPUT_DIR / f"{case}_scenarios.csv")
        ref = pd.read_csv(OUTPUT_DIR / f"{case}_reference_labels.csv")
        df = scen.merge(sev[["scenario_id", "severity_true", "severity_est"]], on="scenario_id")
        df = df.merge(ref[["scenario_id"] + stage9.TOOLS], on="scenario_id", suffixes=("", "_ref"))

        rows = []
        for _, r in df.iterrows():
            tiers_true = stage9.assign_tier(pd.Series({**r.to_dict(), "severity_true": r.severity_true}))
            tiers_est = stage9.assign_tier(pd.Series({**r.to_dict(), "severity_true": r.severity_est}))
            for t in stage9.TOOLS:
                rows.append({
                    "tool": t,
                    "tier_true": tiers_true[t], "tier_est": tiers_est[t],
                    "scoring_flip": (tiers_true[t] in ACCEPTED) != (tiers_est[t] in ACCEPTED),
                })
        fl = pd.DataFrame(rows)
        per_tool = fl.groupby("tool").agg(
            n=("tool", "size"),
            any_flips=("tier_true", lambda s: 0),  # placeholder replaced below
        )
        any_flip = fl.assign(changed=fl.tier_true != fl.tier_est).groupby("tool").changed.agg(["sum", "mean"])
        score_flip = fl.groupby("tool").scoring_flip.agg(["sum", "mean"])
        total_judg = len(fl)
        total_score_flips = int(fl.scoring_flip.sum())
        summary[case] = {
            "scenarios": int(len(df)),
            "scenario_tool_judgments": total_judg,
            "scoring_relevant_flips": total_score_flips,
            "scoring_relevant_flip_pct": round(100.0 * total_score_flips / total_judg, 3),
            "per_tool_scoring_flips": {
                t: {"n": int(score_flip.loc[t, "sum"]), "pct": round(100.0 * score_flip.loc[t, "mean"], 3)}
                for t in stage9.TOOLS
            },
            "per_tool_any_tier_change": {
                t: {"n": int(any_flip.loc[t, "sum"]), "pct": round(100.0 * any_flip.loc[t, "mean"], 3)}
                for t in stage9.TOOLS
            },
        }
        print(f"=== {case}: {len(df)} scen, {total_judg} scenario-tool judgments ===")
        print(f"  scoring-relevant flips (accepted-set membership changes): "
              f"{total_score_flips} ({100.0*total_score_flips/total_judg:.3f}%)")
        for t in ["n1_security", "opf"]:
            print(f"  {t}: any-tier {any_flip.loc[t,'sum']:.0f} ({100*any_flip.loc[t,'mean']:.2f}%), "
                  f"scoring {score_flip.loc[t,'sum']:.0f} ({100*score_flip.loc[t,'mean']:.2f}%)")
    # rho per case for context
    for case in summary:
        note = json.load(open(OUTPUT_DIR / f"{case}_stage8_separation_note.json"))
        summary[case]["spearman_rho_est_vs_true"] = round(note["severity"]["rho"], 4)
    with open(RESULTS_DIR / "severity_label_noise_bound.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved {RESULTS_DIR/'severity_label_noise_bound.json'}")
    print("[PASS] Stage 8b complete — labeling-noise bound quantified")

if __name__ == "__main__":
    main()
