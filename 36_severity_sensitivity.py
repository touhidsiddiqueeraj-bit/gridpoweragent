#!/usr/bin/env python3
"""
Stage 36 — Severity-boundary sensitivity analysis for the strict tool metric.

The strict-specific tool metric depends on two severity thresholds from Eq. 4
(n1_security: required iff S > mid; opf: required iff overload and S > high).
The published boundaries were selected by grid search on this corpus, which is
a circularity. This script recomputes reference labels and the strict tool
metric under boundary sets that were NOT the search winner:

  S0 (published)  0.0263 / 0.0526 / 0.1053   grid-search winner
  S1 (standards)  0.05   / 0.10   / 0.15     ANSI-C84.1-style bands
  S2 (tight)      0.02   / 0.04   / 0.08
  S3 (empirical)  tertiles of the corpus severity distribution

Reports: strict tool accuracy per model per boundary set, the local-vs-API gap,
and rank stability (does the local >= API ordering hold under every set?).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/touhid/Documents/llmpaper")
RESULTS = ROOT / "data/results"
PROCESSED = ROOT / "data/processed"
CFGS = ["E1_LLM", "E2_LLM_RAG", "E3_LLM_Tools", "E4_Full"]
TOOLS8 = ["power_flow", "state_estimation", "contingency", "n1_security", "opf",
          "grid_query_topology", "grid_query_limits", "grid_query_equipment"]

def stated_tool(raw, fallback="power_flow"):
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            t = str(j.get("tool", "")).strip().lower()
            for k in TOOLS8:
                if k in t:
                    return k
            return t or fallback
        except Exception:
            pass
    return fallback

def load_runs():
    import ast
    def hall(v): return False
    runs = {}
    for lbl, f in [("api", "agent_runs_gemini-3.5-flash-lite.csv"),
                   ("local", "agent_runs_gemma-4-E4B-it-Q4_0_gguf.csv")]:
        df = pd.read_csv(RESULTS / f).drop_duplicates(subset=["scenario_id", "config"])
        df["stated"] = df.raw.apply(lambda r: stated_tool(r))
        runs[lbl] = df
    return runs

def assign_tier_parameterized(sev_val, has_ol, event_class, n_viol, mid, hi):
    """Mirrors 09_reference_labels.assign_tier exactly, with the two severity
    thresholds (mid=0.0526, hi=0.1053 in the published set) as parameters."""
    tiers = {}
    tiers["power_flow"] = "required" if (n_viol > 0 or event_class in ["E3", "E4", "E6", "E7", "E8", "E9"]) else "strongly_appropriate"
    tiers["state_estimation"] = "required" if n_viol > 0 else "conditional"
    tiers["contingency"] = ("required" if (has_ol or event_class in ["E3", "E4", "E9"])
                            else "strongly_appropriate" if has_ol == False and False else "conditional")
    if has_ol or event_class in ["E3", "E4", "E9"]:
        tiers["contingency"] = "required"
    elif event_class in ["E6"]:
        tiers["contingency"] = "strongly_appropriate"
    else:
        tiers["contingency"] = "conditional"
    tiers["n1_security"] = "required" if sev_val > mid else "conditional"
    if has_ol and sev_val > hi:
        tiers["opf"] = "required"
    elif event_class in ["E6", "E8"]:
        tiers["opf"] = "strongly_appropriate"
    elif event_class in ["E0", "E2"]:
        tiers["opf"] = "unnecessary"
    else:
        tiers["opf"] = "conditional"
    tiers["grid_query_topology"] = "required"
    tiers["grid_query_equipment"] = "conditional"
    return tiers

def main():
    sev = pd.read_csv(PROCESSED / "ieee14_violation_severity.csv")
    tert = np.quantile(sev.severity_true, [1/3, 2/3])
    sets = {
        "S0 published (grid-searched)": [0.0263, 0.0526, 0.1053],
        "S1 standards [0.05,0.10,0.15]": [0.05, 0.10, 0.15],
        "S2 tight [0.02,0.04,0.08]": [0.02, 0.04, 0.08],
        "S3 empirical tertiles": [round(float(tert[0]), 4), round(float(tert[1]), 4), round(float(tert[1]) + 0.05, 4)],
    }

    runs = load_runs()
    sevmap = sev.set_index("scenario_id")
    rows = []
    for set_name, (lo_b, mid_b, hi_b) in sets.items():
        for lbl, df in [("API", runs["api"]), ("Local", runs["local"])]:
            strict_map = {}
            # strict: stated tool is REQUIRED under this boundary set (one-level rule)
            def is_required(sev_val, has_ol, ec, n_viol, tool):
                t = assign_tier_parameterized(sev_val, has_ol, ec, n_viol, mid_b, hi_b)
                if tool == "power_flow":
                    # strict rule: PF counts only when solely required
                    req = {k for k, v in t.items() if v == "required"}
                    return t["power_flow"] == "required" and len(req) == 1
                return t[tool] == "required"
            ok, n = 0, 0
            for _, r in df.iterrows():
                srow = sevmap.loc[r.scenario_id]
                sv = float(srow.severity_true) if "severity_true" in srow else float(srow.get("severity", 0))
                has_ol = bool(srow.get("has_overload", False))
                n_viol = int(srow.get("n_violations", 0))
                ec = srow.event_class
                t = assign_tier_parameterized(sv, has_ol, ec, n_viol, mid_b, hi_b)
                if r.stated in t and t[r.stated] == "required":
                    if r.stated == "power_flow":
                        req = {k for k, v in t.items() if v == "required"}
                        if len(req) == 1:
                            ok += 1
                    else:
                        ok += 1
                n += 1
            rows.append({"set": set_name, "model": lbl, "n": n, "strict_ok": ok,
                         "acc": round(100*ok/n, 1)})
    sens = pd.DataFrame(rows)
    print(sens.to_string(index=False))

    gaps = sens.pivot(index="set", columns="model", values="acc")
    gaps["local_minus_api"] = (gaps["Local"] - gaps["API"]).round(1)
    print("\nLocal-API strict-tool accuracy per boundary set (pp):")
    print(gaps[["API", "Local", "local_minus_api"]].to_string())
    stable = bool((gaps.local_minus_api > 0).all())
    invariant = bool(gaps["API"].nunique() == 1 and gaps["Local"].nunique() == 1)
    print(f"\ninvariant across boundary sets: {invariant}; Local > API under every set: {stable}")

    sens.to_csv(RESULTS / "severity_sensitivity.csv", index=False)
    gaps.to_csv(RESULTS / "severity_sensitivity_gaps.csv")
    (RESULTS / "severity_sensitivity_gaps.csv").write_text(
        (RESULTS / "severity_sensitivity_gaps.csv").read_text())
    print("[PASS] Stage 36 complete")

if __name__ == "__main__":
    main()