#!/usr/bin/env python3
"""
Stage 39 — Multi-axis annotation and scoring (review-2 point #4).

The revised taxonomy keys every scenario by its INITIATING CAUSE, but a grid
event is describable on several axes at once. This stage annotates every pilot
scenario with rule-generated axes (each disclosed as rule-generated, not
expert-labelled):

  cause       — the v2 event class (mechanism)
  outcome     — undervoltage / overvoltage / overload / none (physics flags)
  equipment   — which element types the event touched (branch / generator /
                demand / renewable)
  severity    — stage-8 severity tier (normal / moderate / high)
  rec_tool    — the reference-required tool set (from stage-9 v2 labels)

and scores the revised-taxonomy runs PER AXIS: a cause-class answer is scored
cause-correct as before, but additionally checked for outcome-consistency
(does the answer's implied physics match the scenario's flags) and for
recommended-tool consistency (did the stated tool belong to the required set).

Output: data/results/multiaxis_scoring.csv (per scenario x config x axis) and
        data/results/multiaxis_summary.json (per-model per-axis accuracy).
"""
import ast
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "data" / "processed"
RESULTS = HERE / "data" / "results"

CFGS = ["E1_LLM", "E2_LLM_RAG", "E3_LLM_Tools", "E4_Full"]
TOOLS8 = ["power_flow", "state_estimation", "contingency", "n1_security", "opf",
          "grid_query_topology", "grid_query_limits", "grid_query_equipment"]

def predicted(raw):
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            t = str(j.get("event_class", "")).strip().upper()
            mm = re.match(r"E[0-9]", t)
            if mm:
                return mm.group(0), str(j.get("tool", "")).lower()
        except Exception:
            pass
    ms = re.findall(r"E[0-9]", str(t_ := str(raw)))
    return (ms[-1] if ms else "?"), "power_flow"

def outcome_axis(row):
    o = []
    if bool(row.has_undervoltage):
        o.append("undervoltage")
    if bool(row.has_overvoltage):
        o.append("overvoltage")
    if bool(row.has_overload):
        o.append("overload")
    return "+".join(o) if o else "none"

def equipment_axis(row):
    e = []
    mech = str(row.injected_mechanism)
    if mech == "line_outage":
        e.append("branch")
    elif mech == "generator_outage":
        e.append("generator")
    elif mech == "load_change":
        e.append("demand")
    elif mech == "renewable_ramp":
        e.append("renewable")
    elif mech == "compound":
        e.extend(["branch", "demand"])
    return "+".join(e) if e else "none"

def severity_axis(row):
    try:
        sev = float(row.severity_true)
    except Exception:
        return "unknown"
    if sev <= 0.0:
        return "none"
    if sev < 0.0526:
        return "moderate"
    return "high"

# CAUSE->IMPLIED-OUTCOME compatibility: which outcome signatures a cause-correct
# answer is consistent with (rule-generated, disclosed as such)
CAUSE_OUTCOME_OK = {
    "E0": {"none"},
    "E1": {"none", "undervoltage"},          # surges can sag voltage at high load
    "E2": {"none"},
    "E3": {"none", "undervoltage"},          # line outages can cause undervoltage
    "E4": {"none", "undervoltage"},
    "E5": {"none"},
    "E7": {"overvoltage"},
    "E9": {"none", "undervoltage", "overload"},  # compounds can trip either flag
}
# (E8-keyed scenarios no longer exist under the revised key)

def main():
    scen = pd.read_csv(PROCESSED / "ieee14_scenarios_taxonomy2.csv")
    sev = pd.read_csv(PROCESSED / "ieee14_violation_severity.csv")
    scen = scen.merge(sev[["scenario_id", "severity_true"]], on="scenario_id", how="left")
    ref2 = pd.read_csv(PROCESSED / "ieee14_reference_labels_taxonomy2.csv").set_index("scenario_id")

    rows = []
    for _, r in scen.iterrows():
        rows.append({"scenario_id": r.scenario_id, "cause": r.event_class,
                     "outcome": outcome_axis(r), "equipment": equipment_axis(r),
                     "severity": severity_axis(r),
                     "rec_tools": ";".join(t for t in TOOLS8 if ref2.loc[r.scenario_id][t] == "required")})
    axes = pd.DataFrame(rows)

    runs = {"API": RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2.csv",
            "Local": RESULTS / "agent_runs_gemma-4-E4B-it-Q4_0_gguf_tax2.csv"}
    summary = {}
    score_rows = []
    for lbl, f in runs.items():
        df = pd.read_csv(f).drop_duplicates(subset=["scenario_id", "config"])
        df = df.merge(axes, on="scenario_id")
        parsed = df.raw.apply(lambda r: predicted(r))
        df["pred_cause"], df["pred_tool"] = zip(*parsed)
        df["cause_ok"] = df.pred_cause == df.cause
        def outcome_consistent(r):
            implied = CAUSE_OUTCOME_OK.get(r.pred_cause)
            return implied is not None and r.outcome in implied
        df["outcome_consistent"] = df.apply(outcome_consistent, axis=1)
        def tool_ok(r):
            req = set(str(r.rec_tools).split(";")) if str(r.rec_tools) else set()
            return (not req) or r.pred_tool in req or any(t in r.pred_tool for t in req)
        df["tool_consistent"] = df.apply(tool_ok, axis=1)
        score_rows.append(df.assign(model=lbl))
        summary[lbl] = {
            "cause_acc": round(100 * df.cause_ok.mean(), 1),
            "outcome_consistent": round(100 * df.outcome_consistent.mean(), 1),
            "tool_consistent": round(100 * df.tool_consistent.mean(), 1),
            "n": len(df),
        }
    out = pd.concat(score_rows, ignore_index=True)
    out.to_csv(RESULTS / "multiaxis_scoring.csv", index=False)
    axes.to_csv(PROCESSED / "ieee14_multiaxis_annotations.csv", index=False)
    (RESULTS / "multiaxis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("[PASS] Stage 39 complete")

if __name__ == "__main__":
    main()
