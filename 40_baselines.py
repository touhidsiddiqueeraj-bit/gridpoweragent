#!/usr/bin/env python3
"""
Stage 40 — Conventional baselines on the observation conditions (review-2 #12).

Trains simple supervised classifiers (logistic regression, gradient boosting)
on the SAME observation features the LLM sees, to contextualize LLM accuracy:

  state-only features : pre load/solar/wind/SOC + post V range, peak loading,
                        violation count, under/over/overload flags
  telemetry features  : switching counts (branch/gen out, demand/renewable
                        elements changed) + system-wide deltas (load, slack,
                        losses, dV_min, dV_max, dPeakLoading)

Train/test split: train on the full revised corpus (3,000 scenarios, class-
stratified 80/20 for internal validation), evaluate on the 140-scenario pilot
draw with the model fitted on the remaining 2,860 scenarios. Baselines
(majority class, uniform random) come from baselines.json.

Output: data/results/classifier_baselines.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "data" / "processed"
RESULTS = HERE / "data" / "results"
CFGS = ["E1_LLM", "E2_LLM_RAG", "E3_LLM_Tools", "E4_Full"]

def switching_features(row):
    tgt = [t for t in str(row.injected_targets).split(";") if t and t.lower() != "nan"]
    return {
        "n_branch_out": sum(1 for t in tgt if t.startswith("line") or t.startswith("trafo")),
        "n_gen_out": sum(1 for t in tgt if t.startswith("gen")),
        "n_load_changed": sum(1 for t in tgt if t.startswith("load")),
        "n_renew_changed": sum(1 for t in tgt if t.startswith("sgen") or t.startswith("wind") or t.startswith("solar")),
    }

def state_features(row):
    return {"pre_load": float(row.pre_load_scale), "pre_solar": float(row.pre_solar_fraction),
            "pre_wind": float(row.pre_wind_fraction), "pre_soc": float(row.pre_bess_soc),
            "post_v_min": float(row.post_v_min_pu), "post_v_max": float(row.post_v_max_pu),
            "post_peak": float(row.post_peak_loading_percent), "n_viol": float(row.n_violations),
            "under": float(bool(row.has_undervoltage)), "over": float(bool(row.has_overvoltage)),
            "overload": float(bool(row.has_overload))}

def delta_features(row):
    def f(c):
        try:
            v = float(getattr(row, c))
            return 0.0 if np.isnan(v) else v
        except Exception:
            return 0.0
    return {"d_load": f("delta_load_mw"), "d_slack": f("delta_slack_p_mw"),
            "d_losses": f("delta_losses_mw"), "d_vmin": f("delta_v_min_pu"),
            "d_vmax": f("delta_v_max_pu"), "d_peak": f("delta_peak_loading_percent")}

def build(frame, kind):
    rows = []
    for _, r in frame.iterrows():
        feat = {"scenario_id": r.scenario_id, "event_class": r.event_class}
        if kind == "state":
            feat.update(state_features(r))
        elif kind == "telemetry":
            feat.update(state_features(r))
            feat.update(switching_features(r))
            feat.update(delta_features(r))
        rows.append(feat)
    return pd.DataFrame(rows)

def main():
    scen = pd.read_csv(PROCESSED / "ieee14_scenarios_taxonomy2.csv")
    pilot = set(pd.read_csv(RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2.csv").scenario_id.unique())
    pilot_df = scen[scen.scenario_id.isin(pilot)].reset_index(drop=True)
    rest_df = scen[~scen.scenario_id.isin(pilot)].reset_index(drop=True)

    out = {"note": "classifiers fitted on the non-pilot corpus scenarios, evaluated on the 140-scenario pilot draw; features mirror the LLM observation blocks",
           "n_train": len(rest_df), "n_test": len(pilot_df)}
    for kind in ["state", "telemetry"]:
        tr, te = build(rest_df, kind), build(pilot_df, kind)
        Xtr, ytr = tr.drop(columns=["scenario_id", "event_class"]), tr.event_class
        Xte, yte = te.drop(columns=["scenario_id", "event_class"]), te.event_class
        for name, clf in [("logistic_regression", LogisticRegression(max_iter=2000, C=1.0)),
                          ("gradient_boosting", GradientBoostingClassifier(random_state=0))]:
            clf.fit(Xtr, ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            out[f"{kind}_{name}_acc"] = round(100 * acc, 1)
    base = json.load(open(RESULTS / "baselines.json"))
    out["majority_acc"] = round(100 * base["majority_class"]["acc"], 1)
    out["random_acc"] = round(100 * base["random_uniform"]["mean"], 1)
    (RESULTS / "classifier_baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("[PASS] Stage 40 complete")

if __name__ == "__main__":
    main()
