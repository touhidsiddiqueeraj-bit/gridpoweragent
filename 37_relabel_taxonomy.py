#!/usr/bin/env python3
"""
Stage 37 — Taxonomy v2: separate the cause and outcome axes (E6/E8 relabel).

The v1 key assigned the OUTCOME classes E6 (undervoltage) / E8 (thermal
overload) to scenarios whose injected mechanisms are line outages, generator
outages, and outage+surge compounds. Cause-correct answers therefore scored
zero — the mixed-axis finding in the paper. v2 rekeys every E6/E8 scenario
by its injected mechanism:

    E6 + line_outage       -> E3        E8 + compound     -> E9
    E6 + generator_outage  -> E4        E8 + line_outage  -> E3

E7 (overvoltage) is kept unchanged: its ladder mechanisms match the outcome
reading and it already scores 1.0. Post-event physics, scenario IDs, and the
has_* outcome flags are untouched; the outcome information stays in the
columns. Provenance column event_class_v1 preserves the old key.

Outputs (data/processed/* is gitignored — regenerate deterministically):
  data/processed/{case}_scenarios_taxonomy2.csv / .jsonl
  data/processed/{case}_reference_labels_taxonomy2.csv  (assign_tier re-run,
  because tool tiers key on event_class)

Run:  python3 37_relabel_taxonomy.py [--case ieee14|case39]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "data" / "processed"

# cause-class names, matching the v1 taxonomy wording
CLASS_NAMES = {
    "E0": "Normal", "E1": "Load Surge", "E2": "Load Drop",
    "E3": "Transmission-Line Outage", "E4": "Generator Outage",
    "E5": "Renewable Ramp", "E7": "Overvoltage", "E9": "Compound",
}

# explicit, exhaustive mechanism -> cause class for the relabeled rows
RELABEL = {
    ("E6", "line_outage"): "E3",
    ("E6", "generator_outage"): "E4",
    ("E8", "compound"): "E9",
    ("E8", "line_outage"): "E3",
    # case39 ladders key E8 surges under load_change (ieee14 has none)
    ("E8", "load_change"): "E1",
}


def relabel_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_class_v1"] = df["event_class"]
    mask = df["event_class"].isin(["E6", "E8"])
    unmapped = []
    new_ec = []
    for _, r in df.iterrows():
        if not mask.loc[r.name]:
            new_ec.append(r["event_class"])
            continue
        key = (r["event_class"], str(r["injected_mechanism"]))
        if key not in RELABEL:
            unmapped.append(key)
            new_ec.append(r["event_class"])
        else:
            new_ec.append(RELABEL[key])
    if unmapped:
        raise SystemExit(f"[FAIL] unmapped E6/E8 mechanisms: {sorted(set(unmapped))} — "
                         f"extend RELABEL before rerunning")
    df["event_class"] = new_ec
    df["event_name"] = df["event_class"].map(CLASS_NAMES)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="ieee14", choices=["ieee14", "case39"])
    args = ap.parse_args()
    case = args.case

    scen_path = PROCESSED / f"{case}_scenarios.csv"
    jsonl_path = PROCESSED / f"{case}_scenarios.jsonl"
    sev = pd.read_csv(PROCESSED / f"{case}_violation_severity.csv")

    scen = pd.read_csv(scen_path)
    n = len(scen)
    out = relabel_frame(scen)
    changed = int((out.event_class_v1 != out.event_class).sum())
    print(f"[INFO] relabeled {changed}/{n} rows (E6/E8 -> cause classes)")

    # -- provenance crosstab: v1 key -> v2 key --------------------------------
    ct = pd.crosstab(out.loc[out.event_class_v1.isin(["E6", "E8"]), "event_class_v1"],
                     out.loc[out.event_class_v1.isin(["E6", "E8"]), "event_class"])
    print("\nv1 -> v2 relabel crosstab:")
    print(ct.to_string())

    # -- write scenarios -------------------------------------------------------
    out_csv = PROCESSED / f"{case}_scenarios_taxonomy2.csv"
    out.to_csv(out_csv, index=False)
    if jsonl_path.exists():
        import ast
        with open(jsonl_path) as f, open(PROCESSED / f"{case}_scenarios_taxonomy2.jsonl", "w") as g:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["event_class_v1"] = rec.get("event_class")
                if rec["event_class_v1"] in ("E6", "E8"):
                    # jsonl nests the mechanism inside injected_event (flat field is null)
                    inj = rec.get("injected_event")
                    if isinstance(inj, str):
                        inj = ast.literal_eval(inj)
                    mech = (inj or {}).get("mechanism") or rec.get("injected_mechanism")
                    rec["event_class"] = RELABEL[(rec["event_class_v1"], str(mech))]
                    rec["event_name"] = CLASS_NAMES[rec["event_class"]]
                g.write(json.dumps(rec) + "\n")
        print(f"[INFO] wrote {out_csv} and .jsonl")
    else:
        print(f"[INFO] wrote {out_csv} (no .jsonl source found)")

    # -- reference labels: assign_tier keys on event_class, so re-derive ------
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage09", HERE / "09_reference_labels.py")
    s09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s09)

    merged = out.merge(sev[["scenario_id", "severity_true", "cat_true"]], on="scenario_id")
    assert len(merged) == len(out), "severity merge lost rows"
    tools = s09.TOOLS
    records = []
    for _, row in merged.iterrows():
        tiers = s09.assign_tier(row)
        records.append({"scenario_id": row.scenario_id, "event_class": row.event_class,
                        **tiers, "severity": float(row.severity_true),
                        "provenance": "deterministic Stage5+Stage8, LLM-independent; taxonomy v2 (37)",
                        "leakage_audit": "severity_tier not in LLM input (input = structured_grid_state: voltages/loadings/outages only)"})
    labels = pd.DataFrame(records)
    labels_csv = PROCESSED / f"{case}_reference_labels_taxonomy2.csv"
    labels.to_csv(labels_csv, index=False)
    print(f"[INFO] wrote {labels_csv} ({len(labels)} rows, {len(tools)} tools)")

    # -- pilot-draw distribution under the new key ----------------------------
    if case == "ieee14":
        runs = pd.read_csv(HERE / "data/results/agent_runs_gemini-3.5-flash-lite.csv")
    else:
        runs = pd.read_csv(HERE / "data/results/agent_runs_gemini-3_5-flash-lite_case39.csv")
    pilot_ids = set(runs.scenario_id.unique())
    pilot = out[out.scenario_id.isin(pilot_ids)]
    print(f"\nPilot draw ({len(pilot_ids)} scenarios) — class counts v1 -> v2:")
    comp = pd.DataFrame({"v1": pilot.event_class_v1.value_counts(),
                         "v2": pilot.event_class.value_counts()}).fillna(0).astype(int)
    print(comp.to_string())
    print(f"\n[PASS] Stage 37 complete ({case})")


if __name__ == "__main__":
    main()
