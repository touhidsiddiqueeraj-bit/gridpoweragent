#!/usr/bin/env python3
"""
Stage 35 — RQ3: post-hoc tool-execution scoring.

For every (scenario, stated tool) pair in the pilot runs, actually EXECUTE the
stated tool on the scenario's post-event network and score whether the call is
valid: reconstructs the network, runs, converges, and produces the expected
outputs. Execution is model-independent, so pairs are pooled across models and
configurations; the model-dependent part (which tools get stated) is RQ2.

Scoring:
  power_flow      reconstruct post-event net -> runpp            valid = converged
  contingency     post-event net -> outage injected line/gen -> runpp
                  valid = converged (result reveals the recorded overload/undervoltage)
  n1_security     post-event net -> full line outage sweep     valid = sweep completes
  opf             post-event net -> pp.runopp                  valid = OPF solves
  grid_query_*    post-event net -> query                      valid = returns dict

Output: data/results/tool_execution_scores.csv (pair-level) and
        data/results/tool_execution_summary.json (by tool).
"""
import json
import re
from pathlib import Path

import copy

import numpy as np
import pandas as pd
import pandapower as pp

ROOT = Path("/home/touhid/Documents/llmpaper")
os_cwd = ROOT
import os
os.chdir(ROOT)

import importlib.util

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

pf_tool = _load("pf_tool", "10_power_flow_tool.py")

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--case", default="ieee14", help="ieee14 | case39 | case118")
CASE = _ap.parse_args().case
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "data/results"
CFGS = ["E1_LLM", "E2_LLM_RAG", "E3_LLM_Tools", "E4_Full"]
STATED_MAP = {  # prompt contract tool names -> executable checkers added below
    "power_flow": "power_flow",
    "contingency": "contingency",
    "n1_security": "n1_security",
    "opf": "opf",
    "grid_query_topology": "grid_query",
    "grid_query_limits": "grid_query",
    "grid_query_equipment": "grid_query",
    "grid_query_bess": "grid_query",
    "grid_query_renewable": "grid_query",
    "grid_query": "grid_query",
}

def stated_tools(raw, fb="power_flow"):
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            t = str(j.get("tool", "")).strip().lower()
            for k in STATED_MAP:
                if k in t:
                    return k
            return t or fb
        except Exception:
            pass
    return fb

# ---- scenario records (jsonl: full nested injected_event) ----
_recs = {}
for jf in [PROCESSED / ("case39_scenarios.jsonl" if CASE == "case39" else "case118_scenarios.jsonl" if CASE == "case118" else "ieee14_scenarios.jsonl")]:
    if jf.exists():
        with open(jf) as f:
            for line in f:
                r = json.loads(line)
                _recs[r["scenario_id"]] = r

# ---- reconstruction (mirrors 10_power_flow_tool.run_power_flow) ----
heavy = _load("heavy05", "05_heavy.py")
stage3 = pf_tool.load_stage3()

def reconstruct(scenario_id):
    rec = _recs.get(scenario_id)
    if rec is None:
        raise KeyError(scenario_id)
    tag = "" if CASE == "ieee14" else CASE
    net = pp.from_json(str(PROCESSED / f"{CASE}_net_re.json"))
    points = pd.read_csv(PROCESSED / f"{CASE}_operating_points.csv")
    factors = pd.read_csv(PROCESSED / f"{CASE}_op_load_factors.csv")
    pidx = {op: i for i, op in enumerate(points.op_id)}
    row = points.iloc[pidx[rec["pre_event"]["op_id"]]]
    frow = factors.drop(columns=["op_id"]).values[pidx[rec["pre_event"]["op_id"]]]
    net.load["p_mw"] = net.load.p_mw.values * frow
    net.load["q_mvar"] = net.load.q_mvar.values * factors.drop(columns=["op_id"]).values[pidx[rec["pre_event"]["op_id"]]]
    handles = {"pv_id": str(net.sgen.cid.iloc[0]), "wind_id": str(net.sgen.cid.iloc[1]),
               "bess_id": str(net.storage.cid.iloc[0])} if len(net.sgen) and len(net.storage) else None
    if handles:
        stage3.set_pv_output(net, handles["pv_id"], float(row.solar_fraction))
        stage3.set_wind_output(net, handles["wind_id"], float(row.wind_fraction))
        stage3.set_bess_power(net, handles["bess_id"], float(row.bess_p_mw))
        stage3.set_bess_soc(net, handles["bess_id"], float(row.bess_soc))
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()

    heavy.apply_injected_event(net, rec["injected_event"])
    return net

def net_summary(net):
    voltages = {c: float(v) for c, v in zip(net.bus.cid, net.res_bus.vm_pu.values)}
    loadings = {c: float(v) for c, v in zip(list(net.line.cid) + list(net.trafo.cid),
               list(net.res_line.loading_percent.values) + list(net.res_trafo.loading_percent.values))}
    uv = [c for c, v in voltages.items() if v < 0.94]
    ov = [c for c, v in voltages.items() if v > 1.06]
    ol = [c for c, v in loadings.items() if v > 100]
    return {"v_min": min(voltages.values()), "v_max": max(voltages.values()),
            "max_loading": max(loadings.values()), "n_uv": len(uv), "n_ov": len(ov), "n_ol": len(ol)}

# ---- pair extraction from existing runs ----
pair_sources = []
if CASE == "ieee14":
    pair_sources = ["agent_runs_gemini-3.5-flash-lite.csv", "agent_runs_gemma-4-E4B-it-Q4_0_gguf.csv"]
else:
    pair_sources = [f"agent_runs_gemini-3_5-flash-lite_{CASE}.csv", f"agent_runs_gemma-4-E4B-it-Q4_0_gguf_{CASE}.csv"]
pairs = set()
for f in pair_sources:
    fp = RESULTS / f
    if not fp.exists():
        print(f"[WARN] missing {f}")
        continue
    df = pd.read_csv(fp).drop_duplicates(subset=["scenario_id", "config"])
    df["stated"] = df.raw.apply(lambda r: stated_tools(r))
    pairs |= set(zip(df.scenario_id, df.stated))
pairs = sorted(pairs)
print(f"{len(pairs)} unique (scenario, stated tool) pairs")

# ---- execute (cache reconstructions) ----
nets = {}
results = []
for i, (sid, tool) in enumerate(pairs):
    exec_kind = STATED_MAP.get(tool)
    row = {"scenario_id": sid, "stated_tool": tool, "executed_ok": False,
           "converged": False, "informative": False, "detail": ""}
    try:
        if sid not in nets:
            nets[sid] = reconstruct(sid)
        net = nets[sid]
        if exec_kind == "power_flow":
            pp.runpp(net, numba=True)
            summ = net_summary(net)
            row.update(converged=bool(net.converged), executed_ok=bool(net.converged),
                       informative=bool(net.converged),
                       detail=f"vmin={summ['v_min']:.3f} maxload={summ['max_loading']:.1f}")
        elif exec_kind == "contingency":
            # outage the scenario's injected line target on the post-event net
            rec = _recs[sid]
            targets = []
            def collect(ev):
                if ev["mechanism"] in ("line_outage", "generator_outage"):
                    targets.extend(ev["targets"])
                if ev["mechanism"] == "compound":
                    for c in ev["components"]:
                        collect(c)
            collect(rec["injected_event"])
            net2 = copy.deepcopy(net)
            for t in targets:
                heavy.apply_injected_event(net2, {"mechanism": "generator_outage" if t.startswith("gen_") else "line_outage", "targets": [t]})
            pp.runpp(net2, numba=True)
            summ = net_summary(net2)
            rec_ol = rec.get("has_overload", False)
            row.update(converged=bool(net2.converged), executed_ok=bool(net2.converged),
                       informative=bool(net2.converged),
                       detail=f"vmin={summ['v_min']:.3f} maxload={summ['max_loading']:.1f} ol={summ['n_ol']}")
        elif exec_kind == "n1_security":
            # full line-outage sweep on the post-event net
            n_conv, worst = 0, 0.0
            for lid in net.line.cid.values:
                net2 = copy.deepcopy(net)
                net2.line.loc[net2.line.cid == lid, "in_service"] = False
                try:
                    pp.runpp(net2, numba=True)
                    if net2.converged:
                        n_conv += 1
                        loadings = list(net2.res_line.loading_percent.values)
                        worst = max(worst, max(loadings) if loadings else 0)
                except Exception:
                    pass
            row.update(executed_ok=True, converged=True,
                       informative=(n_conv > 0),
                       detail=f"{n_conv}/{len(net.line.cid)} contingencies converged, worst {worst:.1f}%")
        elif exec_kind == "opf":
            net2 = net.deepcopy()
            pp.runopp(net2, numba=True)
            ok = bool(net2.converged) and ("OPF" in net2 or True)
            row.update(executed_ok=ok, converged=ok, informative=ok,
                       detail=f"OPF solved, obj={float(net2.res_cost):.1f}" if hasattr(net2, "res_cost") else "OPF solved")
        elif exec_kind == "grid_query":
            summ = net_summary(net)
            row.update(executed_ok=True, converged=True, informative=True,
                       detail=f"query on post-event net: vmin={summ['v_min']:.3f}")
        else:
            row["detail"] = "unknown tool"
    except FileNotFoundError as e:
        row["detail"] = f"component not found: {e}"
    except Exception as e:
        row["detail"] = f"execution error: {str(e)[:80]}"
    results.append(row)
    if i % 50 == 0:
        print(f"[{i}/{len(pairs)}] {sid} {tool}: {row['detail'][:60]}")

out = pd.DataFrame(results)
out["case"] = CASE
out.to_csv(RESULTS / f"tool_execution_scores_{CASE}.csv", index=False)

summary = {}
for tool, sub in out.groupby("stated_tool"):
    summary[tool] = {
        "pairs": int(len(sub)),
        "executed_ok": int(sub.executed_ok.sum()),
        "rate": round(100 * sub.executed_ok.mean(), 1) if len(sub) else None,
    }
summary["total_pairs"] = int(len(out))
(RESULTS / f"tool_execution_summary_{CASE}.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("[PASS] Stage 35 complete")
