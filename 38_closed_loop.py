#!/usr/bin/env python3
"""
Stage 38 — Closed-loop tool-use pilot (review-2 revision, RQ3/RQ1).

Unlike stages 19-30 (one-shot prompts, tools stated but never executed in
loop), this harness runs a genuine agent loop:

    agent proposes a structured tool call WITH arguments
      -> harness validates the arguments against the network
      -> harness executes the real pandapower operation
      -> tool result is returned to the agent
      -> agent interprets, iterates (<= MAX_TURNS), then answers

Metrics per scenario: v2-key diagnosis, argument-validity rate (referenced
elements exist), execution-success rate (calls that ran), turns used, and a
tool-usage histogram. The final answer must invoke the same JSON contract as
the one-shot pilot, so results are directly comparable with the telemetry
condition of Sec. VII-C.

Usage:
  python3 38_closed_loop.py --n-test 20 [--out FILE] [--max-turns 4]
  env: GEMINI_API_KEYS (comma-separated) via secrets.env
"""
import argparse
import copy
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data" / "results"
PROCESSED = HERE / "data" / "processed"

# reuse the API pool + telemetry observation machinery so rules cannot drift
spec = importlib.util.spec_from_file_location("runner", HERE / "19_22_run_agents_gemini.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

# network reconstruction: same recipe as stage 35 (vendored, not imported —
# importing 35 would re-run its whole pair-execution loop)
import os
import pandapower as pp

def _load(name, fname):
    spec2 = importlib.util.spec_from_file_location(name, HERE / fname)
    m2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(m2)
    return m2

os.chdir(HERE)
pf_tool = _load("pf_tool", "10_power_flow_tool.py")
heavy05 = _load("heavy05", "05_heavy.py")
_stage3 = pf_tool.load_stage3()
_RECS = {}
with open(PROCESSED / "ieee14_scenarios.jsonl") as _f:
    for _line in _f:
        _r = json.loads(_line)
        _RECS[_r["scenario_id"]] = _r

def reconstruct(scenario_id):
    rec = _RECS[scenario_id]
    net = pp.from_json(str(PROCESSED / "ieee14_net_re.json"))
    points = pd.read_csv(PROCESSED / "ieee14_operating_points.csv")
    factors = pd.read_csv(PROCESSED / "ieee14_op_load_factors.csv")
    pidx = {op: i for i, op in enumerate(points.op_id)}
    row = points.iloc[pidx[rec["pre_event"]["op_id"]]]
    net.load["p_mw"] = net.load.p_mw.values * factors.drop(columns=["op_id"]).values[pidx[rec["pre_event"]["op_id"]]]
    net.load["q_mvar"] = net.load.q_mvar.values * factors.drop(columns=["op_id"]).values[pidx[rec["pre_event"]["op_id"]]]
    handles = {"pv_id": str(net.sgen.cid.iloc[0]), "wind_id": str(net.sgen.cid.iloc[1]),
               "bess_id": str(net.storage.cid.iloc[0])} if len(net.sgen) and len(net.storage) else None
    if handles:
        _stage3.set_pv_output(net, handles["pv_id"], float(row.solar_fraction))
        _stage3.set_wind_output(net, handles["wind_id"], float(row.wind_fraction))
        _stage3.set_bess_power(net, handles["bess_id"], float(row.bess_p_mw))
        _stage3.set_bess_soc(net, handles["bess_id"], float(row.bess_soc))
    heavy05.apply_injected_event(net, rec["injected_event"])
    return net

MAX_TURNS = 4
TOOL_MANIFEST = """Tools (call at most one per turn):
1. {"action":"tool_call","tool":"run_power_flow","args":{}}
   Full AC power flow on the observed network. Returns: v_min, v_max, max_loading, violation lists.
2. {"action":"tool_call","tool":"contingency_test","args":{"element":"<branch or generator name>"}}
   Takes ONE branch or generator out of service and re-solves. Returns: v_min, max_loading, overload count for that contingency.
   Valid element names: the branch/generator names appearing in the observation.
3. {"action":"tool_call","tool":"n1_sweep","args":{}}
   Full branch-outage sweep. Returns: contingencies converged, worst loading, worst branch.
4. {"action":"tool_call","tool":"run_opf","args":{}}
   Solves the optimal power flow. Returns: solvable, generation cost.
5. {"action":"tool_call","tool":"grid_query","args":{"table":"violations"|"voltages"|"loadings"}}
   Returns the requested table for the observed network.
Final answer (exactly one, as the last turn):
   {"action":"final_answer","event_class":"E0-E5|E7|E9","confidence":0.0-1.0,"tool":"<tool your diagnosis relied on most>","reason":"one sentence"}
Respond with exactly one JSON object per turn."""

SYSTEM = """You are a grid-aware LLM operator performing event diagnosis with tool access.
Diagnose the INJECTED EVENT CLASS (cause axis) of a power-system scenario.
Taxonomy: E0 Normal (no disturbance), E1 Load Surge (+% demand), E2 Load Drop (-% demand), E3 Transmission-Line Outage, E4 Generator Outage, E5 Renewable Ramp, E7 Overvoltage (V>1.05), E9 Compound (2 mechanisms).
Rules:
- E0 Normal: no switching, no demand/renewable change, no violations.
- E1 Load Surge: demand increases, no switching.
- E2 Load Drop: demand decreases, no switching.
- E3: a branch is OPEN / an outage is present, demand unchanged.
- E4: a generator is OFFLINE, demand unchanged.
- E5 Renewable Ramp: a renewable element changed output, no switching, demand unchanged.
- E7 Overvoltage: only when post-event shows overvoltage (V>1.05).
- E9 Compound: a switching event AND a demand change appear together.
You may run tools to probe the network before answering. Use the returned measurements."""

def observation_block(row):
    """Telemetry-only observation (same as the --evidence condition)."""
    tgt = [t for t in str(row.injected_targets).split(";") if t and t.lower() != "nan"]
    branch = [t for t in tgt if t.startswith("line") or t.startswith("trafo")]
    gens = [t for t in tgt if t.startswith("gen")]
    loads = [t for t in tgt if t.startswith("load")]
    renew = [t for t in tgt if t.startswith("sgen") or t.startswith("wind") or t.startswith("solar")]

    def d(col, spec="+.2f", unit=" MW"):
        v = getattr(row, col, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        if pd.isna(v):
            v = 0.0
        return f"{v:{spec}}{unit}"

    lines = [
        f"Switching: " + (f"branches OPEN: {', '.join(branch)}" if branch else "no branches opened")
        + " | " + (f"generators OFFLINE: {', '.join(gens)}" if gens else "no generators offline"),
        f"Changed demand elements: " + (", ".join(loads) if loads else "none"),
        f"Changed renewable elements: " + (", ".join(renew) if renew else "none"),
        f"Delta load: {d('delta_load_mw')} | Delta slack generation: {d('delta_slack_p_mw')} | Delta losses: {d('delta_losses_mw')}",
        f"Delta V_min: {d('delta_v_min_pu','+.4f',' pu')} | Delta V_max: {d('delta_v_max_pu','+.4f',' pu')} | Delta peak loading: {d('delta_peak_loading_percent','+.3f',' pp')}",
    ]
    return "Observed evidence (from telemetry):\n" + "\n".join(lines) + "\n"

def net_summary(net):
    voltages = {c: float(v) for c, v in zip(net.bus.cid.astype(str), net.res_bus.vm_pu.values)}
    loadings = {c: float(v) for c, v in zip(list(net.line.cid.astype(str)) + list(net.trafo.cid.astype(str)),
               list(net.res_line.loading_percent.values) + list(net.res_trafo.loading_percent.values))}
    uv = [c for c, v in voltages.items() if v < 0.94]
    ov = [c for c, v in voltages.items() if v > 1.06]
    ol = [c for c, v in loadings.items() if v > 100]
    return {"v_min": min(voltages.values()), "v_max": max(voltages.values()),
            "max_loading": max(loadings.values()), "n_uv": len(uv), "n_ov": len(ov), "n_ol": len(ol)}

# ---------------- tool execution on the real network ----------------
class ToolBox:
    def __init__(self, scenario_id):
        self.net = reconstruct(scenario_id)
        self.branches = sorted(set(list(self.net.line.cid.astype(str)) + list(self.net.trafo.cid.astype(str))))
        self.gens = sorted(set(list(self.net.gen.cid.astype(str)) + list(self.net.sgen.cid.astype(str))
                               + list(self.net.storage.cid.astype(str))))
        self.buses = sorted(set(str(b) for b in self.net.bus.cid))
        self.history = []

    def _summary(self, net):
        s = net_summary(net)
        ol = [c for c, v in zip(list(net.line.cid.astype(str)) + list(net.trafo.cid.astype(str)),
                                list(net.res_line.loading_percent.values) + list(net.res_trafo.loading_percent.values)) if v > 100]
        s["overloaded_branches"] = ol
        return s

    def call(self, tool, args):
        """Returns (executed_ok, valid_args, compact_result_text)."""
        args = args if isinstance(args, dict) else {}
        if tool == "run_power_flow":
            import pandapower as pp
            pp.runpp(self.net, numba=True)
            s = self._summary(self.net)
            ok = bool(self.net.converged)
            return ok, True, json.dumps({"converged": ok, "v_min_pu": round(s["v_min"], 4),
                                         "v_max_pu": round(s["v_max"], 4), "max_loading_pct": round(s["max_loading"], 1),
                                         "undervoltage_buses": s["n_uv"], "overvoltage_buses": s["n_ov"],
                                         "overloads": s["overloaded_branches"]})
        if tool == "contingency_test":
            el = str(args.get("element", ""))
            valid = el in self.branches or el in self.gens
            if not valid:
                return False, False, json.dumps({"error": f"unknown element '{el}'",
                                                 "valid_branches": self.branches, "valid_generators": self.gens})
            net2 = copy.deepcopy(self.net)
            if el in self.branches:
                mask = (net2.line.cid.astype(str) == el) | (net2.trafo.cid.astype(str) == el)
                net2.line.loc[net2.line.cid.astype(str) == el, "in_service"] = False
                net2.trafo.loc[net2.trafo.cid.astype(str) == el, "in_service"] = False
            else:
                net2.gen.loc[net2.gen.cid.astype(str) == el, "in_service"] = False
                net2.sgen.loc[net2.sgen.cid.astype(str) == el, "in_service"] = False
                net2.storage.loc[net2.storage.cid.astype(str) == el, "in_service"] = False
            import pandapower as pp
            try:
                pp.runpp(net2, numba=True)
            except Exception:
                return True, True, json.dumps({"converged": False, "note": "power flow did not converge with this element out"})
            s = self._summary(net2)
            return bool(net2.converged), True, json.dumps({"converged": bool(net2.converged),
                                                           "element_out": el, "v_min_pu": round(s["v_min"], 4),
                                                           "max_loading_pct": round(s["max_loading"], 1),
                                                           "overloads": s["overloaded_branches"]})
        if tool == "n1_sweep":
            import pandapower as pp
            worst, worst_el, nconv = 0.0, None, 0
            for el in self.branches:
                net2 = copy.deepcopy(self.net)
                net2.line.loc[net2.line.cid.astype(str) == el, "in_service"] = False
                net2.trafo.loc[net2.trafo.cid.astype(str) == el, "in_service"] = False
                try:
                    pp.runpp(net2, numba=True)
                except Exception:
                    continue
                if not net2.converged:
                    continue
                nconv += 1
                load = float(max(net2.res_line.loading_percent.values) if len(net2.res_line) else 0)
                if load > worst:
                    worst, worst_el = load, el
            return True, True, json.dumps({"contingencies_converged": nconv, "total_branches": len(self.branches),
                                           "worst_loading_pct": round(worst, 1), "worst_branch": worst_el})
        if tool == "run_opf":
            import pandapower as pp
            net2 = copy.deepcopy(self.net)
            try:
                pp.runopp(net2, numba=True)
            except Exception as e:
                return False, True, json.dumps({"solvable": False, "note": str(e)[:80]})
            ok = bool(net2.converged)
            cost = float(net2.res_cost) if hasattr(net2, "res_cost") else None
            out = {"solvable": ok}
            if cost is not None:
                out["objective_cost"] = round(cost, 2)
            return ok, True, json.dumps(out)
        if tool == "grid_query":
            table = str(args.get("table", "violations"))
            s = self._summary(self.net)
            if table == "voltages":
                v = {c: round(float(x), 4) for c, x in zip(self.net.bus.cid.astype(str), self.net.res_bus.vm_pu.values)}
                return True, True, json.dumps({"bus_voltages_pu": v})
            if table == "loadings":
                v = {c: round(float(x), 1) for c, x in zip(list(self.net.line.cid.astype(str)) + list(self.net.trafo.cid.astype(str)),
                                                           list(self.net.res_line.loading_percent.values) + list(self.net.res_trafo.loading_percent.values))}
                return True, True, json.dumps({"branch_loadings_pct": v})
            s2 = self._summary(self.net)
            return True, True, json.dumps({"v_min_pu": round(s2["v_min"], 4), "v_max_pu": round(s2["v_max"], 4),
                                           "max_loading_pct": round(s2["max_loading"], 1), "n_under": s2["n_uv"],
                                           "n_over": s2["n_ov"], "n_overload": s2["n_ol"],
                                           "overloads": s2["overloaded_branches"]})
        return False, False, json.dumps({"error": f"unknown tool '{tool}'"})

# ---------------- agent turn ----------------
def agent_turn(pool, messages, model):
    return runner.call_gemini_pooled(_flatten(messages), pool, model=model)

def _flatten(messages):
    """Gemini contents: user/model alternation with text parts."""
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return {"contents": contents, "generationConfig": {"temperature": 0, "maxOutputTokens": 512}}

def parse_action(text):
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            if j.get("action") in ("tool_call", "final_answer"):
                return j
            # tolerate a bare final answer without the action field
            if "event_class" in j:
                j.setdefault("action", "final_answer")
                return j
            if "tool" in j:
                j.setdefault("action", "tool_call")
                return j
        except Exception:
            pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--rpm", type=int, default=45, help="aggregate RPM across the key pool")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids-from", default="agent_runs_gemini-3.5-flash-lite.csv")
    ap.add_argument("--api-keys", default=None)
    args = ap.parse_args()

    keys = [k.strip() for k in (args.api_keys or __import__("os").getenv("GEMINI_API_KEYS", "")).split(",") if k.strip()]
    if not keys:
        raise SystemExit("[GATED] no Gemini credentials — set GEMINI_API_KEYS")
    pool = runner.GeminiKeyPool(keys, args.rpm)
    runner.preflight_keys(pool, "gemini-3.5-flash-lite")

    out_csv = Path(args.out) if args.out else RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2_closedloop.csv"
    ckpt = Path(str(out_csv).replace(".csv", "_checkpoint.json"))

    scen = pd.read_csv(PROCESSED / "ieee14_scenarios_taxonomy2.csv")
    if args.ids_from:
        ids = set(pd.read_csv(RESULTS / args.ids_from).scenario_id.unique())
        scen = scen[scen.scenario_id.isin(ids)].reset_index(drop=True)
    rng = np.random.default_rng(runner.MASTER_SEED)
    idx = rng.choice(len(scen), size=min(args.n_test, len(scen)), replace=False)
    test = scen.iloc[idx]

    done = set()
    if out_csv.exists():
        try:
            done = set(zip(pd.read_csv(out_csv).scenario_id, pd.read_csv(out_csv).config))
            print(f"[INFO] resume: {len(done)} rows done")
        except Exception:
            pass

    rows = []
    for _, srow in test.iterrows():
        if (srow.scenario_id, "closed_loop") in done:
            continue
        box = ToolBox(srow.scenario_id)
        messages = [{"role": "user", "content": SYSTEM + "\n" + observation_block(srow)
                     + "\n" + TOOL_MANIFEST + f"\nScenario {srow.scenario_id}\nDiagnose the event. You may run tools first."}]
        n_calls_valid = n_calls_exec = n_tool_calls = 0
        tool_hist = {}
        final = None
        for turn in range(args.max_turns):
            try:
                text, _lat = agent_turn(pool, messages, "gemini-3.5-flash-lite")
            except RuntimeError as e:
                print(f"[STOP] credential pool exhausted: {e} — rows stay checkpointed")
                _save(rows, out_csv, ckpt)
                return
            act = parse_action(text)
            messages.append({"role": "assistant", "content": text})
            if act is None:
                messages.append({"role": "user", "content": "Malformed response. Reply with one JSON object: a tool_call or the final_answer."})
                continue
            if act.get("action") == "final_answer":
                final = act
                break
            tool = str(act.get("tool", ""))
            n_tool_calls += 1
            tool_hist[tool] = tool_hist.get(tool, 0) + 1
            ok, valid, result = box.call(tool, act.get("args", {}))
            n_calls_valid += int(valid)
            n_calls_exec += int(ok)
            messages.append({"role": "user", "content": f"TOOL RESULT ({tool}, executed={ok}, args_valid={valid}): {result}\nContinue: run another tool_call or give the final_answer."})
        if final is None:
            messages.append({"role": "user", "content": "Turn limit reached. Respond NOW with only the final_answer JSON object."})
            try:
                text, _lat = agent_turn(pool, messages, "gemini-3.5-flash-lite")
                final = parse_action(text) or {"event_class": "?", "confidence": 0.0, "tool": "none", "reason": "no final answer"}
            except RuntimeError as e:
                print(f"[STOP] credential pool exhausted: {e}")
                _save(rows, out_csv, ckpt)
                return
        pred = str(final.get("event_class", "?")).strip().upper()
        m = re.match(r"E[0-9]", pred)
        pred = m.group(0) if m else "?"
        correct = (pred == srow.event_class)
        rows.append({"scenario_id": srow.scenario_id, "event_class": srow.event_class,
                     "config": "closed_loop", "model": "gemini-3.5-flash-lite",
                     "correct_diag": bool(correct), "turns": turn + 1, "n_tool_calls": n_tool_calls,
                     "n_calls_valid": n_calls_valid, "n_calls_exec": n_calls_exec,
                     "pred": pred, "tools_used": json.dumps(tool_hist),
                     "final_reason": str(final.get("reason", ""))[:200],
                     "raw": json.dumps(final)[:400]})
        rows_df = pd.DataFrame(rows)
        if done:
            try:
                prev = pd.read_csv(out_csv)
                rows_df = pd.concat([prev, rows_df], ignore_index=True).drop_duplicates(
                    subset=["scenario_id", "config"], keep="last")
            except Exception:
                pass
        rows_df.to_csv(out_csv, index=False)
        ckpt.write_text(json.dumps({"done": len(done) + len(rows)}))
        print(f"[{len(done)+len(rows)}] {srow.scenario_id} true={srow.event_class} pred={pred} "
              f"correct={correct} turns={turn+1} tools={n_tool_calls} valid={n_calls_valid} exec={n_calls_exec}", flush=True)

    df = pd.read_csv(out_csv)
    print("\n=== CLOSED-LOOP SUMMARY ===")
    print(f"scenarios: {len(df)} | diag: {df.correct_diag.mean()*100:.1f}%")
    print(f"tool calls: {df.n_tool_calls.sum()} | args valid: {df.n_calls_valid.sum()} | executed: {df.n_calls_exec.sum()}")
    if df.n_tool_calls.sum():
        print(f"argument-validity rate: {df.n_calls_valid.sum()/df.n_tool_calls.sum()*100:.1f}% | "
              f"execution rate: {df.n_calls_exec.sum()/df.n_tool_calls.sum()*100:.1f}%")
    print("turns:", df.turns.value_counts().sort_index().to_dict())
    print("[PASS] Stage 38 complete")

def _save(rows, out_csv, ckpt):
    if not rows:
        return
    rows_df = pd.DataFrame(rows)
    try:
        prev = pd.read_csv(out_csv)
        rows_df = pd.concat([prev, rows_df], ignore_index=True).drop_duplicates(
            subset=["scenario_id", "config"], keep="last")
    except Exception:
        pass
    rows_df.to_csv(out_csv, index=False)
    ckpt.write_text(json.dumps({"saved": len(rows_df)}))

if __name__ == "__main__":
    main()
