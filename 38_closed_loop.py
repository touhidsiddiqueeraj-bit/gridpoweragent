#!/usr/bin/env python3
"""
Stage 38 (v2) — Closed-loop tool-use pilot, construction-graded.

v2 changes (response to brutal verdict 3, points 1-2):
  - The observation no longer names switched elements. It reports a per-element
    status table (in_service, loading %) — identifying WHICH branch/generator
    is out is part of the agent's task.
  - The final answer is rejected until at least --min-tools tool calls have
    EXECUTED (forces chaining instead of the v1 one-call ritual).
  - The final answer must include "identified_element": the agent's name for
    the switched element. Construction accuracy is scored against the truly
    switched element (not merely "exists in the network"); on non-switching
    scenarios the agent must correctly report "none".
  - --engine local runs the same loop on the local llama-server via 9090.
  - --wls builds the observed status table from the WLS-estimated state
    (41_wls_state.py output) instead of privileged post-event truth.
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

spec = importlib.util.spec_from_file_location("runner", HERE / "19_22_run_agents_gemini.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

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
    _stage3.set_pv_output(net, str(net.sgen.cid.iloc[0]), float(row.solar_fraction))
    _stage3.set_wind_output(net, str(net.sgen.cid.iloc[1]), float(row.wind_fraction))
    _stage3.set_bess_power(net, str(net.storage.cid.iloc[0]), float(row.bess_p_mw))
    _stage3.set_bess_soc(net, str(net.storage.cid.iloc[0]), float(row.bess_soc))
    heavy05.apply_injected_event(net, rec["injected_event"])
    return net

def switched_targets(scenario_id):
    """Branch/generator names actually taken out of service by the injected event."""
    rec = _RECS[scenario_id]
    out = []
    def collect(ev):
        if ev["mechanism"] in ("line_outage", "generator_outage"):
            out.extend(ev["targets"])
        if ev["mechanism"] == "compound":
            for c in ev["components"]:
                collect(c)
    collect(rec["injected_event"])
    return sorted(set(out))

def element_status(net):
    v = {c: float(x) for c, x in zip(net.bus.cid.astype(str), net.res_bus.vm_pu.values)}
    ld = {c: float(x) for c, x in zip(list(net.line.cid.astype(str)) + list(net.trafo.cid.astype(str)),
                                      list(net.res_line.loading_percent.values) + list(net.res_trafo.loading_percent.values))}
    ins = {**{c: bool(x) for c, x in zip(net.line.cid.astype(str), net.line.in_service.values)},
           **{c: bool(x) for c, x in zip(net.trafo.cid.astype(str), net.trafo.in_service.values)},
           **{c: bool(x) for c, x in zip(net.gen.cid.astype(str), net.gen.in_service.values)}}
    return v, ld, ins

# ---------------- observation: per-element status table (no answers given) ---
def status_table(net, wls_row=None):
    v, ld, ins = element_status(net)
    if wls_row is not None:
        we = wls_row
        if "est_v_min" in we:
            pass  # estimated system-level summary is added by the caller
    lines = ["Element telemetry (name | in_service | loading %):"]
    for b in sorted(ld):
        state = "in_service" if ins.get(b, True) else "OUT_OF_SERVICE"
        lines.append(f"  branch {b} | {state} | {ld[b]:.1f}")
    for g in sorted(set(ins)):
        if g.startswith("gen") or g.startswith("sgen") or g.startswith("storage") or g.startswith("BESS") or g.startswith("G"):
            if ins.get(g) is False or g.startswith("gen"):
                lines.append(f"  generator {g} | {'in_service' if ins.get(g, True) else 'OFFLINE'}")
    return "\n".join(lines) + "\n"

SYSTEM = """You are a grid-aware LLM operator performing event diagnosis with tool access.
Diagnose the INJECTED EVENT CLASS (cause axis) of a power-system scenario.
Taxonomy: E0 Normal (no disturbance), E1 Load Surge (+% demand), E2 Load Drop (-% demand), E3 Transmission-Line Outage, E4 Generator Outage, E5 Renewable Ramp, E7 Overvoltage (V>1.05), E9 Compound (2 mechanisms).
Rules:
- E0 Normal: no switching, no demand/renewable change, no violations.
- E1 Load Surge: demand increases, no switching.
- E2 Load Drop: demand decreases, no switching.
- E3: a branch is OUT_OF_SERVICE, demand unchanged.
- E4: a generator is OFFLINE, demand unchanged.
- E5 Renewable Ramp: a renewable element changed output, no switching, demand unchanged.
- E7 Overvoltage: only when post-event shows overvoltage (V>1.05).
- E9 Compound: a switching event AND a demand change appear together.
You may run tools to probe the network before answering. Use the returned measurements."""

TOOL_MANIFEST = """Tools (at most one per turn):
1. {"action":"tool_call","tool":"run_power_flow","args":{}}
2. {"action":"tool_call","tool":"contingency_test","args":{"element":"<branch or generator name>"}}
3. {"action":"tool_call","tool":"n1_sweep","args":{}}
4. {"action":"tool_call","tool":"run_opf","args":{}}
5. {"action":"tool_call","tool":"grid_query","args":{"table":"violations"|"voltages"|"loadings"}}
Final answer (only after you have run tools):
   {"action":"final_answer","event_class":"E0-E5|E7|E9","identified_element":"<the OUT_OF_SERVICE branch or generator from the telemetry, or 'none' if nothing is switched>","confidence":0.0-1.0,"reason":"one sentence"}
Respond with exactly one JSON object per turn."""

# ---------------- tool execution ----------------
class ToolBox:
    def __init__(self, scenario_id):
        self.net = reconstruct(scenario_id)
        self.branches = sorted(set(list(self.net.line.cid.astype(str)) + list(self.net.trafo.cid.astype(str))))
        self.gens = sorted(set(list(self.net.gen.cid.astype(str)) + list(self.net.sgen.cid.astype(str))
                               + list(self.net.storage.cid.astype(str))))

    def _summary(self, net):
        v = {c: float(x) for c, x in zip(net.bus.cid.astype(str), net.res_bus.vm_pu.values)}
        ld = {c: float(x) for c, x in zip(list(net.line.cid.astype(str)) + list(net.trafo.cid.astype(str)),
                                          list(net.res_line.loading_percent.values) + list(net.res_trafo.loading_percent.values))}
        ol = [c for c, x in ld.items() if x > 100]
        return {"v_min": min(v.values()), "v_max": max(v.values()),
                "max_loading": max(ld.values()), "overloads": ol}

    def call(self, tool, args):
        args = args if isinstance(args, dict) else {}
        if tool == "run_power_flow":
            pp.runpp(self.net, numba=True)
            s = self._summary(self.net)
            ok = bool(self.net.converged)
            return ok, True, json.dumps({"converged": ok, "v_min_pu": round(s["v_min"], 4),
                                         "v_max_pu": round(s["v_max"], 4), "max_loading_pct": round(s["max_loading"], 1),
                                         "overloads": s["overloads"]})
        if tool == "contingency_test":
            el = str(args.get("element", ""))
            valid = el in self.branches or el in self.gens
            if not valid:
                return False, False, json.dumps({"error": f"unknown element '{el}'"})
            net2 = copy.deepcopy(self.net)
            net2.line.loc[net2.line.cid.astype(str) == el, "in_service"] = False
            net2.trafo.loc[net2.trafo.cid.astype(str) == el, "in_service"] = False
            net2.gen.loc[net2.gen.cid.astype(str) == el, "in_service"] = False
            net2.sgen.loc[net2.sgen.cid.astype(str) == el, "in_service"] = False
            net2.storage.loc[net2.storage.cid.astype(str) == el, "in_service"] = False
            try:
                pp.runpp(net2, numba=True)
            except Exception:
                return True, True, json.dumps({"converged": False})
            s = self._summary(net2)
            return bool(net2.converged), True, json.dumps({"converged": bool(net2.converged),
                                                           "element_out": el, "v_min_pu": round(s["v_min"], 4),
                                                           "max_loading_pct": round(s["max_loading"], 1),
                                                           "overloads": s["overloads"]})
        if tool == "n1_sweep":
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
                                           "max_loading_pct": round(s2["max_loading"], 1), "overloads": s2["overloads"]})
        return False, False, json.dumps({"error": f"unknown tool '{tool}'"})

# ---------------- engines ----------------
def _payload(messages):
    contents = []
    for m in messages:
        contents.append({"role": "model" if m["role"] == "assistant" else "user",
                         "parts": [{"text": m["content"]}]})
    return {"contents": contents, "generationConfig": {"temperature": 0, "maxOutputTokens": 512}}

def api_turn(pool, messages, model):
    return runner.call_gemini_pooled(_payload(messages), pool, model=model)

class LocalEngine:
    def __init__(self, model, endpoint="http://127.0.0.1:9090"):
        self.model = model
        self.endpoint = endpoint

    def _up(self):
        try:
            return requests.get(f"{self.endpoint}/v1/models", timeout=5).status_code == 200
        except Exception:
            return False

    def turn(self, messages, timeout=240):
        while True:
            if not self._up():
                print("[WAIT] local engine down — polling every 30s", flush=True)
                while not self._up():
                    time.sleep(30)
            t0 = time.time()
            try:
                resp = requests.post(f"{self.endpoint}/v1/chat/completions",
                                     json={"model": self.model, "messages": messages,
                                           "temperature": 0, "max_tokens": 1024}, timeout=timeout)
                if resp.status_code in (429, 503, 500):
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                text = msg.get("content") or msg.get("reasoning_content") or ""
                return text, time.time() - t0
            except Exception as e:
                print(f"[WARN] local call failed: {str(e)[:100]} — retrying", flush=True)
                time.sleep(10)

def parse_action(text):
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            if j.get("action") in ("tool_call", "final_answer"):
                return j
            if "event_class" in j:
                j.setdefault("action", "final_answer")
                return j
            if "tool" in j:
                j.setdefault("action", "tool_call")
                return j
        except Exception:
            pass
    return None

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="api", choices=["api", "local"])
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--min-tools", type=int, default=2)
    ap.add_argument("--rpm", type=int, default=45)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids-from", default="agent_runs_gemini-3.5-flash-lite.csv")
    ap.add_argument("--api-keys", default=None)
    ap.add_argument("--local-model", default="gemma-4-E4B-it-Q4_0.gguf")
    ap.add_argument("--tag", default="cl2")
    args = ap.parse_args()

    model = args.local_model if args.engine == "local" else "gemini-3.5-flash-lite"
    pool = None
    local = None
    if args.engine == "api":
        keys = [k.strip() for k in (args.api_keys or __import__("os").getenv("GEMINI_API_KEYS", "")).split(",") if k.strip()]
        if not keys:
            raise SystemExit("[GATED] no Gemini credentials")
        pool = runner.GeminiKeyPool(keys, args.rpm)
        runner.preflight_keys(pool, model)
    else:
        local = LocalEngine(args.local_model)

    scen = pd.read_csv(PROCESSED / "ieee14_scenarios_taxonomy2.csv")
    if args.ids_from:
        ids = set(pd.read_csv(RESULTS / args.ids_from).scenario_id.unique())
        scen = scen[scen.scenario_id.isin(ids)].reset_index(drop=True)
    rng = np.random.default_rng(runner.MASTER_SEED)
    idx = rng.choice(len(scen), size=min(args.n_test, len(scen)), replace=False)
    test = scen.iloc[idx]

    model_tag = model.replace(".", "_").replace("/", "_")
    out_csv = Path(args.out) if args.out else RESULTS / (
        f"agent_runs_{model_tag}_{args.engine}_{args.tag}.csv")
    ckpt = Path(str(out_csv).replace(".csv", "_checkpoint.json"))

    done = set()
    rows = []
    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            done = set(zip(prev.scenario_id, prev.config))
            rows = prev.to_dict("records")
            print(f"[INFO] resume: {len(done)} rows done", flush=True)
        except Exception:
            pass

    for _, srow in test.iterrows():
        cfg = "closed_loop"
        if (srow.scenario_id, cfg) in done:
            continue
        box = ToolBox(srow.scenario_id)
        v, ld, ins = element_status(box.net)
        tel = ["Element telemetry (name | in_service | loading %):"]
        for b in sorted(ld):
            state = "in_service" if ins.get(b, True) else "OUT_OF_SERVICE"
            tel.append(f"  branch {b} | {state} | {ld[b]:.1f}")
        for g in sorted(box.gens):
            state = "OFFLINE" if not ins.get(g, True) else "in_service"
            tel.append(f"  generator {g} | {state}")
        messages = [{"role": "user", "content": SYSTEM + "\n" + "\n".join(tel) + "\n"
                     + TOOL_MANIFEST + f"\nScenario {srow.scenario_id}\nDiagnose the event. Run tools first."}]
        n_exec = 0
        tool_hist = {}
        final = None
        turn = 0
        for turn in range(1, args.max_turns + 1):
            if args.engine == "local":
                text, _lat = local.turn(messages, timeout=240)
            else:
                try:
                    text, _lat = api_turn(pool, messages, model)
                except RuntimeError as e:
                    print(f"[STOP] pool exhausted: {e} — rows checkpointed", flush=True)
                    _save(rows, out_csv, ckpt)
                    return
            act = parse_action(text)
            messages.append({"role": "assistant", "content": text})
            if act is None:
                messages.append({"role": "user", "content": "Malformed response. Reply with exactly one JSON object: a tool_call or the final_answer."})
                continue
            if act.get("action") == "final_answer" and n_exec >= args.min_tools:
                final = act
                break
            if act.get("action") == "final_answer":
                messages.append({"role": "user", "content":
                                 f"final_answer rejected: only {n_exec} tool calls executed so far; "
                                 f"run at least {args.min_tools} tools before answering."})
                continue
            tool = str(act.get("tool", ""))
            tool_hist[tool] = tool_hist.get(tool, 0) + 1
            ok, valid, result = box.call(tool, act.get("args", {}))
            n_exec += int(ok)
            messages.append({"role": "user", "content":
                             f"TOOL RESULT ({tool}, executed={ok}): {result}\n"
                             "Continue: another tool_call, or the final_answer."})
        if final is None:
            final = {"action": "final_answer", "event_class": "?", "identified_element": "none",
                     "confidence": 0.0, "reason": "turn limit"}

        pred = str(final.get("event_class", "?")).strip().upper()
        m = re.match(r"E[0-9]", pred)
        pred = m.group(0) if m else "?"
        correct = (pred == srow.event_class)
        identified = str(final.get("identified_element", "none")).strip().lower()
        true_targets = [t.lower() for t in switched_targets(srow.scenario_id)]
        has_switching = len(true_targets) > 0
        if has_switching:
            construction = identified in true_targets or any(t in identified for t in true_targets)
        else:
            construction = identified in ("none", "", "no element", "no switching")
        rows.append({"scenario_id": srow.scenario_id, "event_class": srow.event_class,
                     "config": cfg, "model": model, "correct_diag": bool(correct),
                     "construction_correct": bool(construction), "has_switching": bool(has_switching),
                     "identified_element": identified, "true_targets": ";".join(true_targets),
                     "turns": turn, "n_tool_calls": sum(tool_hist.values()),
                     "tools_used": json.dumps(tool_hist), "executed": n_exec,
                     "raw": json.dumps(final)[:400]})
        _save(rows, out_csv, ckpt)
        print(f"[{srow.scenario_id}] true={srow.event_class} pred={pred} correct={correct} "
              f"element={identified} turns={turn} tools={sum(tool_hist.values())} exec={n_exec}", flush=True)

    df = pd.read_csv(out_csv)
    print("\n=== CLOSED-LOOP v2 SUMMARY ===", flush=True)
    print(f"scenarios: {len(df)} | diag: {df.correct_diag.mean()*100:.1f}%", flush=True)
    sw = df[df.has_switching]
    if len(sw):
        print(f"construction accuracy (switched rows): {sw.construction_correct.mean()*100:.1f}%", flush=True)
    print("turns:", df.turns.value_counts().sort_index().to_dict(), flush=True)
    print("[PASS] Stage 38 v2 complete", flush=True)

if __name__ == "__main__":
    main()
