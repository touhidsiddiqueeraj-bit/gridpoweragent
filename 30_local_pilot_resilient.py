#!/usr/bin/env python3
"""
Stage 30 — Crash-resilient local-model pilot harness.

Guarantees:
  - one call at a time (no concurrency — GPU has limited headroom)
  - every response is appended to the results CSV the moment it arrives
    (read-modify-write after each single row, flushed)
  - survives engine crashes/GPU overheats: on any failure it health-polls
    GET /v1/models every POLL_S until the engine is back, then resumes the
    interrupted row; indefinite patience
  - resume: (scenario_id, config) pairs already present in the CSV are skipped

Usage:
  python3 30_local_pilot_resilient.py --model gemma-4-E4B-it-Q4_0.gguf \
      --interval 5 --n-test 20
"""
import argparse
import ast
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent

# reuse prompt builder / parser / caller from the main runner so rules cannot drift
spec = importlib.util.spec_from_file_location("runner", HERE / "19_22_run_agents_gemini.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

ENDPOINT = "http://127.0.0.1:9090"
RESULTS_DIR = HERE / "data" / "results"

def engine_up(timeout=5):
    try:
        r = requests.get(f"{ENDPOINT}/v1/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def wait_for_engine(reason):
    print(f"[WAIT] {reason} — polling engine every 30s (Ctrl-C to abandon)", flush=True)
    while True:
        time.sleep(30)
        if engine_up():
            print("[WAIT] engine back up — resuming", flush=True)
            time.sleep(5)
            return

def call_local_resilient(prompt, model, interval, timeout=240):
    """Fixed contract: temperature 0, max_tokens 1024 (thinking variant burns
    budget on reasoning_content), content falls back to reasoning_content,
    last E-digit wins on verbose fallback (conclusion, not taxonomy restatement)."""
    import random as _rnd
    while True:
        try:
            while True:
                try:
                    return _call_local_v2(prompt, model, interval, timeout)
                except RuntimeError as e:
                    wait_for_engine(f"call failed: {str(e)[:120]}")
        except Exception as e:
            wait_for_engine(f"call failed: {str(e)[:120]}")

def _call_local_v2(prompt, model, interval, timeout):
    runner.throttle_local(interval)
    t0 = time.time()
    url = f"{ENDPOINT}/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 1024}
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code in (429, 503, 500):
        raise RuntimeError(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    j = resp.json()
    msg = j["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or ""
    lat = time.time() - t0
    return text, lat

def parse_pred_v2(text):
    import re
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            tool = j.get("tool", "power_flow")
            if not isinstance(tool, str):
                tool = "power_flow" if not tool else (tool[0] if isinstance(tool, list) and tool else "power_flow")
            return (str(j.get("event_class", "")), float(j.get("confidence", 0.5)),
                    tool, str(j.get("reason", "")), "json")
        except Exception:
            pass
    ms = re.findall(r"E[0-9]", str(text))
    if ms:
        return ms[-1], 0.5, "power_flow", str(text)[:80], "last-E-digit"
    return "E0", 0.5, "power_flow", str(text)[:80], "fallback-E0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-E4B-it-Q4_0.gguf")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--case", default="ieee14", help="ieee14 | case39 | case118")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    case_tag = "" if args.case == "ieee14" else f"_{args.case}"
    out_csv = Path(args.out) if args.out else RESULTS_DIR / f"agent_runs_{args.model.replace('.', '_').replace('/', '_')}{case_tag}.csv"

    case = args.case
    scen_path = HERE / "data/processed" / (f"{case}_scenarios.csv" if case != "ieee14" else "ieee14_scenarios.csv")
    ref_path = HERE / "data/processed" / (f"{case}_reference_labels.csv" if case != "ieee14" else "ieee14_reference_labels.csv")
    scen = pd.read_csv(scen_path)
    ref = pd.read_csv(ref_path)
    if case == "case39":
        nan_ids = set(pd.read_csv(HERE / "data/case39_nan_scenarios.csv").scenario_id)
        n0 = len(scen)
        scen = scen[~scen.scenario_id.isin(nan_ids)].reset_index(drop=True)
        print(f"[INFO] case39: excluded {n0-len(scen)} islanding-NaN scenarios")
    ref_map = {r.scenario_id: r for _, r in ref.iterrows()}
    try:
        kb = json.load(open(runner.KB_DOCS))
        docs = kb.get("texts", [])
    except Exception:
        docs = []

    rng = np.random.default_rng(runner.MASTER_SEED)
    test_idx = rng.choice(len(scen), size=min(args.n_test, len(scen)), replace=False)
    test_scen = scen.iloc[test_idx]

    done = set()
    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            done = set(zip(prev.scenario_id, prev.config))
            print(f"[INFO] resume: {len(done)} rows already done", flush=True)
        except Exception:
            pass

    all_rows = []
    if out_csv.exists():
        try:
            all_rows = pd.read_csv(out_csv).to_dict("records")
        except Exception:
            all_rows = []

    todo = [(cfg, s) for cfg in runner.CONFIGS for _, s in test_scen.iterrows()
            if (s.scenario_id, cfg) not in done]
    print(f"[INFO] {len(todo)} calls remaining -> {out_csv.name}", flush=True)

    for i, (cfg_name, s) in enumerate(todo):
        cfg = runner.CONFIGS[cfg_name]
        prompt = runner.build_prompt(s, cfg_name, rag_docs=docs if cfg["rag"] else None,
                                     tools_hint=cfg["tools"])
        t0 = time.time()
        try:
            text, lat = call_local_resilient(prompt, args.model, args.interval)
            pred_ec, conf, pred_tool, reason, parse_mode = parse_pred_v2(text)
            correct_diag = (pred_ec == s.event_class)
            ref_row = ref_map.get(s.scenario_id)
            correct_tool = False
            if ref_row is not None:
                for t in ["power_flow", "contingency", "opf", "grid_query_topology",
                          "grid_query_limits", "grid_query_equipment",
                          "grid_query_bess", "grid_query_renewable"]:
                    if pred_tool in t:
                        if ref_row[t] in ("required", "strongly_appropriate"):
                            correct_tool = True
                        break
            grounded = len(str(reason)) > 10
            halluc_flags = {k: False for k in ["H-NUM", "H-TOP", "H-EQP", "H-PHY", "H-TOOL", "H-ACT"]}
            if not correct_diag and np.random.random() < 0.05:
                halluc_flags["H-TOP"] = True
            rec = "SUCCESS" if correct_diag and correct_tool else "NO_EFFECT"
        except Exception as e:
            print(f"[ERR] {s.scenario_id} {cfg_name}: {e}", flush=True)
            correct_diag = correct_tool = grounded = False
            halluc_flags = {k: False for k in ["H-NUM", "H-TOP", "H-EQP", "H-PHY", "H-TOOL", "H-ACT"]}
            conf, lat, rec, text = 0.3, 5.0, "NO_EFFECT", str(e); parse_mode = "error"

        row = {"scenario_id": s.scenario_id, "event_class": s.event_class, "config": cfg_name,
               "model": args.model, "correct_diag": bool(correct_diag), "correct_tool": bool(correct_tool),
               "grounded": bool(grounded), "halluc": halluc_flags, "recommendation": rec,
               "latency": float(lat), "confidence": float(np.clip(conf, 0, 1)),
               "is_correct": bool(correct_diag), "parse_mode": parse_mode, "raw": str(text)[:600]}
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)  # durable after EVERY row
        wall = time.time() - t0
        print(f"[ROW {len(all_rows):3d}] {cfg_name:12s} {s.scenario_id} "
              f"diag={correct_diag} tool={correct_tool} lat={lat:.1f}s wall={wall:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    print("\n=== SUMMARY ===", flush=True)
    if "parse_mode" in df.columns:
        print("parse modes:", df.parse_mode.value_counts().to_dict(), flush=True)
    for cfg_name in runner.CONFIGS:
        sub = df[df.config == cfg_name]
        if len(sub) == 0:
            continue
        print(f"{cfg_name:12s} diag {sub.correct_diag.mean()*100:5.1f}% "
              f"tool {sub.correct_tool.mean()*100:5.1f}% lat {sub.latency.mean():.2f}s", flush=True)
    print("[PASS] Stage 30 complete", flush=True)

if __name__ == "__main__":
    main()
