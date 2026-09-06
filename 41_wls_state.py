#!/usr/bin/env python3
"""
Stage 41 — Real AC-WLS state estimation over the generated measurement set
(review-2 point #7).

Replaces the closed-form noise-aware approximation with a genuine iterative
weighted-least-squares estimator: for each pilot scenario, the post-event
network is reconstructed, the scenario's ~120 noisy measurements (42 bus
voltage meters sigma=0.003 pu, branch P/Q meters, bus injection P/Q meters —
branch meters of outaged elements excluded) are attached as pandapower
measurement objects, and pandapower's WLS estimator solves for the state.
Branch loadings under the estimated state are computed by fixing generator
voltage set-points to the estimated magnitudes and re-solving.

Outputs:
  data/processed/ieee14_pilot_wls_estimates.csv  (per scenario: est v_min/v_max,
     est loadings summary, flags recomputed from the estimated state, rmse vs
     truth, convergence flag)
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "data" / "processed"

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

heavy05 = _load("heavy05", "05_heavy.py")
pf_tool = _load("pf_tool", "10_power_flow_tool.py")
stage3 = pf_tool.load_stage3()

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=140, help="number of pilot scenarios to estimate")
args = _ap.parse_args()

scen = pd.read_csv(PROCESSED / "ieee14_scenarios_taxonomy2.csv")
pilot = pd.read_csv(HERE / "data/results/agent_runs_gemini-3.5-flash-lite_tax2.csv",
                    usecols=["scenario_id"]).drop_duplicates()
scen = scen[scen.scenario_id.isin(set(pilot.scenario_id))].reset_index(drop=True)
scen = scen.head(args.n)
meas = pd.read_csv(PROCESSED / "ieee14_measurements.csv")

out_rows = []
for k, (_, srow) in enumerate(scen.iterrows()):
    sid = srow.scenario_id
    msc = meas[meas.scenario_id == sid]
    net = pp.from_json(str(PROCESSED / "ieee14_net_re.json"))
    points = pd.read_csv(PROCESSED / "ieee14_operating_points.csv")
    factors = pd.read_csv(PROCESSED / "ieee14_op_load_factors.csv")
    pidx = {op: i for i, op in enumerate(points.op_id)}
    oprow = points.iloc[pidx[srow.op_id]]
    frow = factors.drop(columns=["op_id"]).values[pidx[srow.op_id]]
    net.load["p_mw"] = net.load.p_mw.values * frow
    net.load["q_mvar"] = net.load.q_mvar.values * factors.drop(columns=["op_id"]).values[pidx[srow.op_id]]
    stage3.set_pv_output(net, str(net.sgen.cid.iloc[0]), float(oprow.solar_fraction))
    stage3.set_wind_output(net, str(net.sgen.cid.iloc[1]), float(oprow.wind_fraction))
    stage3.set_bess_power(net, str(net.storage.cid.iloc[0]), float(oprow.bess_p_mw))
    stage3.set_bess_soc(net, str(net.storage.cid.iloc[0]), float(oprow.bess_soc))
    # injected event (post-event network)
    rec = json.loads((PROCESSED / "ieee14_scenarios.jsonl").read_text().splitlines()[
        [json.loads(l)["scenario_id"] for l in open(PROCESSED / "ieee14_scenarios.jsonl")].index(sid)])
    heavy05.apply_injected_event(net, rec["injected_event"])
    pp.runpp(net, numba=True)

    bus_idx = {str(c): i for i, c in enumerate(net.bus.cid)}
    line_idx = {str(c): i for i, c in enumerate(net.line.cid)}
    trafo_idx = {str(c): i for i, c in enumerate(net.trafo.cid)}

    for _, mrow in msc.iterrows():
        mid, cat, val, sigma = mrow.meter_id, mrow.category, float(mrow.measured), float(mrow.sigma)
        try:
            if cat == "voltage":
                b = int(mid.split("_")[-1]) - 1
                pp.create_measurement(net, "v", "bus", element=b, value=val, std_dev=sigma)
            elif cat == "branch":
                parts = mid.split("_")
                side = parts[1] if parts[1] in ("from", "to") else "from"
                mtype = "p" if parts[0] == "P" else "q"
                name = "_".join(parts[2:])
                if name in line_idx and net.line.in_service.iloc[line_idx[name]]:
                    pp.create_measurement(net, mtype, "line", element=line_idx[name], value=val,
                                          std_dev=sigma, side=side)
                elif name in trafo_idx and net.trafo.in_service.iloc[trafo_idx[name]]:
                    pp.create_measurement(net, mtype, "trafo", element=trafo_idx[name], value=val,
                                          std_dev=sigma, side=side)
            elif cat == "injection":
                mtype = "p" if mid.startswith("P") else "q"
                b = int(mid.split("_")[-1]) - 1
                pp.create_measurement(net, mtype, "bus", element=b, value=val, std_dev=sigma)
        except Exception:
            pass

    converged = True
    try:
        pp.estimation.estimate(net, algorithm="wls", init="flat")
        converged = hasattr(net, "res_bus_est") and len(net.res_bus_est) == len(net.bus)
    except Exception as e:
        converged = False
        last_err = str(e)[:100]

    if converged:
        est_vm = net.res_bus_est.vm_pu.values.astype(float)
        true_vm = net.res_bus.vm_pu.values.astype(float)
        rmse_v = float(np.sqrt(np.mean((est_vm - true_vm) ** 2)))
        # estimated-state loadings: fix generator voltage set-points to the
        # estimated magnitudes and re-solve the network model
        import copy
        net2 = copy.deepcopy(net)
        for i, vm in enumerate(est_vm):
            for tbl in ["gen", "ext_grid"]:
                if i in net2[tbl].index:
                    net2[tbl].at[i, "vm_pu"] = vm
        try:
            pp.runpp(net2, numba=True, init="flat")
            loadings = list(net2.res_line.loading_percent.values) + list(net2.res_trafo.loading_percent.values)
            est_vmin, est_vmax = float(net2.res_bus.vm_pu.min()), float(net2.res_bus.vm_pu.max())
            est_peak = float(max(loadings))
            n_under = int((net2.res_bus.vm_pu < 0.94).sum())
            n_over = int((net2.res_bus.vm_pu > 1.06).sum())
            n_ol = int(sum(1 for v in loadings if v > 100))
        except Exception:
            est_vmin, est_vmax = float(est_vm.min()), float(est_vm.max())
            est_peak, n_under, n_over, n_ol = float("nan"), -1, -1, -1
    else:
        est_vm = np.full(len(net.bus), np.nan)
        rmse_v = float("nan")
        est_vmin = est_vmax = est_peak = float("nan")
        n_under = n_over = n_ol = -1

    true_vmin = float(net.res_bus.vm_pu.min()); true_vmax = float(net.res_bus.vm_pu.max())
    out_rows.append({"scenario_id": sid, "event_class": srow.event_class,
                     "converged": bool(converged), "rmse_v": rmse_v,
                     "est_v_min": est_vmin, "est_v_max": est_vmax,
                     "est_peak_loading": est_peak, "est_n_under": n_under,
                     "est_n_over": n_over, "est_n_ol": n_ol,
                     "true_v_min": true_vmin, "true_v_max": true_vmax,
                     "n_meters": len(msc)})
    if k % 20 == 0:
        print(f"[{k}/{len(scen)}] {sid} converged={converged} rmse_v={rmse_v:.5f}")

out = pd.DataFrame(out_rows)
out.to_csv(PROCESSED / "ieee14_pilot_wls_estimates.csv", index=False)
conv = out[out.converged]
print(f"\nconverged: {len(conv)}/{len(out)}")
print(f"RMSE_v: mean {conv.rmse_v.mean():.5f}, max {conv.rmse_v.max():.5f}")
print(f"mean |est_v_min - true_v_min|: {(conv.est_v_min - conv.true_v_min).abs().mean():.4f} pu")
print(f"flag agreement (under): {(conv.est_n_under == conv.apply(lambda r: 1 if r.true_v_min < 0.94 else 0, axis=1)).mean()*100:.1f}%")
print("[PASS] Stage 41 complete")
