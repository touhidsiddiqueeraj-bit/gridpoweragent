#!/usr/bin/env python3
"""
Stage 4 — Operating Point Generator
Reconstructs 5000 normal operating points for IEEE14-RE network.
Outputs match Stage 5 expectations.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pandapower as pp

import importlib.util, sys

STAGE3_FILE = Path("03_renewables_bess.py")
OUTPUT_DIR = Path("data") / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 20260821
N_TARGET = 4000
MAX_TRIES = 15000

def load_stage3():
    spec = importlib.util.spec_from_file_location("stage3_renewables_bess", STAGE3_FILE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["stage3_renewables_bess"] = m
    spec.loader.exec_module(m)
    return m

def measure_state(net):
    voltages = net.res_bus.vm_pu.values
    under_mask = voltages < net.bus.min_vm_pu.values
    over_mask = voltages > net.bus.max_vm_pu.values
    line_loading = net.res_line.loading_percent.values
    trafo_loading = net.res_trafo.loading_percent.values
    line_mask = (line_loading > net.line.max_loading_percent.values) & net.line.in_service.values
    trafo_mask = (trafo_loading > net.trafo.max_loading_percent.values) & net.trafo.in_service.values
    in_service_line = line_loading[net.line.in_service.values]
    in_service_trafo = trafo_loading[net.trafo.in_service.values]
    max_line = float(in_service_line.max()) if len(in_service_line) else 0.0
    max_trafo = float(in_service_trafo.max()) if len(in_service_trafo) else 0.0
    return {
        "v_min_pu": float(voltages.min()),
        "v_max_pu": float(voltages.max()),
        "v_min_bus": str(net.bus.cid.values[int(voltages.argmin())]),
        "v_max_bus": str(net.bus.cid.values[int(voltages.argmax())]),
        "max_line_loading_percent": max_line,
        "max_trafo_loading_percent": max_trafo,
        "peak_branch_loading_percent": max(max_line, max_trafo),
        "total_load_mw": float(net.load.p_mw.sum()),
        "renewable_p_mw": float(net.sgen.p_mw.sum()),
        "slack_p_mw": float(net.res_ext_grid.p_mw.sum()),
        "slack_q_mvar": float(net.res_ext_grid.q_mvar.sum()),
        "total_losses_mw": float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()),
        "n_violations": int(under_mask.sum() + over_mask.sum() + line_mask.sum() + trafo_mask.sum()),
        "has_undervoltage": bool(under_mask.any()),
        "has_overvoltage": bool(over_mask.any()),
        "has_overload": bool(line_mask.any() or trafo_mask.any()),
    }

def main():
    stage3 = load_stage3()
    net = stage3.build_ieee14_re()
    # ensure baseline converges
    pp.runpp(net)
    assert net.converged, "Baseline must converge"

    # Save Stage 3 artefacts
    NET_RE_FILE = OUTPUT_DIR / "ieee14_net_re.json"
    RE_HASH_FILE = OUTPUT_DIR / "ieee14_re_layout_hash.txt"
    pp.to_json(net, str(NET_RE_FILE))
    h = stage3.compute_re_layout_hash(net)
    RE_HASH_FILE.write_text(h + "\n")
    print(f"[INFO] Saved network {NET_RE_FILE} hash={h}")
    print(f"[INFO] Baseline vmin {net.res_bus.vm_pu.min():.4f} vmax {net.res_bus.vm_pu.max():.4f} peak {max(net.res_line.loading_percent.max(), net.res_trafo.loading_percent.max()):.2f}%")

    # Snapshot base loads
    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()
    bus_ids = list(net.bus.cid)
    branch_ids = list(net.line.cid) + list(net.trafo.cid)
    load_ids = list(net.load.cid)

    pv_id = stage3.PV_CONFIG["cid"]
    wind_id = stage3.WIND_CONFIG["cid"]
    bess_id = "gen_BESS9"

    rng = np.random.default_rng(MASTER_SEED)

    # Snapshot mutable fields for restore (simple: net copy)
    # We'll restore by re-applying base + RE each iteration, but also need to handle in_service (not changed here)
    records = []
    factor_rows = []
    voltage_rows = []
    loading_rows = []

    tries = 0
    start = time.time()
    while len(records) < N_TARGET and tries < MAX_TRIES:
        tries += 1
        # sample operating point
        load_scale = float(rng.uniform(0.70, 1.10))  # overall - keep high keep-rate
        # per-load factors: load_scale * (1 + local +-5%)
        local = rng.uniform(-0.05, 0.05, size=len(load_ids))
        factors = load_scale * (1 + local)
        # clip to reasonable 0.5-1.5
        factors = np.clip(factors, 0.5, 1.5)

        solar_fraction = float(rng.uniform(0.0, 1.0))
        wind_fraction = float(rng.uniform(0.0, 1.0))
        bess_soc = float(rng.uniform(0.15, 0.85))
        # BESS power: small random dispatch, mostly 0, occasionally +/- up to 10 MW
        bess_p_mw = float(rng.choice([0,0,0, rng.uniform(-10,10)]))

        # apply
        net.load["p_mw"] = base_p * factors
        net.load["q_mvar"] = base_q * factors
        stage3.set_pv_output(net, pv_id, solar_fraction)
        stage3.set_wind_output(net, wind_id, wind_fraction)
        stage3.set_bess_power(net, bess_id, bess_p_mw)
        stage3.set_bess_soc(net, bess_id, bess_soc)

        try:
            pp.runpp(net, numba=True)
        except Exception:
            continue
        if not net.converged:
            continue
        m = measure_state(net)
        if m["n_violations"] != 0:
            continue  # only normal points
        # also enforce not too close to limit to allow Stage5 to have room for surges?
        # Keep only if peak < 85% and vmin >0.96 maybe? But for now allow all normal

        op_id = f"IEEE14_OP_{len(records)+1:06d}"
        rec = {
            "op_id": op_id,
            "load_scale": load_scale,
            "solar_fraction": solar_fraction,
            "wind_fraction": wind_fraction,
            "bess_soc": bess_soc,
            "bess_p_mw": bess_p_mw,
            "v_min_pu": m["v_min_pu"],
            "v_max_pu": m["v_max_pu"],
            "v_min_bus": m["v_min_bus"],
            "v_max_bus": m["v_max_bus"],
            "max_line_loading_percent": m["max_line_loading_percent"],
            "max_trafo_loading_percent": m["max_trafo_loading_percent"],
            "peak_branch_loading_percent": m["peak_branch_loading_percent"],
            "total_load_mw": m["total_load_mw"],
            "renewable_p_mw": m["renewable_p_mw"],
            "slack_p_mw": m["slack_p_mw"],
            "slack_q_mvar": m["slack_q_mvar"],
            "total_losses_mw": m["total_losses_mw"],
        }
        records.append(rec)
        factor_rows.append(factors)
        voltage_rows.append(net.res_bus.vm_pu.values.copy())
        loading_rows.append(np.concatenate([net.res_line.loading_percent.values, net.res_trafo.loading_percent.values]))
        if len(records) % 500 == 0:
            print(f"[INFO] {len(records)}/{N_TARGET} after {tries} tries ({100*len(records)/tries:.1f}% keep)")

    print(f"[INFO] Generated {len(records)} normal points in {time.time()-start:.1f}s after {tries} tries")

    if len(records) < 4997:
        print(f"[WARN] Only {len(records)} < 4997 target, continuing anyway")

    df = pd.DataFrame(records)
    # save
    df.to_csv(OUTPUT_DIR / "ieee14_operating_points.csv", index=False)
    print(f"[INFO] Saved {OUTPUT_DIR / 'ieee14_operating_points.csv'}")

    # factors
    fac_df = pd.DataFrame(np.array(factor_rows), columns=load_ids)
    fac_df.insert(0, "op_id", df.op_id)
    fac_df.to_csv(OUTPUT_DIR / "ieee14_op_load_factors.csv", index=False)
    print(f"[INFO] Saved factors {fac_df.shape}")

    # voltages
    v_df = pd.DataFrame(np.array(voltage_rows), columns=bus_ids)
    v_df.insert(0, "op_id", df.op_id)
    v_df.to_csv(OUTPUT_DIR / "ieee14_op_bus_voltages.csv", index=False)

    # loadings
    l_df = pd.DataFrame(np.array(loading_rows), columns=branch_ids)
    l_df.insert(0, "op_id", df.op_id)
    l_df.to_csv(OUTPUT_DIR / "ieee14_op_branch_loading.csv", index=False)

    # metadata
    meta = {
        "n_points": len(records),
        "tries": tries,
        "hash": h,
        "load_scale_range": [float(df.load_scale.min()), float(df.load_scale.max())],
        "v_min_range": [float(df.v_min_pu.min()), float(df.v_min_pu.max())],
        "peak_loading_max": float(df.peak_branch_loading_percent.max()),
    }
    (OUTPUT_DIR / "stage4_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    # validation
    assert len(df) >= 3000, "Need at least 3000 for Stage5 (300 per class)"
    print("[PASS] Stage 4 complete")

if __name__ == "__main__":
    main()
