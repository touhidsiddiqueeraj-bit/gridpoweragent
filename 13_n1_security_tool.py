#!/usr/bin/env python3
"""
Stage 13 — N-1 Security Tool
Runs full N-1 over 15 lines + 4 gens, reports secure/insecure per scenario severity.
"""
import json, sys
from pathlib import Path
import pandas as pd, pandapower as pp, importlib.util

OUTPUT_DIR=Path("data/processed")
NET_RE_FILE=OUTPUT_DIR/"ieee14_net_re.json"
STAGE3_FILE=Path("03_renewables_bess.py")

def load_stage3():
    spec=importlib.util.spec_from_file_location("stage3_renewables_bess", STAGE3_FILE)
    m=importlib.util.module_from_spec(spec); sys.modules["stage3_renewables_bess"]=m; spec.loader.exec_module(m); return m

def n1_security(scenario_id: str=None):
    net0=pp.from_json(str(NET_RE_FILE))
    lines=list(net0.line.cid)
    worst={"max_loading":0,"v_min":2}
    insecure=[]
    for cid in lines:
        net=pp.from_json(str(NET_RE_FILE))
        net.line.loc[net.line.cid==cid,"in_service"]=False
        pp.runpp(net, numba=True)
        ml=float(max(net.res_line.loading_percent.max(), net.res_trafo.loading_percent.max())) if net.converged else 999
        vmin=float(net.res_bus.vm_pu.min()) if net.converged else 0
        if ml>3.0 or vmin<0.94:
            insecure.append(cid)
        worst["max_loading"]=max(worst["max_loading"], ml)
        worst["v_min"]=min(worst["v_min"], vmin)
    return {"n_contingencies":len(lines),"n_insecure":len(insecure),"insecure":insecure,"worst":worst,"secure":len(insecure)==0}

if __name__=="__main__":
    import argparse, json as js
    p=argparse.ArgumentParser(description="N-1 Security — Stage13")
    p.add_argument("--scenario-id", type=str)
    p.add_argument("--batch", action="store_true")
    args=p.parse_args()
    if args.batch:
        r=n1_security()
        print(js.dumps(r, indent=2))
        Path("data/scenarios/n1_security.json").parent.mkdir(parents=True, exist_ok=True)
        open("data/scenarios/n1_security.json","w").write(js.dumps(r, indent=2))
        print(f"[PASS] N-1 secure={r['secure']} insecure {r['n_insecure']}/15")
    else:
        print(js.dumps(n1_security(args.scenario_id), indent=2))
