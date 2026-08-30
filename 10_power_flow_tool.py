#!/usr/bin/env python3
"""
Stage 10 — Power-Flow Tool (per proposal: structured PF API)
Resolves naming conflict: Stage 10 = Power-Flow Tool, Stage 12 = Contingency.
Provides: run_power_flow(scenario_id) -> {converged, voltages, loadings, losses, violations}
"""
import json, sys
from pathlib import Path
import pandas as pd, pandapower as pp, importlib.util

OUTPUT_DIR = Path("data")/"processed"
NET_RE_FILE = OUTPUT_DIR/"ieee14_net_re.json"
SCENARIOS_CSV = OUTPUT_DIR/"ieee14_scenarios.csv"
SCENARIOS_JSONL = OUTPUT_DIR/"ieee14_scenarios.jsonl"
STAGE3_FILE = Path("03_renewables_bess.py")

def load_stage3():
    spec=importlib.util.spec_from_file_location("stage3_renewables_bess", STAGE3_FILE)
    m=importlib.util.module_from_spec(spec)
    sys.modules["stage3_renewables_bess"]=m
    spec.loader.exec_module(m)
    return m

def snapshot(net):
    MU={"load":("p_mw","q_mvar","in_service"),"line":("in_service",),"trafo":("in_service",),"gen":("p_mw","vm_pu","in_service"),"ext_grid":("vm_pu","in_service"),"sgen":("p_mw","q_mvar","in_service"),"storage":("p_mw","soc_percent","in_service"),"shunt":("q_mvar","p_mw","in_service")}
    return {t:{c:getattr(net,t)[c].values.copy() for c in cs if c in getattr(net,t).columns} for t,cs in MU.items()}
def restore(net,snap):
    for t,cs in snap.items():
        for c,v in cs.items():
            getattr(net,t)[c]=v.copy()

def run_power_flow(scenario_id: str):
    stage3=load_stage3()
    net=pp.from_json(str(NET_RE_FILE))
    snap=snapshot(net)
    base_p=net.load.p_mw.values.copy(); base_q=net.load.q_mvar.values.copy()
    handles={"pv_id":str(net.sgen.cid.iloc[0]),"wind_id":str(net.sgen.cid.iloc[1]),"bess_id":str(net.storage.cid.iloc[0])}
    points=pd.read_csv(OUTPUT_DIR/"ieee14_operating_points.csv")
    factors=pd.read_csv(OUTPUT_DIR/"ieee14_op_load_factors.csv")
    point_index={op_id:i for i,op_id in enumerate(points.op_id)}
    factor_matrix=factors.drop(columns=["op_id"]).values
    # find scenario
    with open(SCENARIOS_JSONL) as f:
        rec=None
        for line in f:
            j=json.loads(line)
            if j["scenario_id"]==scenario_id:
                rec=j; break
    if rec is None:
        raise KeyError(f"{scenario_id} not found")
    row=points.iloc[point_index[rec["pre_event"]["op_id"]]]
    net.load["p_mw"]=base_p*factor_matrix[point_index[rec["pre_event"]["op_id"]]]
    net.load["q_mvar"]=base_q*factor_matrix[point_index[rec["pre_event"]["op_id"]]]
    stage3.set_pv_output(net, handles["pv_id"], float(row.solar_fraction))
    stage3.set_wind_output(net, handles["wind_id"], float(row.wind_fraction))
    stage3.set_bess_power(net, handles["bess_id"], float(row.bess_p_mw))
    stage3.set_bess_soc(net, handles["bess_id"], float(row.bess_soc))
    # apply injected
    def apply(net, ev):
        m=ev["mechanism"]
        if m=="none": return
        if m=="compound":
            for c in ev["components"]: apply(net,c); return
        if m=="load_change":
            f=1+ev["magnitude_percent"]/100
            mask=net.load.cid.isin(ev["targets"]).values
            net.load.loc[mask,"p_mw"]*=f; net.load.loc[mask,"q_mvar"]*=f; return
        if m=="line_outage":
            net.line.loc[net.line.cid.isin(ev["targets"]).values,"in_service"]=False; return
        if m=="generator_outage":
            net.gen.loc[net.gen.cid.isin(ev["targets"]).values,"in_service"]=False; return
        if m=="renewable_ramp":
            d=ev["delta_availability"]; mask=net.sgen.cid.isin(ev["targets"]).values
            rated=net.sgen.loc[mask,"rated_mw"].values.astype(float)
            cur=net.sgen.loc[mask,"p_mw"].values.astype(float)
            avail=np.clip(cur/rated+d,0,1)
            net.sgen.loc[mask,"p_mw"]=rated*avail; net.sgen.loc[mask,"availability"]=avail; return
        if m=="avr_setpoint_shift":
            d=ev["delta_vm_pu"]
            mask=net.gen.cid.isin(ev["targets"]).values
            net.gen.loc[mask,"vm_pu"]+=d
            mask2=net.ext_grid.cid.isin(ev["targets"]).values
            if mask2.any(): net.ext_grid.loc[mask2,"vm_pu"]+=d; return
        if m=="shunt_overcompensation":
            net.shunt["q_mvar"]*=ev["factor"]; return
        if m=="bess_dispatch":
            net.storage["p_mw"]=float(ev["p_mw"]); return
        raise ValueError(m)
    import numpy as np
    apply(net, rec["injected_event"])
    pp.runpp(net, numba=True)
    # build result
    voltages={cid:float(v) for cid,v in zip(net.bus.cid, net.res_bus.vm_pu.values)}
    loadings={cid:float(v) for cid,v in zip(list(net.line.cid)+list(net.trafo.cid), list(net.res_line.loading_percent.values)+list(net.res_trafo.loading_percent.values))}
    violations={
        "undervoltage":[cid for cid,v in voltages.items() if v < float(net.bus.loc[net.bus.cid==cid,"min_vm_pu"].values[0])],
        "overvoltage":[cid for cid,v in voltages.items() if v > float(net.bus.loc[net.bus.cid==cid,"max_vm_pu"].values[0])],
        "overload":[cid for cid,v in loadings.items() if v > float((net.line.loc[net.line.cid==cid,"max_loading_percent"].values[0] if cid in net.line.cid.values else net.trafo.loc[net.trafo.cid==cid,"max_loading_percent"].values[0]))]
    }
    return {"scenario_id":scenario_id,"converged":bool(net.converged),"voltages":voltages,"loadings":loadings,"losses_mw":float(net.res_line.pl_mw.sum()+net.res_trafo.pl_mw.sum()),"violations":violations,"injected":rec["injected_event"],"consequences":rec["consequences"]}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description="Power-Flow Tool — Stage 10")
    p.add_argument("--scenario-id", help="IEEE14_SCN_xxxxxx")
    p.add_argument("--list", action="store_true", help="list first 5 scenario ids")
    p.add_argument("--batch", action="store_true", help="run all 3000 and write CSV (compatibility)")
    args=p.parse_args()
    if args.list:
        df=pd.read_csv(SCENARIOS_CSV)
        print(df.scenario_id.head().to_string())
        sys.exit(0)
    if args.scenario_id:
        import json as js
        print(js.dumps(run_power_flow(args.scenario_id), indent=2))
        sys.exit(0)
    if args.batch:
        # for compatibility with old Stage10 batch expectation
        print("Batch mode: use 12_contingency_tool.py --batch for N-1; for PF batch use --scenario-id loop")
        sys.exit(0)
    p.print_help()
