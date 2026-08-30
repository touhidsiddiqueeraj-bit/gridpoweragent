#!/usr/bin/env python3
"""
Stage 11 — Grid/Component Query Tool
Provides topology, limits, equipment, BESS, renewable queries for RAG.
"""
import json, sys
from pathlib import Path
import pandas as pd, pandapower as pp, importlib.util

OUTPUT_DIR=Path("data/processed")
NET_RE_FILE=OUTPUT_DIR/"ieee14_net_re.json"
STAGE3_FILE=Path("03_renewables_bess.py")

def load_net():
    return pp.from_json(str(NET_RE_FILE))

def query_topology():
    net=load_net()
    return {"buses": net.bus.cid.tolist(), "lines": [{"cid":r.cid,"from":r.bus_ieee if hasattr(r,'bus_ieee') else None,"to":None,"max_loading":float(r.max_loading_percent)} for _,r in net.line.iterrows()], "trafos": net.trafo.cid.tolist(), "gens": net.gen.cid.tolist()}

def query_limits():
    net=load_net()
    return {"voltage":{"min":float(net.bus.min_vm_pu.min()),"max":float(net.bus.max_vm_pu.max())},"thermal_line_max":float(net.line.max_loading_percent.max()),"thermal_trafo_max":float(net.trafo.max_loading_percent.max())}

def query_equipment(component_id: str):
    net=load_net()
    for tbl in ["bus","line","trafo","gen","load","sgen","storage","shunt"]:
        df=getattr(net,tbl)
        if "cid" in df.columns and component_id in df.cid.values:
            row=df[df.cid==component_id].iloc[0]
            return {"table":tbl,"cid":component_id,"data": row.to_dict()}
    return {"error": f"{component_id} not found"}

def query_bess():
    net=load_net()
    r=net.storage.iloc[0]
    return {"cid":r.cid,"bus":int(r.bus),"p_mw":float(r.p_mw),"soc_percent":float(r.soc_percent),"p_max":20.0,"e_max":40.0}

def query_renewable():
    net=load_net()
    return [{"cid":r.cid,"bus":int(r.bus),"p_mw":float(r.p_mw),"rated":float(r.rated_mw),"availability":float(r.availability)} for _,r in net.sgen.iterrows()]

if __name__=="__main__":
    import argparse, json as js
    p=argparse.ArgumentParser(description="Grid Query Tool — Stage 11")
    p.add_argument("--topology", action="store_true")
    p.add_argument("--limits", action="store_true")
    p.add_argument("--equipment", type=str)
    p.add_argument("--bess", action="store_true")
    p.add_argument("--renewable", action="store_true")
    args=p.parse_args()
    if args.topology: print(js.dumps(query_topology(), indent=2))
    elif args.limits: print(js.dumps(query_limits(), indent=2))
    elif args.equipment: print(js.dumps(query_equipment(args.equipment), indent=2, default=str))
    elif args.bess: print(js.dumps(query_bess(), indent=2))
    elif args.renewable: print(js.dumps(query_renewable(), indent=2))
    else: p.print_help()
