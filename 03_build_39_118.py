#!/usr/bin/env python3
"""
Build IEEE 39 and 118 RE networks (heavy non-LLM)
"""
from pathlib import Path
import pandapower as pp, hashlib, importlib.util, sys

def ensure_cids(net, pv_cid, wind_cid, bess_cid):
    # Bus
    if "cid" not in net.bus.columns: net.bus["cid"]=[f"bus_{i+1}" for i in net.bus.index]
    if "bus_ieee" not in net.bus.columns: net.bus["bus_ieee"]=[i+1 for i in net.bus.index]
    is_large = len(net.bus) > 30
    net.bus["min_vm_pu"]=0.94; net.bus["max_vm_pu"]=1.10 if is_large else 1.05
    net.bus.loc[net.ext_grid.bus.values[0],"max_vm_pu"]=1.06 if not is_large else 1.10
    # Load
    if "cid" not in net.load.columns: net.load["cid"]=[f"load_bus_{net.bus.loc[b,'cid'].split('_')[1]}" for b in net.load.bus]
    if "bus_ieee" not in net.load.columns: net.load["bus_ieee"]=[int(net.bus.loc[b,"cid"].split("_")[1]) for b in net.load.bus]
    # Line
    if "cid" not in net.line.columns: net.line["cid"]=[f"line_{a+1}_{b+1}" for a,b in zip(net.line.from_bus, net.line.to_bus)]
    if "cid" not in net.trafo.columns: net.trafo["cid"]=[f"trafo_{a+1}_{b+1}" for a,b in zip(net.trafo.hv_bus, net.trafo.lv_bus)]
    # Gen
    if "cid" not in net.gen.columns: net.gen["cid"]=[f"gen_{i}" for i in net.gen.index]
    # sgen/storage/shunt cids handled by caller
    # Limits: 14 tuned 3% (base 1.45% -> overload at 4%), 39 realistic 100% (base 75% -> overload), 118 tuned 6% (base 4.5% -> overload)
    n=len(net.bus)
    if n <= 20:
        net.line["max_loading_percent"]=3.0; net.trafo["max_loading_percent"]=4.0
    elif n == 39:
        net.line["max_loading_percent"]=100.0; net.trafo["max_loading_percent"]=100.0
    else: # 118
        net.line["max_loading_percent"]=6.0; net.trafo["max_loading_percent"]=6.0
    # gen vm clamp
    for idx in net.gen.index: net.gen.at[idx,"vm_pu"]=1.02 + (idx%3)*0.01
    net.ext_grid["vm_pu"]=1.06

def build_case(case_fn, pv_bus_ieee, wind_bus_ieee, bess_bus_ieee, pv_rated, wind_rated):
    net=getattr(pp.networks, case_fn)()
    pv_idx=pv_bus_ieee-1; wind_idx=wind_bus_ieee-1; bess_idx=bess_bus_ieee-1
    # create sgens
    pp.create_sgen(net, bus=pv_idx, p_mw=pv_rated*0.5, q_mvar=0, name="sgen_PV")
    net.sgen.at[net.sgen.index[-1],"cid"]="sgen_PV"
    net.sgen.at[net.sgen.index[-1],"rated_mw"]=pv_rated; net.sgen.at[net.sgen.index[-1],"availability"]=0.5
    pp.create_sgen(net, bus=wind_idx, p_mw=wind_rated*0.5, q_mvar=0, name="sgen_WIND")
    net.sgen.at[net.sgen.index[-1],"cid"]="sgen_WIND"
    net.sgen.at[net.sgen.index[-1],"rated_mw"]=wind_rated; net.sgen.at[net.sgen.index[-1],"availability"]=0.5
    pp.create_storage(net, bus=bess_idx, p_mw=0, max_e_mwh=40, soc_percent=50, min_e_mwh=4, max_p_mw=20, min_p_mw=-20, name="gen_BESS")
    net.storage.at[net.storage.index[-1],"cid"]="gen_BESS"
    if len(net.shunt)==0:
        pp.create_shunt(net, bus=bess_idx, q_mvar=19, p_mw=0)
        net.shunt.at[net.shunt.index[-1],"cid"]="shunt_bus"
    else:
        net.shunt.at[net.shunt.index[0],"cid"]="shunt_bus"
        net.shunt.at[net.shunt.index[0],"q_mvar"]=19
    ensure_cids(net, "sgen_PV","sgen_WIND","gen_BESS")
    return net

def hash_net(net):
    parts=[]
    for _,r in net.sgen.sort_values("cid").iterrows(): parts.append(f"{r.cid}:{r.rated_mw}:{r.bus}")
    for _,r in net.storage.sort_values("cid").iterrows(): parts.append(f"{r.cid}:{r.max_e_mwh}:{r.bus}")
    raw="|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()

if __name__=="__main__":
    for case, pv, wind, bess, pr, wr in [("case39",30,37,15,30,40),("case118",80,30,60,50,60)]:
        net=build_case(case,pv,wind,bess,pr,wr)
        pp.runpp(net, numba=True)
        print(case, "converged", net.converged, "vmin", net.res_bus.vm_pu.min(), "vmax", net.res_bus.vm_pu.max(), "peak", max(net.res_line.loading_percent.max(), net.res_trafo.loading_percent.max() if len(net.res_trafo) else 0))
        out=Path(f"data/processed/{case}_net_re.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        pp.to_json(net, str(out))
        h=hash_net(net)
        Path(f"data/processed/{case}_re_layout_hash.txt").write_text(h+"\n")
        print(f"  saved {out} hash {h[:12]}")
