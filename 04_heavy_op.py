#!/usr/bin/env python3
import sys
from pathlib import Path
import pandapower as pp, numpy as np, pandas as pd, json, time, hashlib

case=sys.argv[1] if len(sys.argv)>1 else "case39"
N=int(sys.argv[2]) if len(sys.argv)>2 else (5000 if case=="case39" else 7000)
print(f"Heavy OP gen {case} N={N}")
net=pp.from_json(f"data/processed/{case}_net_re.json")
base_p=net.load.p_mw.values.copy()
base_q=net.load.q_mvar.values.copy()
pv_id=str(net.sgen.cid.iloc[0]); wind_id=str(net.sgen.cid.iloc[1]); bess_id=str(net.storage.cid.iloc[0])

# For RE setting, need stage logic but we can inline
def set_pv(net, cid, av):
    mask=net.sgen.cid==cid
    rated=net.sgen.loc[mask,"rated_mw"].values[0]
    net.sgen.loc[mask,"p_mw"]=rated*np.clip(av,0,1)
    net.sgen.loc[mask,"availability"]=np.clip(av,0,1)
def set_bess_p(net, cid, p): net.storage.loc[net.storage.cid==cid,"p_mw"]=float(p)
def set_bess_soc(net, cid, soc): net.storage.loc[net.storage.cid==cid,"soc_percent"]=float(np.clip(soc,0,1)*100)

bus_ids=list(net.bus.cid)
branch_ids=list(net.line.cid)+list(net.trafo.cid)
load_ids=list(net.load.cid)

rng=np.random.default_rng(20260821)
records=[]; factor_rows=[]; voltage_rows=[]; loading_rows=[]
tries=0; start=time.time()
# snapshot for restore via base+RE only (in_service not touched)
while len(records)<N and tries< N*5:
    tries+=1
    if case=="case39":
        load_scale=float(rng.uniform(0.88,1.08))
    elif case=="case118":
        load_scale=float(rng.uniform(0.75,1.10))
    else:
        load_scale=float(rng.uniform(0.85,1.15))  # narrower for heavy to keep normals
    factors=load_scale*(1+rng.uniform(-0.05,0.05,size=len(load_ids)))
    factors=np.clip(factors,0.5,1.5)
    solar=float(rng.uniform(0,1)); wind=float(rng.uniform(0,1)); soc=float(rng.uniform(0.15,0.85)); bess_p=float(rng.choice([0,0,rng.uniform(-10,10)]))
    net.load["p_mw"]=base_p*factors
    net.load["q_mvar"]=base_q*factors
    set_pv(net,pv_id,solar); set_pv(net,wind_id,wind)
    set_bess_p(net,bess_id,bess_p); set_bess_soc(net,bess_id,soc)
    try: pp.runpp(net, numba=True)
    except: continue
    if not net.converged: continue
    # check violations
    vmin=net.res_bus.vm_pu.min(); vmax=net.res_bus.vm_pu.max()
    max_line=net.res_line.loading_percent.max() if len(net.res_line) else 0
    max_trafo=net.res_trafo.loading_percent.max() if len(net.res_trafo) else 0
    peak=max(max_line,max_trafo)
    under=(net.res_bus.vm_pu.values < net.bus.min_vm_pu.values).any()
    over=(net.res_bus.vm_pu.values > net.bus.max_vm_pu.values).any()
    overload=((net.res_line.loading_percent.values > net.line.max_loading_percent.values).any() or (net.res_trafo.loading_percent.values > net.trafo.max_loading_percent.values).any())
    if under or over or overload: continue
    op_id=f"{case.upper()}_OP_{len(records)+1:06d}"
    # compute totals
    total_load=float(net.load.p_mw.sum()); renewable=float(net.sgen.p_mw.sum()); slack=float(net.res_ext_grid.p_mw.sum())
    losses=float(net.res_line.pl_mw.sum()+net.res_trafo.pl_mw.sum())
    rec={"op_id":op_id,"load_scale":load_scale,"solar_fraction":solar,"wind_fraction":wind,"bess_soc":soc,"bess_p_mw":bess_p,
         "v_min_pu":float(vmin),"v_max_pu":float(vmax),"v_min_bus":str(net.bus.cid.values[int(net.res_bus.vm_pu.values.argmin())]),
         "v_max_bus":str(net.bus.cid.values[int(net.res_bus.vm_pu.values.argmax())]),
         "max_line_loading_percent":float(max_line),"max_trafo_loading_percent":float(max_trafo),"peak_branch_loading_percent":float(peak),
         "total_load_mw":total_load,"renewable_p_mw":renewable,"slack_p_mw":slack,"slack_q_mvar":float(net.res_ext_grid.q_mvar.sum()),"total_losses_mw":losses}
    records.append(rec); factor_rows.append(factors); voltage_rows.append(net.res_bus.vm_pu.values.copy()); loading_rows.append(np.concatenate([net.res_line.loading_percent.values, net.res_trafo.loading_percent.values]))
    if len(records)%500==0: print(f"  {len(records)}/{N} after {tries} tries ({time.time()-start:.1f}s)")

df=pd.DataFrame(records)
outdir=Path(f"data/processed")
df.to_csv(outdir/f"{case}_operating_points.csv", index=False)
# handle empty case gracefully
if len(df)>0:
    pd.DataFrame(np.array(factor_rows), columns=load_ids).assign(op_id=df.op_id).to_csv(outdir/f"{case}_op_load_factors.csv", index=False)
    pd.DataFrame(np.array(voltage_rows), columns=bus_ids).assign(op_id=df.op_id).to_csv(outdir/f"{case}_op_bus_voltages.csv", index=False)
    pd.DataFrame(np.array(loading_rows), columns=branch_ids).assign(op_id=df.op_id).to_csv(outdir/f"{case}_op_branch_loading.csv", index=False)
else:
    # create empty headers so resume can detect 0
    pd.DataFrame(columns=["op_id"]+load_ids).to_csv(outdir/f"{case}_op_load_factors.csv", index=False)
    pd.DataFrame(columns=["op_id"]+bus_ids).to_csv(outdir/f"{case}_op_bus_voltages.csv", index=False)
    pd.DataFrame(columns=["op_id"]+branch_ids).to_csv(outdir/f"{case}_op_branch_loading.csv", index=False)
print(f"Saved {len(df)} points for {case} in {time.time()-start:.1f}s")
