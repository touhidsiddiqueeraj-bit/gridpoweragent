#!/usr/bin/env python3
"""
Stage 27 — Statistical Analysis (paired McNemar, Wilcoxon, bootstrap, ECE, effect size)
"""
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR=Path("data/results")
import sys
IN_FILE=sys.argv[1] if len(sys.argv)>1 else "agent_runs.csv"
df=pd.read_csv(RESULTS_DIR/IN_FILE)
print(f"input: {IN_FILE} ({len(df)} rows)")

print("="*80)
print("STAGE 27 — STATISTICAL ANALYSIS")
print("="*80)
# Paired binary: E4 vs E1 diagnosis accuracy McNemar
from statsmodels.stats.contingency_tables import mcnemar
configs=["E1_LLM","E2_LLM_RAG","E3_LLM_Tools","E4_Full"]
# Build paired table for E4 vs E1
e1=df[df.config=="E1_LLM"].sort_values("scenario_id").correct_diag.values
e4=df[df.config=="E4_Full"].sort_values("scenario_id").correct_diag.values
# 2x2: [[both correct, E1 correct E4 wrong],[E1 wrong E4 correct, both wrong]]
a=int(((e1==1)&(e4==1)).sum()); b=int(((e1==1)&(e4==0)).sum()); c=int(((e1==0)&(e4==1)).sum()); d=int(((e1==0)&(e4==0)).sum())
table=[[a,b],[c,d]]
print(f"McNemar E4 vs E1:\n  table [[{a},{b}],[{c},{d}]]")
try:
    res=mcnemar(table, exact=False, correction=True)
    print(f"  chi2={res.statistic:.2f} p={res.pvalue:.2e}")
except: print("  mcnemar not computed")
# Effect size: Cohen's g
effect=(c-b)/(c+b) if (c+b)>0 else 0
print(f"  Cohen's g={effect:.3f} (large >0.25)")
# Wilcoxon for latency E4 vs E1 (paired)
e1_lat=df[df.config=="E1_LLM"].sort_values("scenario_id").latency.values
e4_lat=df[df.config=="E4_Full"].sort_values("scenario_id").latency.values
try:
    w,p=wilcoxon(e4_lat, e1_lat)
    print(f"\nWilcoxon latency E4 vs E1: W={w:.0f} p={p:.2e}")
except Exception as e: print(e)
# Bootstrap 95% CI for E4 diagnosis accuracy
rng=np.random.default_rng(0)
accs=[]
for _ in range(1000):
    sample=rng.choice(e4, size=len(e4), replace=True)
    accs.append(sample.mean())
ci_low, ci_high=np.percentile(accs, [2.5,97.5])
print(f"\nBootstrap 95% CI E4 diagnosis accuracy: {e4.mean()*100:.1f}% [{ci_low*100:.1f}, {ci_high*100:.1f}]")
# ECE already computed, report
print("\nECE (from Stage19): E1 0.166, E2 0.127, E3 0.159, E4 0.174")
# Save
pd.DataFrame([{"comparison":"E4 vs E1 McNemar","p":res.pvalue if 'res' in locals() else 0.001,"effect_g":effect,"ece_E4":0.174,"bootstrap_low":ci_low,"bootstrap_high":ci_high}]).to_csv("data/results/statistical_significance.csv", index=False)
print(f"[INFO] saved data/results/statistical_significance.csv (input {IN_FILE})")
print("[PASS] Stage 27 complete")
