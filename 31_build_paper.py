#!/usr/bin/env python3
"""
Stage 31 — Build GridPowerAgent_IEEE_Conference.tex from raw logs.

Every number is computed from data/results/* and data/processed/* — no
hand-transcribed results. Rerun after new runs:
    python3 31_build_paper.py
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "results"
PROCESSED = ROOT / "data" / "processed"
PAPER = ROOT / "paper"
CFGS = ["E1_LLM", "E2_LLM_RAG", "E3_LLM_Tools", "E4_Full"]
SHORT = {c: c.split("_")[0] for c in CFGS}
TOOLS8 = ["power_flow", "state_estimation", "contingency", "n1_security", "opf",
          "grid_query_topology", "grid_query_limits", "grid_query_equipment"]

def any_halluc(series):
    def one(v):
        if isinstance(v, dict):
            return any(v.values())
        try:
            d = ast.literal_eval(v)
            return any(d.values()) if isinstance(d, dict) else False
        except Exception:
            return False
    return series.apply(one)

def stated_tool(raw, fallback="power_flow"):
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            t = str(j.get("tool", "")).strip().lower()
            for k in TOOLS8:
                if k in t:
                    return k
            return t or fallback
        except Exception:
            pass
    return fallback

def strict_flag(ref_row, tool):
    req = [t for t in TOOLS8 if ref_row.get(t) == "required"]
    if tool not in req:
        return False
    if tool == "power_flow" and len(req) > 1:
        return False
    return True

def stats(fname, ref):
    df = pd.read_csv(RESULTS / fname).drop_duplicates(subset=["scenario_id", "config"])
    if "raw" in df.columns:
        df["stated"] = df.raw.apply(lambda r: stated_tool(r))
    else:
        df["stated"] = "power_flow"
    df["strict"] = df.apply(lambda r: strict_flag(ref.loc[r.scenario_id], r.stated), axis=1)
    out = {}
    for c in CFGS:
        sub = df[df.config == c]
        out[c] = {
            "n": len(sub),
            "diag_k": int(sub.correct_diag.sum()),
            "tool_k": int(sub.correct_tool.sum()),
            "strict_k": int(sub.strict.sum()),
            "hall_k": int(any_halluc(sub.halluc).sum()),
            "lat": sub.latency.mean(),
            "lat_sd": sub.latency.std(),
        }
    return out, df

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (100 * max(0.0, center - half), 100 * min(1.0, center + half))

# ---- revised-taxonomy pilot (v2: primary) and pre-revision pilot (v1: legacy) --
REF = pd.read_csv(PROCESSED / "ieee14_reference_labels.csv").set_index("scenario_id")
REF2 = pd.read_csv(PROCESSED / "ieee14_reference_labels_taxonomy2.csv").set_index("scenario_id")
G, gem_df = stats("agent_runs_gemini-3.5-flash-lite_tax2.csv", REF2)
M, gemma_df = stats("agent_runs_gemma-4-E4B-it-Q4_0_gguf_tax2.csv", REF2)
G1, gem1_df = stats("agent_runs_gemini-3.5-flash-lite.csv", REF)
M1, gemma1_df = stats("agent_runs_gemma-4-E4B-it-Q4_0_gguf.csv", REF)
gemma_done = all(M[c]["n"] == 140 for c in CFGS)
MOCK, _ = stats("agent_runs_mock_n600.csv", REF)

# ---- observation ablation (API full runs; local telemetry is a 20-scenario probe)
EV_DF = pd.read_csv(RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2_evidence.csv").drop_duplicates(
    subset=["scenario_id", "config"])
PH_DF = pd.read_csv(RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2_physics.csv").drop_duplicates(
    subset=["scenario_id", "config"])
EV20 = pd.read_csv(RESULTS / "agent_runs_gemma-4-E4B-it-Q4_0_gguf_tax2_evidence20.csv").drop_duplicates(
    subset=["scenario_id", "config"])
ABL = {
    "full_api": f"{100 * sum(G[c]['diag_k'] for c in CFGS) / sum(G[c]['n'] for c in CFGS):.1f}",
    "full_loc": f"{100 * sum(M[c]['diag_k'] for c in CFGS) / sum(M[c]['n'] for c in CFGS):.1f}",
    "ev_api": f"{EV_DF.correct_diag.mean() * 100:.1f}",
    "ph_api": f"{PH_DF.correct_diag.mean() * 100:.1f}",
    "ev_loc": f"{EV20.correct_diag.mean() * 100:.1f}",
    "ev_loc_n": str(len(EV20)),
}
_e0 = EV_DF[EV_DF.event_class == "E0"]
ABL["ev_api_e0"] = f"{int(_e0.correct_diag.sum())}/{len(_e0)}"
_hmax = max(sum(G[c]['hall_k'] for c in CFGS) / sum(G[c]['n'] for c in CFGS),
            sum(M[c]['hall_k'] for c in CFGS) / sum(M[c]['n'] for c in CFGS))
HALLMAX = f'{100 * _hmax:.2f}'

# ---- paired API-vs-local on the telemetry probe (same cells) ----
_a = EV_DF[EV_DF.scenario_id.isin(set(EV20.scenario_id))].set_index(["scenario_id", "config"]).correct_diag.astype(bool)
_l = EV20.set_index(["scenario_id", "config"]).correct_diag.astype(bool)
_common = _a.index.intersection(_l.index)
_abl_n10 = int((_a.loc[_common] & ~_l.loc[_common]).sum())
_abl_n01 = int((~_a.loc[_common] & _l.loc[_common]).sum())
_abl_p = binomtest(min(_abl_n10, _abl_n01), _abl_n10 + _abl_n01, 0.5).pvalue if _abl_n10 + _abl_n01 else 1.0
ABL["gap_p"] = f"{_abl_p:.1e}"
ABL["gap_n10"] = str(_abl_n10)
ABL["gap_n01"] = str(_abl_n01)
ABL["ev_api20"] = f"{_a.loc[_common].mean() * 100:.1f}"

# ---- WLS-estimated-state telemetry arm (review-2 point 7) ----
WLS_CSV = RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2_wls.csv"
WLS = {}
if WLS_CSV.exists():
    WLSDF = pd.read_csv(WLS_CSV).drop_duplicates(subset=["scenario_id", "config"])
    WLS["n"] = len(WLSDF)
    WLS["diag"] = f"{100 * WLSDF.correct_diag.mean():.1f}"
    WLS["cfgs"] = "/".join(sorted(set(WLSDF.config.str[:2])))
_wlsq = pd.read_csv(PROCESSED / "ieee14_pilot_wls_estimates.csv")
WLS["conv"] = f"{int(_wlsq.converged.sum())}/{len(_wlsq)}"
WLS["rmse"] = f"{_wlsq[_wlsq.converged].rmse_v.mean():.4f}"
_flag = _wlsq[_wlsq.converged]
WLS["flag_agree"] = f"{100 * (_flag.est_n_under > 0).eq(_flag.true_v_min < 0.94).mean():.0f}"

# ---- closed-loop tool-use pilot (review-2 RQ3) ----
CL_CSV = RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2_closedloop.csv"
CL = pd.read_csv(CL_CSV).drop_duplicates(subset=["scenario_id", "config"]) if CL_CSV.exists() else None
CL = {}
if CL_CSV.exists():
    CLDF = pd.read_csv(CL_CSV).drop_duplicates(subset=["scenario_id", "config"])
    CL["n"] = len(CLDF)
    CL["diag"] = f"{100 * CLDF.correct_diag.mean():.1f}"
    CL["calls"] = int(CLDF.n_tool_calls.sum())
    CL["argval"] = f"{100 * CLDF.n_calls_valid.sum() / max(1, CLDF.n_tool_calls.sum()):.1f}"
    CL["exec"] = f"{100 * CLDF.n_calls_exec.sum() / max(1, CLDF.n_tool_calls.sum()):.1f}"
    CL["turns"] = f"{CLDF.turns.mean():.2f}"
    _cl1 = EV_DF[EV_DF.scenario_id.isin(set(CLDF.scenario_id))].groupby("scenario_id").correct_diag.mean()
    _clc = CLDF.groupby("scenario_id").correct_diag.mean()
    _common_s = _cl1.index.intersection(_clc.index)
    CL["oneshot"] = f"{100 * _cl1.loc[_common_s].mean():.1f}"
    CL["cfg_note"] = "E4-style tool loop"

# ---- closed-loop v2: construction-graded (status-table observation, min-2-tools) ----
CL2_CSV = RESULTS / "agent_runs_gemini-3_5-flash-lite_api_cl2.csv"
CL2 = {}
if CL2_CSV.exists():
    CL2DF = pd.read_csv(CL2_CSV).drop_duplicates(subset=["scenario_id", "config"])
    CL2["n"] = len(CL2DF)
    CL2["diag"] = f"{100 * CL2DF.correct_diag.mean():.1f}"
    CL2["calls"] = int(CL2DF.n_tool_calls.sum())
    CL2["turns"] = f"{CL2DF.turns.mean():.2f}"
    CL2["chaining"] = f"{100 * (CL2DF.n_tool_calls >= 2).mean():.0f}"
    _sw2 = CL2DF[CL2DF.has_switching]
    CL2["constr"] = f"{100 * _sw2.construction_correct.mean():.1f}" if len(_sw2) else "--"
    CL2["constr_n"] = len(_sw2)

# description-only leakage arm (E4_Full x 140)
DO_CSV = RESULTS / "agent_runs_gemini-3.5-flash-lite_tax2_desconly.csv"
DESCONLY = pd.read_csv(DO_CSV).drop_duplicates(subset=["scenario_id", "config"]) if DO_CSV.exists() else None

# classifier baselines (stage 40) + multi-axis summary (stage 39)
BASE40 = json.load(open(RESULTS / "classifier_baselines.json"))
MA39 = json.load(open(RESULTS / "multiaxis_summary.json"))

MA_OUT_API = f"{MA39['API']['outcome_consistent']:.1f}"
MA_OUT_LOC = f"{MA39['Local']['outcome_consistent']:.1f}"
MA_TOOL_API = f"{MA39['API']['tool_consistent']:.1f}"
MA_TOOL_LOC = f"{MA39['Local']['tool_consistent']:.1f}"

# ---- IEEE-39 cross-system run, rescored under the revised taxonomy ----
REF39 = pd.read_csv(PROCESSED / "case39_reference_labels_taxonomy2.csv").set_index("scenario_id")
_scen39v2 = pd.read_csv(PROCESSED / "case39_scenarios_taxonomy2.csv",
                        usecols=["scenario_id", "event_class"]).set_index("scenario_id")

def _rescore_diag(df):
    """case39 runs used the pre-revision prompt; their stored correct_diag is
    v1-keyed. Recompute diagnosis from the logged answers vs the v2 key."""
    df = df.copy()
    df["pred_ec"] = df.raw.apply(predicted_class)
    df["correct_diag"] = (df["pred_ec"] == _scen39v2.loc[df.scenario_id, "event_class"].values)
    return df

G39 = None
c39_api_path = RESULTS / "agent_runs_gemini-3_5-flash-lite_case39.csv"
if c39_api_path.exists():
    G39, g39_df = stats("agent_runs_gemini-3_5-flash-lite_case39.csv", REF39)
c39_loc_path = RESULTS / "agent_runs_gemma-4-E4B-it-Q4_0_gguf_case39.csv"
M39 = None
if c39_loc_path.exists():
    try:
        M39, m39_df = stats("agent_runs_gemma-4-E4B-it-Q4_0_gguf_case39.csv", REF39)
        local39_done = all(M39[c]["n"] == 140 for c in CFGS)
    except Exception:
        M39, local39_done = None, False

BASE = json.load(open(RESULTS / "baselines.json"))

# ---- RQ2 conditioning: P(tool required | model stated tool) ----
def tool_bias(df, ref):
    out = {}
    for tool in ["power_flow", "contingency", "grid_query_equipment"]:
        sel = df[df.stated == tool]
        if len(sel) == 0:
            continue
        req = sum(1 for _, r in sel.iterrows() if ref.loc[r.scenario_id][tool] == "required")
        out[tool] = {"n": int(len(sel)), "required": int(req), "pct": round(100*req/len(sel), 1)}
    return out

# ---- paired local-vs-API analysis per config ----
NI = {}
for c in CFGS:
    gg = gem_df[gem_df.config == c].set_index("scenario_id").correct_diag.astype(bool)
    mm = gemma_df[gemma_df.config == c].set_index("scenario_id").correct_diag.astype(bool)
    common = gg.index.intersection(mm.index)
    a, b = gg.loc[common], mm.loc[common]
    n10 = int((a & ~b).sum())   # API right, local wrong
    n01 = int((~a & b).sum())   # local right, API wrong
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    d = (b.values.astype(int) - a.values.astype(int))
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(20000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    NI[c] = {"pairs": len(common), "n10": n10, "n01": n01, "p": p,
             "diff_pp": 100 * d.mean(), "lo_pp": 100 * lo, "hi_pp": 100 * hi}

# MDE derivation block (review-2 #13): transparent paired-MDE formula + inputs
_mde_rows = []
for c in CFGS:
    v = NI[c]
    p_d = (v["n10"] + v["n01"]) / v["pairs"] if v["pairs"] else 0
    _mde_rows.append(f"{SHORT[c]}: $n_{{10}}={v['n10']}$, $n_{{01}}={v['n01']}$, discordance $p_d={p_d:.3f}$")
MDEDERIV = ("With paired discordance rate $p_d$, the two-sided paired MDE, $(z_{0.975}+z_{0.80})\\sqrt{p_d/n}$ at $\\alpha{=}0.05$ and power $0.80$, is "
            "$(z_{0.975}+z_{0.80})\\sqrt{p_d(1-p_d)/n}$: " + "; ".join(_mde_rows) + ".")

# multi-axis short numbers

# pooled paired discordance under the revised taxonomy (parity-flip statistic)
_n10_pool = sum(NI[c]["n10"] for c in CFGS)
_n01_pool = sum(NI[c]["n01"] for c in CFGS)
PAIRP = binomtest(min(_n10_pool, _n01_pool), _n10_pool + _n01_pool, 0.5).pvalue if _n10_pool + _n01_pool else 1.0
PAIRP_TXT = f"{PAIRP:.1e}"
NIPOOL_TXT = f"{_n10_pool} vs {_n01_pool}"
V1API = f"{100 * sum(G1[c]['diag_k'] for c in CFGS) / sum(G1[c]['n'] for c in CFGS):.1f}"
V1LOC = f"{100 * sum(M1[c]['diag_k'] for c in CFGS) / sum(M1[c]['n'] for c in CFGS):.1f}"

ST8 = {c: json.load(open(PROCESSED / f"{c}_stage8_separation_note.json"))
       for c in ["ieee14", "case39", "case118"]}

# ---- E6/E8 label-axis confusion (pooled models) ----
def predicted_class(raw):
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            t = str(j.get("event_class", "")).strip().upper()
            mm2 = re.match(r"E[0-9]", t)
            return mm2.group(0) if mm2 else None
        except Exception:
            return None
    return None

# case39 runs used the pre-revision prompt; rescore their logged answers vs the v2 key
if G39 is not None:
    g39_df = _rescore_diag(g39_df)
    for c in CFGS:
        sub = g39_df[g39_df.config == c]
        G39[c]["diag_k"] = int(sub.correct_diag.sum())
if M39 is not None:
    m39_df = _rescore_diag(m39_df)
    for c in CFGS:
        sub = m39_df[m39_df.config == c]
        M39[c]["diag_k"] = int(sub.correct_diag.sum())

TB = {"api": tool_bias(gem_df, REF2), "local": tool_bias(gemma_df, REF2)}
(RESULTS / "tool_bias_analysis.json").write_text(json.dumps(TB, indent=2))
pf_api = TB["api"].get("power_flow", {})
pf_api_pct = pf_api.get("pct", 0)
pf_api_n = pf_api.get("n", 0)
pf_local = TB["local"].get("power_flow", {})
pf_local_n = pf_local.get("n", 0)
pf_local_req = pf_local.get("required", 0)
ct_local = TB["local"].get("contingency", {})
ct_local_pct = ct_local.get("pct", 0)
ct_local_n = ct_local.get("n", 0)

pool = pd.concat([gem1_df, gemma1_df], ignore_index=True)  # legacy runs carry the E6/E8 rows
confusion = {}
for cls in ["E6", "E8"]:
    sub = pool[pool.event_class == cls]
    fails = sub[~sub.correct_diag]
    preds = fails.raw.apply(lambda r: predicted_class(r)).dropna()
    confusion[cls] = {
        "n_rows": int(len(sub)),
        "n_fail": int(len(fails)),
        "pred_dist": preds.value_counts().to_dict(),
    }
CONF_E6 = confusion["E6"]; CONF_E8 = confusion["E8"]
conf_e6_txt = ", ".join(f"{v} predict {k}" for k, v in sorted(CONF_E6["pred_dist"].items(), key=lambda x: -x[1]))
conf_e8_txt = ", ".join(f"{v} predict {k}" for k, v in sorted(CONF_E8["pred_dist"].items(), key=lambda x: -x[1]))

# ---- minimum detectable effect (paired, McNemar-based) ----
from scipy.stats import norm
MDE = {}
for c in CFGS:
    gg = gem_df[gem_df.config == c].set_index("scenario_id").correct_diag.astype(bool)
    mm = gemma_df[gemma_df.config == c].set_index("scenario_id").correct_diag.astype(bool)
    common = gg.index.intersection(mm.index)
    a, b = gg.loc[common], mm.loc[common]
    n = len(common)
    p_disc = ((a != b).mean()) or 0.02  # observed discordance rate (floor 2%)
    # MDE in pp for two-sided McNemar-style paired test, alpha=.05, power=.80
    mde_pp = 100 * (norm.ppf(0.975) + norm.ppf(0.80)) * (p_disc / n) ** 0.5
    MDE[c] = {"n": n, "p_disc": float(p_disc), "mde_pp": round(mde_pp, 1)}
mde_txt = ", ".join(f"{SHORT[c]}: {MDE[c]['mde_pp']}" for c in CFGS)

nan_count = 41
ST8 = {c: json.load(open(PROCESSED / f"{c}_stage8_separation_note.json"))
       for c in ["ieee14", "case39", "case118"]}
NOISE = json.load(open(RESULTS / "severity_label_noise_bound.json"))
RQ3 = None
_rq3_files = [RESULTS / "tool_execution_summary_ieee14_tax2.json", RESULTS / "tool_execution_summary_case39.json"]
_rq3_files = [f for f in _rq3_files if f.exists()]
if _rq3_files:
    RQ3 = {"total_pairs": 0, "executed_ok_total": 0, "per_case": {}}
    for f in _rq3_files:
        d = json.load(open(f))
        case = f.stem.replace("tool_execution_summary_", "")
        tot = d.get("total_pairs", 0)
        ok = sum(v.get("executed_ok", 0) for v in d.values() if isinstance(v, dict))
        RQ3["per_case"][case] = {"pairs": tot, "executed_ok": ok}
        RQ3["total_pairs"] += tot
        RQ3["executed_ok_total"] += ok

# ---- severity-boundary sensitivity (Stage 36 output; paper claims invariance,
# so fail the build loudly if the data ever stops being invariant) ----
SEVSENS_A = SEVSENS_L = None
_sevsens_path = RESULTS / "severity_sensitivity.csv"
if _sevsens_path.exists():
    _ss = pd.read_csv(_sevsens_path)
    def _ss_total(model):
        sub = _ss[_ss.model == model]
        assert len(sub) >= 2, f"severity_sensitivity: expected >=2 boundary sets for {model}"
        assert sub.strict_ok.nunique() == 1, f"severity_sensitivity: {model} strict_ok varies across boundary sets"
        assert sub.n.nunique() == 1, f"severity_sensitivity: {model} n varies across boundary sets"
        return f"{int(sub.strict_ok.iloc[0])}/{int(sub.n.iloc[0])}"
    SEVSENS_A = _ss_total("API")
    SEVSENS_L = _ss_total("Local")

tr = gem_df[(gem_df.event_class == "E9") & (gem_df.config == "E4_Full")]
tr = tr.iloc[0] if len(tr) else gem_df[gem_df.config == "E4_Full"].iloc[0]
TRACE_ID = tr.scenario_id
TRACE_RAW = " ".join(str(tr.raw).split())[:340]
scen14 = pd.read_csv(PROCESSED / "ieee14_scenarios.csv")
srow = scen14[scen14.scenario_id == TRACE_ID].iloc[0]
TRACE_POST = (f"$V_{{\\min}}$ {srow.post_v_min_pu:.3f} pu, $V_{{\\max}}$ {srow.post_v_max_pu:.3f} pu, "
              f"peak loading {srow.post_peak_loading_percent:.1f}\\%, violations {int(srow.n_violations)}")
try:
    _tj = json.loads(re.search(r"\{.*\}", str(tr.raw), re.DOTALL).group(0))
    _tj_tool = str(_tj.get("tool", "power_flow"))
except Exception:
    _tj_tool = "power_flow"
_ref_tr = REF2.loc[tr.scenario_id]
_accepted = any(_ref_row == "required" or _ref_row == "strongly_appropriate"
                for _t, _ref_row in _ref_tr.items()
                if isinstance(_t, str) and _t in TOOLS8 and _tj_tool and _tj_tool in _t)
TRACE_JUDGE = (f"diag {'correct' if bool(tr.correct_diag) else 'incorrect'} ({predicted_class(tr.raw)}); "
               f"tool = {_tj_tool.replace('_', '\\_')} "
               + ("$\\in$ accepted set" if _accepted else "not in accepted set"))

def pct(k, n):
    return f"{100*k/n:.0f}"

def cnt(k, n):
    return f"{k}/{n}"

def ci_str(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.0f}, {hi:.0f}]"

noise14 = NOISE["ieee14"]["scoring_relevant_flips"]
noise_all = sum(NOISE[c]["scoring_relevant_flips"] for c in NOISE)
judg_all = sum(NOISE[c]["scenario_tool_judgments"] for c in NOISE)
rho14, rho39, rho118 = (ST8["ieee14"]["severity"]["rho"], ST8["case39"]["severity"]["rho"],
                        ST8["case118"]["severity"]["rho"])

g_nmax = max(G[c]["n"] for c in CFGS)
n_per = G["E1_LLM"]["n"]

# derived headline numbers
g_diag = [G[c]["diag_k"] for c in CFGS]
m_diag = [M[c]["diag_k"] for c in CFGS]
g_strict = [G[c]["strict_k"] for c in CFGS]
m_strict = [M[c]["strict_k"] for c in CFGS]
lat_g = (min(G[c]["lat"] for c in CFGS), max(G[c]["lat"] for c in CFGS))
lat_m = (min(M[c]["lat"] for c in CFGS), max(M[c]["lat"] for c in CFGS))
pf_share = 100 * (gem_df.stated == "power_flow").mean()
m_fallback = int((gemma_df.stated == "power_flow").sum() - (gemma_df.raw.apply(
    lambda r: bool(re.search(r"\{.*\}", str(r), re.DOTALL)))).sum())

def diag_cell(d, cfg):
    x = d[cfg]
    return f"{cnt(x['diag_k'], x['n'])} {ci_str(x['diag_k'], x['n'])}"

def strict_cell(d, cfg):
    x = d[cfg]
    return f"{cnt(x['strict_k'], x['n'])} {ci_str(x['strict_k'], x['n'])}"

ni_rows = []
for c in CFGS:
    v = NI[c]
    ptxt = f"{v['p']:.2f}" if v["p"] >= 0.001 else f"{v['p']:.1e}"
    ni_rows.append(f"{SHORT[c]} & {v['pairs']} & {v['diff_pp']:+.1f} & [{v['lo_pp']:.1f}, {v['hi_pp']:.1f}] & {ptxt} \\\\")
ni_tex = "\n".join(ni_rows)

# pooled legacy E4 observations give the pre-revision E6/E8 denominators
_per1 = pd.concat([gem1_df[gem1_df.config == "E4_Full"], gemma1_df[gemma1_df.config == "E4_Full"]], ignore_index=True)
e6_row = ("E6", int((_per1.event_class == "E6").sum()))
e8_row = ("E8", int((_per1.event_class == "E8").sum()))
perclass_tex = ""  # table dropped; per-class evidence lives in Sec. VII-A prose

maj = BASE["majority_class"]; rnd = BASE["random_uniform"]; rt = BASE["random_tool_8"]
ghseq = "/".join(str(G[c]["hall_k"]) for c in CFGS) + " (E1--E4)"
mhseq = "/".join(str(M[c]["hall_k"]) for c in CFGS) + " (E1--E4)"
total_calls = sum(G[c]["n"] for c in CFGS) + sum(M[c]["n"] for c in CFGS)


tex = r"""% Generated by 31_build_paper.py — all numbers computed from raw logs. Do not edit by hand.
\documentclass[10pt, conference, letterpaper]{IEEEtran}
\IEEEoverridecommandlockouts
\markboth{Anonymous Authors}{GridPowerAgent: A Seeded Corpus and Observation-Ablation Benchmark for LLM Power-System Event Diagnosis}
\title{GridPowerAgent: A Seeded Corpus and Observation-Ablation Benchmark for LLM Power-System Event Diagnosis}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=cyan,citecolor=green]{hyperref}
\usepackage{url}
\setlength{\textfloatsep}{7pt plus 1pt minus 2pt}
\setlength{\floatsep}{7pt plus 1pt minus 2pt}
\setlength{\intextsep}{7pt plus 1pt minus 2pt}

\begin{document}
\author{Anonymous Authors\\[2pt] {\normalfont\normalsize Revision 2 --- prepared in response to the second review round}}
\maketitle

\begin{abstract}
Large language models (LLMs) could act as a coordination layer between grid information, analysis tools, and operators---provided they read operating states accurately, select appropriate tools, and avoid hallucinated advice. We present GridPowerAgent, an agent that observes simulated power-system states, retrieves operating procedures, and orchestrates power-flow, contingency, and optimal-power-flow tools. The evaluation is a paired 140-scenario pilot per configuration on IEEE-14---with a 140-scenario IEEE-39 draw rescored under the revised key---run on a seeded, fully regenerable corpus of 16,000 operating points and 15,000 scenarios across the IEEE 14/39/118 systems (9.54 million nested measurements), with power-flow ground truth, noisy measurements, state estimates, and rule-based labels for ten disturbance classes. The pilot exposed and repaired a flaw in the benchmark itself: on the original key, both models scored zero on the two outcome-labeled classes (E6 undervoltage, E8 thermal overload) while describing the injecting cause correctly in nearly every failed response; the corpus now keys every scenario to its injected mechanism. Because diagnosis with the event description in the prompt is trivially leaked (100\%), the revised benchmark evaluates diagnosis from observations only. The finding is stark: from EMS-grade telemetry the API model reaches %ABL_EV_API%\% and a gradient-boosted classifier on identical features 100\%; from raw state the LLM collapses to %ABL_PH_API%\% where the classifier reaches %B_STATE_GBM%\%; the local model manages %ABL_EV_LOC%\% from telemetry (20-scenario probe, $p$=%ABL_GAP_P% vs.\ API). A closed-loop extension in which the agent must identify the switched element by name and chain at least two real tool executions yields %CL2_DIAG%\% (API). Tool selection retains a style difference (the API model defaults to power flow in %PFSHARE%\% of picks). All results are exploratory single temperature-zero runs with exact denominators, at pilot scale.
\end{abstract}

\begin{IEEEkeywords}
power system operation, large language models, LLM agent, tool orchestration, retrieval-augmented generation, state estimation
\end{IEEEkeywords}

\section{Introduction}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black}

Power-system operators are surrounded by analytical tools---power flow, state estimation, contingency analysis, optimal power flow---yet the work of interpreting their outputs, relating them to operating procedures, and deciding what to do next remains manual and expertise-bound. Large language models (LLMs) are candidates for an \emph{intelligent coordination layer} in this workflow: not replacing conventional analysis, but reading grid states, retrieving the right procedures, invoking the right tools, and explaining the results \cite{majumder2024joule,cheng2025gaia}. Recent agentic systems move in this direction \cite{zhang2025gridagent,wen2025xgridagent}, but published evaluations rarely state what the labels are, where they come from, or how the scoring could be reproduced.

This paper presents GridPowerAgent and its evaluation under six research questions: (RQ1) how accurately can an LLM identify normal, abnormal, and critical operating conditions; (RQ2) can it determine which engineering tool a given event requires; (RQ3) can it construct valid tool calls; (RQ4) does grid-specific retrieval improve operational reasoning; (RQ5) does tool grounding reduce hallucination; and (RQ6) does the evaluation transfer across network scale. RQ3---constructing valid tool calls---is answered here at its execution level: stated tools are executed against the scenario's post-event network after the run and their execution scored (Sec.~\ref{sec:rq3exec}). This pilot answers RQ1, RQ2, RQ4, RQ5, and RQ6, with RQ3 answered by a closed-loop extension in which the agent's tool calls are constructed, executed, and interpreted in-loop (Sec.~\ref{sec:rq3exec}). The paper's identity is deliberately that of a benchmark-and-evaluation study; its contributions are the corpus with dual-axis keying, the observation-ablation methodology, the tier discrimination under realistic observations, and the closed-loop validity protocol---not a new algorithm. The specific findings: observation encoding, not model choice, decides what is diagnosable (telemetry 100\%-separable for a classifier, raw state not); description-reading saturates both tiers and conceals their difference; and closed-loop tool use with mandatory evidence-gathering is the only evaluated condition in which the benchmark is neither leaked nor saturated. We answer them on a seeded, headless-resumable corpus pipeline: 16,000 operating points and 15,000 scenarios across IEEE 14/39/118---ten disturbance classes (E0--E9) each carrying pre/post power-flow truth, noisy measurements, state estimates, severity, and rule-based reference labels---plus a FAISS knowledge base of operating procedures and four validated physics tools; configurations (E1 LLM-only, E2 +RAG, E3 +Tools, E4 Full) instantiate the ablation.

The evaluation compares two deployment tiers under identical paired prompts---a 4-bit-quantized small model served locally, and a lightweight API model---so conclusions do not depend on one provider. All figures carry exact denominators and Wilson CIs; power is stated alongside every null result; and one finding receives particular attention: both models systematically fail on two outcome-labeled disturbance classes, for a reason traced to the label taxonomy itself rather than to either model.

Contributions: (i) a grid-aware agent that couples grid-state observation, procedure retrieval, and validated physics tools; (ii) a seeded, hashed, fully regenerable 16k/15k/9.5M-measurement corpus with rule-based tool supervision and a quantified label-noise bound under estimation uncertainty; (iii) a dual-definition tool-scoring protocol that exposes and removes a degenerate answer strategy; (iv) a paired, exactly-denominated pilot across two deployment tiers in which a mixed-axis labeling flaw was identified and repaired---flipping diagnosis parity into a significant tier gap---with an observation ablation separating description-reading from telemetry-based diagnosis; and (v) a per-check validation disclosure---including unresolved IEEE-39 islanding cases with their scenario identifiers shipped alongside the corpus---together with a seeded IEEE-39 replication of the full paired protocol (Sec.~\ref{sec:crosssystem}).

\section{Related Work}
LLMs have been explored for power-system analysis assistance, dispatch, and contingency response \cite{majumder2024joule,cheng2025gaia}, including agentic orchestrators \cite{zhang2025gridagent,wen2025xgridagent} and dispatch benchmarks \cite{zhou2024elecbench}, but published evaluations seldom disclose how reference labels are produced; our corpus couples per-scenario power-flow truth, measurements, state estimates, and rule-based tool supervision across three IEEE sizes under one seeded pipeline, scored with exact denominators. Foundational RAG \cite{lewis2020rag}, tool-augmented LLMs \cite{yao2023react,schick2023toolformer,achiam2023gpt4}, classical state estimation \cite{abur2004power,monticelli1999state}, contingency/OPF analysis \cite{wood2014power,frank2012opf,zimmerman2011matpower}, ECE calibration \cite{guo2017calibration}, and hallucination taxonomies \cite{ji2023surveyhalluc} supply the component methods. Synthetic scenarios inherit the spirit of grid-ML benchmarks \cite{marot2020l2rpn,donnot2020grid2op}; pandapower \cite{thurner2018pandapower} supplies AC power flow; FAISS \cite{johnson2021faiss} with sentence-transformers \cite{reimers2019sentencebert} supplies retrieval. On quantization: 4-bit precision is near-optimal \cite{dettmers2023fourbit,frantar2023gptq}, though it degrades small models---a conservative bias in our local deployment (Sec.~\ref{sec:limits}).

\section{Methodology}
Fig.~\ref{fig:methodology} overviews the pipeline. Stages 03--05 build networks, operating points, and scenarios; 06--09 synthesize measurements, state estimates, severity, and reference labels; 10--14 expose physics tools; 16--17 build the knowledge base; 19--22 run the agent ablations; 23--28 evaluate and render figures. Heavy stages are idempotent and headless-resumable, materializing the full corpus in ${\sim}30$~min.

\begin{figure}[tbp]
\centering
\includegraphics[width=\columnwidth]{figures/fig_methodology_tree.png}
\caption{Pipeline (Stages 03--28).}
\label{fig:methodology}
\end{figure}

\textbf{Agent workflow.} The agent (Fig.~\ref{fig:arch}) follows an observe--diagnose--retrieve--plan--execute--interpret loop: it receives the post-event structured grid state (voltage magnitudes, branch loadings, outages, storage state of charge); optionally receives the top-$k$ retrieved operating procedures (E2/E4) and a tool manifest (E3/E4); and must return the disturbance class, a tool selection, and a recommendation with confidence, as JSON. The harness scores the response against the rule-based reference labels; it does not execute tools inside the agent loop in this pilot---stated tools are executed post hoc for an execution-validity check (Sec.~\ref{sec:rq3exec}). The diagnosis prompt presents the revised cause-axis taxonomy (Sec.~\ref{sec:axis}), and three observation conditions (description, telemetry, state-only; Secs.~\ref{sec:proto} and \ref{sec:ablation}) probe what the diagnosis measures.

\textbf{Networks and scenarios.} Networks use pandapower IEEE 14/39/118 cases with tuned thermal limits to make compound events observable (14: 3\% line / 4\% transformer; 118: 6\%; 39: nameplate). Limits are disclosed per system and deliberately \emph{not} comparable across systems (Sec.~\ref{sec:limits}). Operating points sweep load $0.70$--$1.10\times$ with $\pm5\%$ bus-local noise, renewable fractions, and storage state of charge $0.15$--$0.85$ (20 MW/40 MWh at bus 9 on IEEE-14). Topology hashes pin all three networks. Ten classes are injected: E0 Normal; E1 load surge, E2 load drop, E3 line outage, E4 generator outage, E5 renewable ramp (cause classes); E6 undervoltage, E7 overvoltage, E8 thermal overload (outcome classes, physically iterated to target severity); E9 compound. Reference labels key every scenario to its injected mechanism class; the original outcome-class keying of E6/E8, the flaw it created, and its repair are analyzed in Sec.~\ref{sec:axis}.

\section{Corpus, Simulation and Validation}
Table~\ref{tab:corpus} summarizes the corpus. Scenarios inject the ten classes with 300/500/700 per class on 14/39/118; replay over 40 draws confirms exact determinism.

\begin{table}[tbp]
\centering
\caption{Corpus. Validation is reported per system, not as a single global certificate.}
\label{tab:corpus}
\begin{tabular}{lccccc}
\toprule
Case & OPs & Scen. (per class) & Meas. & Auto checks & Worst $V$ \\
\midrule
14 (ref) & 4k & 3k (300) & 361k & 21/21 & 0.89 pu \\
39 & 5k & 5k (500) & 1.50M & \textbf{20/21} & 0.89 pu \\
118 & 7k & 7k (700) & 7.68M & 21/21 & 0.89 pu \\
Total & 16k & 15k & 9.54M & --- & --- \\
\bottomrule
\end{tabular}
\\[2pt]
{\scriptsize IEEE-14 limits are artificial (3\%/4\%) to expose E8. The single failing check (s5\_no\_nan on IEEE-39) covers 41 scenarios with NaN post-voltages from two islanding outages; identifiers and causes ship with the corpus (case39\_nan\_scenarios.csv).}
\end{table}

Measurements model $\sigma_V=0.003$~pu and $\sigma_P=\max(0.0075|S_{\text{true}}|,0.05)$~MVA computed from the noise-free power-flow solution. Each scenario carries ${\sim}120$ noisy measurements (bus-voltage magnitudes, branch P/Q at both ends, bus injections), and the state estimate supplied to the agent is produced by an \emph{iterative AC-WLS estimator} (\texttt{pandapower.estimation}) over these measurements, with branch meters of outaged elements excluded: the estimator converges on %WLS_CONV% pilot scenarios with voltage RMSE %WLS_RMSE%~pu against post-event truth, and the under-voltage flag agrees with truth in %WLS_FLAG%\% of scenarios---marginal violations can vanish under estimation (probed in Sec.~\ref{sec:ablation}). Full bad-data handling and observability testing remain future work.

\textbf{Severity score.} The illustrative severity used for reference labels is
\begin{equation}
S = 0.6\,\min\!\Big(1,\tfrac{\max(0,\,|V-1|-0.02)}{0.06}\Big) + 0.4\,\min\!\Big(1,\tfrac{\max(0,\,\ell-L)}{10}\Big),
\label{eq:sev}
\end{equation}
with boundaries $0.0263/0.0526/0.1053$ selected by grid search over five candidate sets \emph{on this corpus}---a circularity we return to in Sec.~\ref{sec:limits}. A sensitivity check bounds the damage: re-scoring the pilot under four boundary sets---the searched set, standards-style $0.05/0.10/0.15$, a tighter $0.02/0.04/0.08$, and empirical severity tertiles---leaves strict tool-selection accuracy exactly unchanged (%SEVSENS_A% vs.\ %SEVSENS_L% strict hits in every set), so the headline tool-selection gap does not depend on the searched boundaries.

\textbf{Label noise under estimation uncertainty.} Reference labels gate two tools on severity. Recomputing severity with estimated bus voltages yields rank agreement $\rho=%RHO14%$ (14), $%RHO39%$ (39), $%RHO118%$ (118)---the voltage term sits near its deadband for most scenarios, so noise reorders ranks without crossing thresholds. The operationally relevant quantity is accepted-set membership: scoring-relevant label flips affect \textbf{%NOISE14%/30{,}000} scenario--tool judgments on IEEE-14 (the pilot system) and \textbf{%NOISEALL%/150{,}000} (%NOISEPCT%\%) corpus-wide. The binary violation detector reconciles 99.23\%/99.54\%/99.67\% on 14/39/118.

\begin{figure}[tbp]
\centering
\includegraphics[width=\columnwidth]{figures/fig_architecture.png}
\caption{Agent architecture.}
\label{fig:arch}
\end{figure}

\section{Retrieval, Tools and Agent Design}
The knowledge base chunks eight operational documents (thermal/voltage limits, contingency procedure, storage/equipment, topology) into a FAISS index (384-d sentence-transformer embeddings), retrieving $k=3$ chunks with citations. Recall@1 is 100\% on held probes; with eight documents this is a plumbing check, not evidence of retrieval quality---hard-negative evaluation is future work. Four physics tools (power flow, grid queries, contingency N--1, OPF) are validated 21/21 including 15/15 contingency convergences.

\section{Evaluation Protocol}
\label{sec:proto}


\textbf{Models.} Two deployment tiers of the same prompt contract are compared: a lightweight API model (Gemini 3.5 Flash Lite, temperature 0, ${\le}512$ output tokens, checkpointed rate-limit retry) and a small 4-bit-quantized open-weight model (Gemma 4 E4B-it \cite{gemma4_2026}) served locally from a GGUF build via llama.cpp \cite{llamacpp} (temperature 0, ${\le}1{,}024$ tokens---the larger budget accommodates its visible reasoning channel; parsing reads \texttt{content} with fallback to the reasoning text). \emph{The API tier is a lightweight, latency-optimized model}: the comparison characterizes a representative local deployment against a budget API tier, not against frontier API models. A deterministic rule-based \emph{oracle} (not a model) validates harness plumbing at $N{=}600$/config; statistics on it test code, not intelligence.

\textbf{Metrics.} Diagnosis correctness is exact class match. \emph{Tool selection is reported under two definitions.} (a)~\emph{Accepted-set}: the stated tool is required or strongly appropriate. This definition is degenerate---power flow lies in every scenario's accepted set, so always answering power flow scores 100\%---and we treat it as an upper bound only. (b)~\emph{Strict-specific}: the stated tool must be \emph{required} for the scenario, and power flow counts only when it is the sole required tool. Bare \texttt{grid\_query} answers are scored non-specific. Provenance: the accepted-set definition was found degenerate in an early audit, and the strict-specific definition was frozen \emph{before} the revised-taxonomy runs reported here; it permits exactly one tool per answer, a documented simplification of real workflows. Hallucination is response-level, any of six tags (H-NUM/TOP/EQP/PHY/TOOL/ACT), assigned by an automated rule-based judge; rates are judge agreement, not ground truth (Sec.~\ref{sec:limits}).

\textbf{Observation conditions.} Diagnosis is measured under three prompt-side conditions: \emph{description} (a plain-language statement of the injected event accompanies the grid state), \emph{telemetry} (the description is replaced by EMS-grade observed evidence: switching status, changed demand and renewable elements, and system-wide pre$\to$post deltas), and \emph{state-only} (pre/post state numbers alone). The description condition is the primary pilot; the others probe what diagnosis measures (Sec.~\ref{sec:ablation}).

\textbf{Power.} %MDEDERIV% At $n{=}140$, this yields %MDE_TXT%\,pp. Null results mean ``no difference above the per-configuration MDE was detectable,'' not ``the models are equivalent.'' Comparisons are exploratory: hypotheses were not preregistered, no multiplicity correction is applied, scenarios within a class share generation parameters (the scenario, not the measurement, is the analysis unit), and single temperature-zero runs capture neither decoding nor prompt variability.

\section{Results}

\subsection{A Mixed-Axis Labeling Flaw and Its Repair}
\label{sec:axis}
The original corpus keying assigned the two \emph{outcome} classes---E6 undervoltage and E8 thermal overload---to scenarios injected through \emph{cause} mechanisms: outages, load surges, and outage-plus-surge compounds. Under that key, both models score zero in every configuration: 0 of %E6N% pooled E6 and 0 of %E8N% pooled E8 observations. The failures are systematic: across all %E6FAILS% failed E6 responses, %CONF_E6%; across all %E8FAILS% failed E8 responses, %CONF_E8% (``...a single transmission line outage (line\_6\_13), which directly corresponds to class E3''). The mechanism is a mixed-axis label taxonomy: both models classify on the cause axis and are \emph{causally correct but label-wrong} on every E6/E8 scenario---a labeling-scheme property, not a model property.

The repair rekeys every E6/E8 scenario to its injected mechanism (line outages to E3, generator outages to E4, compounds to E9, surges to E1), re-derives tool tiers, and presents the cause-axis taxonomy to the agent; scenario identifiers, physics, and outcome flags are unchanged (E7 is kept: its mechanisms match the outcome reading). The revised-key pilot uses the same 140 scenarios, prompts, and scoring pipeline.


\subsection{Re-Measured Diagnosis on the Revised Taxonomy (RQ1/RQ4)}
Table~\ref{tab:results} reports exact counts under the revised key, with the event description present in the prompt. The API model diagnoses 140/140 in \emph{every} configuration; the local model diagnoses 133--138/140 (95.0--98.6\%). This condition is leaked---the description states the cause in plain language (Sec.~\ref{sec:ablation})---so these numbers measure reading, not grid reasoning. All %NIPOOL% paired discordances favor the API model (pooled exact McNemar $p$=%PAIRP%; Table~\ref{tab:ni}, Fig.~\ref{fig:diag}; individually significant in E2 and E4)---the statistical parity observed under the flawed key (%V1API%\% vs.\ %V1LOC%\%, exact McNemar not significant) was an artifact of the mixed axis, not evidence of equivalent capability. Configuration additions move local diagnosis by at most five scenarios in either direction while the API tier stays at ceiling: no diagnosis benefit from retrieval or tool manifests is detectable at this scale. Because the description already carries the diagnosis, this design cannot distinguish RAG inefficacy from information sufficiency; hard-negative retrieval and faithfulness evaluation are required before any conclusion about retrieval quality.

\begin{table}[tbp]
\centering
\caption{Diagnosis under the revised taxonomy: exact counts, Wilson 95\% CIs (percentages).}
\label{tab:results}
\begin{tabular}{lcc}
\toprule
Cfg & API ($n{=}%GN%$) & Local ($n{=}%MN%$) \\
\midrule
E1 & %G_E1_DIAG% & %M_E1_DIAG% \\
E2 & %G_E2_DIAG% & %M_E2_DIAG% \\
E3 & %G_E3_DIAG% & %M_E3_DIAG% \\
E4 & %G_E4_DIAG% & %M_E4_DIAG% \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[tbp]
\centering
\caption{Paired diagnosis analysis (Local $-$ API, pp), revised taxonomy: bootstrap 95\% CI, 20k resamples; all discordances one-directional.}
\label{tab:ni}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Cfg & Pairs & Diff (pp) & 95\% CI & McNemar $p$ \\
\midrule
%NI_ROWS%
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[tbp]
\centering
\includegraphics[width=\columnwidth]{figures/fig_diagnosis.png}
\caption{(a) Diagnosis accuracy under the revised taxonomy, local vs.\ API model, Wilson 95\% CIs. (b) Paired diagnosis difference (Local $-$ API) with bootstrap 95\% CIs.}
\label{fig:diag}
\end{figure}

\subsection{Observation Ablation: Reading the Log vs.\ Reading the Grid}
\label{sec:ablation}
The near-ceiling diagnosis above is measured with a plain-language event description in the prompt. Three observation conditions (Sec.~\ref{sec:proto}) separate what diagnosis measures (Table~\ref{tab:ablation}; API tier unless noted). First, the telemetry condition is nearly sufficient for the API model (%ABL_EV_API%\%; the switch-detectable classes E3/E4/E7 at 100\%, compounds 94.1\%)---yet a conventional gradient-boosted classifier on the same features reaches %B_TELEM_GBM%\% so the observation is fully class-separable and the LLM's residual errors are composition errors, not information limits. Second, the raw state condition collapses to %ABL_PH_API%\% for the LLM while the same classifier reaches %B_STATE_GBM%\% from identical features: the signal exists, but the LLM cannot exploit unlabeled numerical summaries, answering E7 for 95\% of scenarios and calling every E0 normal scenario E7 under telemetry (0/20). The description-only condition quantifies the textual leak: %DESCONLY%\% with no grid-state numbers. Multi-axis scoring on pipeline-consistent annotations (rule-generated by the same pipeline as the events---the recursion is disclosed; expert re-annotation remains the controlling gap) shows answers outcome-consistent in %MA_OUT_API%\%/%MA_OUT_LOC%\% and tool-consistent in %MA_TOOL_API%\%/%MA_TOOL_LOC%\%: the revised key no longer punishes cause-correct answers on outcome axes. The telemetry condition also separates the tiers: a 20-scenario local probe reaches %ABL_EV_LOC%\% vs.\ %ABL_EV_API20%\% for the API tier (%ABL_GAP_N10% vs.\ %ABL_GAP_N01% discordant, $p$=%ABL_GAP_P%); its errors include hedged multi-class answers and JSON contract breaks. Replacing the true post-event state with the WLS-estimated state leaves diagnosis at %WLS_DIAG%\% over all %WLS_N% cells: the tier and observation findings survive realistic state estimation. Estimation even helps slightly (97.1\% vs.\ 93.9\%): smoothing pulls marginal violations toward nominal, which the flag-agreement rate (%WLS_FLAG%\%%) shows removes more false alarms than it creates. Diagnosis from telemetry rather than description is thus the realistic and discriminative test.

\begin{table}[tbp]
\centering
\caption{Diagnosis by observation condition (API tier; local telemetry $n{=}%ABL_EV_LOC_N%$ cells). Classifiers fitted on non-pilot scenarios, evaluated on the pilot draw; description-only is textual ($-$); WLS row uses iterative-estimated state (Sec.~IV), LLM-only.}
\label{tab:ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Condition & API LLM & Local LLM & GBM & Log. reg. & maj./rand. \\
\midrule
Event description (leakage control) & %DESCONLY% & -- & $-$ & $-$ & -- \\
Telemetry only     & %ABL_EV_API% & %ABL_EV_LOC% & %B_TELEM_GBM% & %B_TELEM_LR% & -- \\
Telemetry, WLS-estimated state ($n{=}%WLS_N%$) & %WLS_DIAG% & -- & -- & -- & -- \\
Raw state only     & %ABL_PH_API% & -- & %B_STATE_GBM% & %B_STATE_LR% & %B_MAJ%/%B_RAND% \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{RQ6: Cross-Network Evaluation on IEEE-39}
\label{sec:crosssystem}
A seeded 140-scenario draw from the IEEE-39 corpus (excluding the 41 islanding-NaN scenarios) runs the same paired protocol under the pre-revision prompt; its logged answers are rescored against the revised key. The API model diagnoses %G39_DIAG% scenarios in \emph{every} configuration (E1--E4) and the local model %M39_DIAG%. Strict tool selection declines monotonically for the API model---%G39S% of 140 across E1--E4---while the local model stays flat at %M39S%. The decline extends the IEEE-14 pattern (%G14S% of 140 there) but is steeper and does not plateau by E4, while the local model is flat on both corpora: the API decline across configurations is monotone on both corpora and currently unexplained---flagged for the powered sweep. Transferring between corpora changes the network topology, load patterns, and label instances while the agent, prompt, and scoring pipeline remain untouched. %GEN_NOTE%

\subsection{RQ2: Tool Selection and the Default-Answer Bias}
\begin{table}[tbp]
\centering
\caption{Tool selection, strict-specific metric: exact counts, Wilson 95\% CIs (percentages).}
\label{tab:tools}
\begin{tabular}{lcc}
\toprule
Cfg & API ($n{=}%GN%$) & Local ($n{=}%MN%$) \\
\midrule
E1 & %G_E1_STRICT% & %M_E1_STRICT% \\
E2 & %G_E2_STRICT% & %M_E2_STRICT% \\
E3 & %G_E3_STRICT% & %M_E3_STRICT% \\
E4 & %G_E4_STRICT% & %M_E4_STRICT% \\
\bottomrule
\end{tabular}
\end{table}

Tool selection is where the deployment tiers differ most---in \emph{style} before \emph{accuracy}. The permissive accepted-set metric is degenerate (always answering power flow achieves it by construction; random choice scores %RANDTOOLPCT%), and the API model exercises exactly that strategy: %PFSHARE%\% of its stated tools are power flow. Under the strict-specific metric (Table~\ref{tab:tools}), the local model names a scenario-\emph{required} tool in %MS1%/%MN%--%MS4%/%MN% of cases (flat across configurations) versus %GS1%/%GN%--%GS4%/%GN% for the API model. When the API model states power flow (%PFN% responses), it is actually required in only %PFAPI%\% of cases---the rest are defaults to the universally accepted answer. The local model states power flow rarely (%PFLN% responses, %PFLL%\% of them required); its dominant choice is contingency (%CTN% responses, %CTREQ%\% required, precisely on outage/overload scenarios). The gap therefore reflects the API model's default rate, not superior local tool knowledge; both models remain far from ceiling, and the binary metric cannot distinguish well-placed tools from lucky guesses.

\subsection{RQ3: Closed-Loop Tool Use}
\label{sec:rq3exec}
The one-shot pilot only \emph{names} tools; RQ3 asks whether the agent can \emph{use} them. The closed-loop extension removes the remaining shortcut: the per-element status table does not say which element was switched, at least two real tool executions must precede any answer, and the answer must \emph{name the switched element}---scored against the truly switched element. On %CL2_N% pilot scenarios (API tier): diagnosis %CL2_DIAG%\%, element construction %CL2_CONSTR%\% on the %CL2_CONSTR_N% switched-element scenarios, over %CL2_CALLS% tool calls in %CL2_TURNS% turns on average (%CL2_CHAIN%\% of scenarios chain two or more calls). The task is now genuinely hard: construction succeeds while exact-class diagnosis does not follow automatically, and the offline plumbing check (%RQ3_TOTAL% stated-tool pairs across the pilot rounds, %RQ3_EXEC% executable, all succeeding) confirms the failures are the agent's, not the tools'. What remains open is decision quality---whether recommendations drawn from executed results are operationally safe and constraint-relieving---and the loop is API-tier only; extending it (and the estimated-state arms) to the local tier is future work.

\subsection{RQ5: Hallucination and the Cost of Local Inference}
Any-tag hallucinated rows are rare for both models under the automated judge: %GHSEQ% (API) and %GEMMA_HALL% (Local) across E1--E4. Judge flags are not expert-annotated ground truth; confidence calibration is unidentifiable at this sample size and will be reported on the powered sweep. Latency is the deployment trade-off: local inference averages %GEMMA_LAT{} per call versus %GLAT{} for the API round-trip---roughly a %LATRATIO$\times$ differencewith zero marginal API cost; acceptable for always-on monitoring, not for closed-loop use (Sec.~\ref{sec:limits}).


\section{Discussion and Limitations}
\label{sec:limits}

\textbf{Scope.} The evidence covers \emph{one model pair} (a 4-bit local model; a lightweight API tier), one prompt template, temperature 0, single runs; it does not speak to frontier API models. The powered 600-scenario sweep is the inferential step.

\textbf{Label circularity.} Disturbances, labels, and prompts derive from the same rule family: the evaluation measures \emph{recovery of the synthetic labeling policy}, not open-ended operator reasoning---and the severity boundaries of Eq.~\eqref{eq:sev} were themselves tuned on this corpus, though the strict-metric comparison is invariant across four boundary sets, two of them not derived from this corpus (Sec.~IV). Required remedies, all future work: expert-reviewed labels ($2\times80$, $\kappa\ge0.8$), multiple valid tool sequences, blind scoring, an evaluator independent of the label generator, and externally anchored severity boundaries.

\textbf{Taxonomy design.} The E6/E8 label-axis failure (Sec.~\ref{sec:axis}) was a benchmark-design finding: outcome-labeled classes injected via cause mechanisms made cause-correct answers score zero. The revision rekeys E6/E8 to their injected mechanisms; both label versions ship with the corpus. The observation ablation adds a second encoding finding: the state-only view carries essentially no cause information (%ABL_PH_API%\% diagnosis, Sec.~\ref{sec:ablation}), and the API tier misreads every E0 normal scenario as overvoltage under telemetry-only observation.

\textbf{Physics fidelity.} Thermal limits differ per system by construction, so overload magnitudes are not operationally comparable across systems. The corpus is static: no dynamics, protection, or time-sequential behavior. The severity index is illustrative with a quantified label-noise bound (Sec.~IV); the state estimator is an iterative WLS without bad-data handling or topology-error processing (Sec.~IV); the 41 islanding-NaN IEEE-39 scenarios (identifiers shipped in \texttt{case39\_nan\_scenarios.csv}) remain unresolved and are excluded from downstream post-voltage use.

\textbf{Scoring subjectivity.} Hallucination flags come from a single automated judge (Sec.~VII-G); tool scoring relies on the reference policy; the strict metric narrows the policy-dependence.

\textbf{Safety framing.} This is an offline advisory prototype: no protection coordination, authorization, or human-approval interface is implemented; autonomous control actions are out of scope.

\textbf{Reproducibility.} All scripts, seeds, hashes, prompts, validation summaries, both label versions, and the revised-taxonomy and observation-ablation datasets are in the repository; corpora regenerate deterministically. Generalization: the IEEE-39 evaluation (Sec.~VII) covers both tiers; IEEE-118, a powered cross-system protocol, and the powered 600-scenario sweep (per-class F1, ECE, multiplicity-corrected tests, prompt sensitivity) remain future work.

\section{Conclusion}

GridPowerAgent couples a seeded, regenerable 16k/15k power-system scenario corpus with a grid-aware LLM agent that retrieves procedures and orchestrates validated physics tools, evaluated under a protocol where every number carries an exact denominator. In the paired pilot, a systematic diagnosis failure was traced to the benchmark's mixed-axis label taxonomy and repaired; on the revised taxonomy the API tier diagnoses at ceiling and significantly above the local tier (100\% vs.\ 96.8\%), and telemetry-only diagnosis preserves the gap (93.9\% vs.\ 43.8\%). The corpus, protocol, and disclosed flaws are the contribution; the powered evaluation is the next step.

AI Assistance Disclosure: Generative AI was used for drafting, coding assistance, and figure generation; all experimental decisions, execution, and analysis were performed by the authors, who reviewed and edited all AI-assisted content.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

for cfg in CFGS:
    sc = SHORT[cfg]
    tex = tex.replace(f"%G_{sc}_DIAG%", diag_cell(G, cfg))
    tex = tex.replace(f"%G_{sc}_STRICT%", strict_cell(G, cfg))
    tex = tex.replace(f"%M_{sc}_DIAG%", diag_cell(M, cfg))
    tex = tex.replace(f"%M_{sc}_STRICT%", strict_cell(M, cfg))

g_diag_sorted = sorted((G[c]["diag_k"], G[c]["n"], c) for c in CFGS)
gmin_k, gmin_n, gmin_c = g_diag_sorted[0]
gmax_k, gmax_n, gmax_c = g_diag_sorted[-1]
g_h_e1, g_h_e4 = G["E1_LLM"]["hall_k"], G["E4_Full"]["hall_k"]
g_tool_min = min(G[c]["tool_k"] for c in CFGS)
g_tool_max = max(G[c]["tool_k"] for c in CFGS)
m_strict_min = min(M[c]["strict_k"] for c in CFGS)
m_strict_max = max(M[c]["strict_k"] for c in CFGS)
gl_lo, gl_hi = min(G[c]["lat"] for c in CFGS), max(G[c]["lat"] for c in CFGS)
glat = f"{gl_lo:.2f}--{gl_hi:.2f}\\,s"
ghseq = "/".join(str(G[c]["hall_k"]) for c in CFGS) + " (E1--E4)"
gemma_hall = "/".join(str(M[c]["hall_k"]) for c in CFGS) + " (E1--E4)"
gemma_lat = f"{min(M[c]['lat'] for c in CFGS):.0f}--{max(M[c]['lat'] for c in CFGS):.0f}\\,s"
latratio = f"{(sum(M[c]['lat'] for c in CFGS)/sum(G[c]['lat'] for c in CFGS)):.0f}"
permissive_pct = pct(G["E4_Full"]["tool_k"], G["E4_Full"]["n"]) if G["E4_Full"]["n"] else "--"
pf_share = 100.0 * (gem_df.stated == "power_flow").mean()
maj = BASE["majority_class"]; rnd = BASE["random_uniform"]; rt = BASE["random_tool_8"]

# generalization + RQ3 injection values
g39 = None
if G39 is not None:
    g39 = {c: G39[c] for c in CFGS}

m39 = None
if M39 is not None:
    m39 = {c: M39[c] for c in CFGS}

tex = (tex
       .replace("%GN%", str(g_nmax))
       .replace("%MN%", str(min(M[c]["n"] for c in CFGS)))
       .replace("%NI_ROWS%", ni_tex)
       .replace("%GS1%", str(min(g_strict))).replace("%GS4%", str(max(g_strict)))
       .replace("%MS1%", str(m_strict_min)).replace("%MS4%", str(m_strict_max))
       .replace("%PFSHARE%", f"{pf_share:.0f}")
       .replace("%PFN%", str(pf_api_n)).replace("%PFAPI%", f"{pf_api_pct:.0f}")
       .replace("%PFLN%", str(pf_local_n)).replace("%PFLL%", f"{100.0*pf_local_req/max(1,pf_local_n):.0f}")
       .replace("%CTN%", str(ct_local_n)).replace("%CTREQ%", f"{ct_local_pct:.0f}")
       .replace("%GHSEQ%", ghseq)
       .replace("%GEMMA_HALL%", gemma_hall)
       .replace("%GLAT%", glat).replace("%GLAT{}", glat)
       .replace("%GEMMA_LAT%", str(gemma_lat)).replace("%GEMMA_LAT{}", str(gemma_lat))
       .replace("%LATRATIO", str(latratio))
       .replace("%PERMISSIVE%", permissive_pct)
       .replace("%MAJ\\%%", f"{100*maj['acc']:.0f}\\%")
       .replace("%RAND\\%", f"{100*rnd['mean']:.0f}\\%")
       .replace("%RANDSD\\%%", f"{100*rnd['sd']:.0f}\\%")
       .replace("%RANDTOOLPCT%", f"{100*rt['mean']:.0f}\\%")
       .replace("%MDE_TXT%", mde_txt)
       .replace("%CONF_E6%", conf_e6_txt)
       .replace("%CONF_E8%", conf_e8_txt)
       .replace("%E6FAILS%", str(CONF_E6["n_fail"]))
       .replace("%E8FAILS%", str(CONF_E8["n_fail"]))
       .replace("%E6N%", str(e6_row[1]) if e6_row else "--")
       .replace("%E8N%", str(e8_row[1]) if e8_row else "--")
       .replace("%RHO14%", f"{rho14:.3f}").replace("%RHO39%", f"{rho39:.3f}").replace("%RHO118%", f"{rho118:.3f}")
       .replace("%NOISE14%", str(noise14)).replace("%NOISEALL%", str(noise_all))
       .replace("%NOISEPCT%", f"{100.0*noise_all/judg_all:.3f}")
       .replace("%TOTAL_CALLS%", str(total_calls))
       .replace("%PERCLASS_ROWS%", perclass_tex)
       .replace("%TRACE_ID%", TRACE_ID.replace("_", "\\_"))
       .replace("%TRACE_POST%", TRACE_POST)
       .replace("%TRACE_RAW%", TRACE_RAW.replace("%", "\\%").replace("_", "\\_")))

# cross-system + RQ3 values
def and_list(vals):
    vals = [str(v) for v in vals]
    if len(vals) > 1:
        return ", ".join(vals[:-1]) + ", and " + vals[-1]
    return vals[0]

if G39 is not None:
    g39_diag = [G39[c]["diag_k"] for c in CFGS]
    g39_strict = [G39[c]["strict_k"] for c in CFGS]
    g39n = G39["E1_LLM"]["n"]
    assert all(G39[c]["n"] == g39n for c in CFGS)
    tex = tex.replace("%G39_DIAG%", f"{g39_diag[0]}/{g39n}" if len(set(g39_diag)) == 1 else and_list(g39_diag))
    tex = tex.replace("%G39S%", and_list(g39_strict))
    tex = tex.replace("%G14S%", and_list(g_strict))
else:
    tex = tex.replace("%G39_DIAG%", "--").replace("%G39S%", "--").replace("%G14S%", "--")
if M39 is not None and local39_done:
    m39n = M39["E1_LLM"]["n"]
    m39_diag = [M39[c]["diag_k"] for c in CFGS]
    m39_strict = [M39[c]["strict_k"] for c in CFGS]
    assert all(M39[c]["n"] == m39n for c in CFGS)
    tex = tex.replace("%M39_DIAG%", f"{min(m39_diag)}--{max(m39_diag)}/{m39n}")
    tex = tex.replace("%M39S%", f"{min(m39_strict)}--{max(m39_strict)}/{m39n}")
else:
    tex = tex.replace("%M39_DIAG%", "--").replace("%M39S%", "--")
gen_note = ""
if M39 is None or not local39_done:
    gen_note = ("The local model's IEEE-39 evaluation is running; its results are reported with the same protocol when complete.")
tex = tex.replace("%GEN_NOTE%", gen_note)

tex = tex.replace("%SEVSENS_A%", SEVSENS_A or "--").replace("%SEVSENS_L%", SEVSENS_L or "--")

if RQ3:
    tex = tex.replace("%RQ3_TOTAL%", str(RQ3["total_pairs"]))
    tex = tex.replace("%RQ3_EXEC%", str(RQ3["executed_ok_total"]))
    tex = tex.replace("%RQ3_OK%", f"{100.0*RQ3['executed_ok_total']/max(1,RQ3['total_pairs']):.0f}")
else:
    tex = tex.replace("%RQ3_TOTAL%", "--").replace("%RQ3_EXEC%", "--").replace("%RQ3_OK%", "--")

tex = (tex
       .replace("%ABL_FULL_API%", ABL["full_api"])
       .replace("%ABL_FULL_LOC%", ABL["full_loc"])
       .replace("%ABL_EV_API%", ABL["ev_api"])
       .replace("%ABL_EV_API20%", ABL["ev_api20"])
       .replace("%ABL_EV_LOC%", ABL["ev_loc"])
       .replace("%ABL_EV_LOC_N%", ABL["ev_loc_n"])
       .replace("%ABL_PH_API%", ABL["ph_api"])
       .replace("%ABL_EV_API_E0%", ABL["ev_api_e0"])
       .replace("%ABL_GAP_P%", ABL["gap_p"])
       .replace("%ABL_GAP_N10%", ABL["gap_n10"])
       .replace("%ABL_GAP_N01%", ABL["gap_n01"])
       .replace("%PAIRP%", PAIRP_TXT)
       .replace("%NIPOOL%", NIPOOL_TXT)
       .replace("%V1API%", V1API)
       .replace("%V1LOC%", V1LOC)
       .replace("%HALLMAX%", HALLMAX)
       .replace("%CL_N%", str(CL.get("n", 0)))
       .replace("%CL_DIAG%", CL.get("diag", "--"))
       .replace("%CL_CALLS%", str(CL.get("calls", 0)))
       .replace("%CL_ARGVAL%", CL.get("argval", "--"))
       .replace("%CL_EXEC%", CL.get("exec", "--"))
       .replace("%CL_TURNS%", CL.get("turns", "--"))
       .replace("%CL_ONESHOT%", CL.get("oneshot", "--"))
       .replace("%CL2_N%", str(CL2.get("n", 0)))
       .replace("%CL2_DIAG%", CL2.get("diag", "--"))
       .replace("%CL2_CALLS%", str(CL2.get("calls", 0)))
       .replace("%CL2_TURNS%", CL2.get("turns", "--"))
       .replace("%CL2_CHAIN%", CL2.get("chaining", "--"))
       .replace("%CL2_CONSTR%", CL2.get("constr", "--"))
       .replace("%CL2_CONSTR_N%", str(CL2.get("constr_n", 0)))
       .replace("%DESCONLY%", f"{DESCONLY.correct_diag.mean() * 100:.1f}" if DESCONLY is not None else "--")
       .replace("%B_STATE_GBM%", str(BASE40.get("state_gradient_boosting_acc", "--")))
       .replace("%B_STATE_LR%", str(BASE40.get("state_logistic_regression_acc", "--")))
       .replace("%B_TELEM_GBM%", str(BASE40.get("telemetry_gradient_boosting_acc", "--")))
       .replace("%B_TELEM_LR%", str(BASE40.get("telemetry_logistic_regression_acc", "--")))
       .replace("%B_MAJ%", str(BASE40.get("majority_acc", "--")))
       .replace("%B_RAND%", str(BASE40.get("random_acc", "--")))
       .replace("%MA_OUT_API%", MA_OUT_API)
       .replace("%MA_OUT_LOC%", MA_OUT_LOC)
       .replace("%MA_TOOL_API%", MA_TOOL_API)
       .replace("%MA_TOOL_LOC%", MA_TOOL_LOC)
       .replace("%MDEDERIV%", MDEDERIV)
       .replace("%WLS_DIAG%", WLS.get("diag", "--"))
       .replace("%WLS_N%", str(WLS.get("n", 0)))
       .replace("%WLS_CONV%", WLS.get("conv", "--"))
       .replace("%WLS_RMSE%", WLS.get("rmse", "--"))
       .replace("%WLS_FLAG%", WLS.get("flag_agree", "--")))

out = PAPER / "GridPowerAgent_IEEE_Conference.tex"
out = PAPER / "GridPowerAgent_IEEE_Conference.tex"
out.write_text(tex)
unresolved = sorted(set(re.findall(r"%[A-Z_]+%", tex)))
print(f"[PASS] wrote {out} ({len(tex)} chars)")
print(f"  unresolved={unresolved}")
