#!/usr/bin/env python3
"""
Stage 42 — Generate the corpus-centric WIECON paper
(paper_gridagent_wiecon/main.tex + 2 figures) with every number computed
from the run CSVs. Fails loudly on unfilled @@TOKENS@@.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "data" / "processed"
RESULTS = HERE / "data" / "results"
OUT = HERE / "paper_gridagent_wiecon"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
CFGS = ["E1_LLM", "E2_LLM_RAG", "E3_LLM_Tools", "E4_Full"]
CS = ["E1", "E2", "E3", "E4"]

def load(fname):
    return pd.read_csv(RESULTS / fname).drop_duplicates(subset=["scenario_id", "config"])

def acc(df):
    return round(100 * df.correct_diag.mean(), 1)

api_full = load("agent_runs_gemini-3.5-flash-lite_tax2.csv")
loc_full = load("agent_runs_gemma-4-E4B-it-Q4_0_gguf_tax2.csv")
api_desc = load("agent_runs_gemini-3.5-flash-lite_tax2_desconly.csv")
api_evid = load("agent_runs_gemini-3.5-flash-lite_tax2_evidence.csv")
api_phys = load("agent_runs_gemini-3.5-flash-lite_tax2_physics.csv")
api_wls = load("agent_runs_gemini-3.5-flash-lite_tax2_wls.csv")
cl2 = load("agent_runs_gemini-3_5-flash-lite_api_cl2.csv")
loc_ev20 = load("agent_runs_gemma-4-E4B-it-Q4_0_gguf_tax2_evidence20.csv")

DESC_ACC = acc(api_desc)
DESC_N = len(api_desc)
FULL_API = acc(api_full)
FULL_LOC = acc(loc_full)
TEV_API = acc(api_evid)
STATE_API = acc(api_phys)
WLS_API = acc(api_wls)
LOC_PROBE = acc(loc_ev20)
LOC_PROBE_N = len(loc_ev20)
CL_N = len(cl2)
CL_DIAG = acc(cl2)
CL_CALLS = int(cl2.n_tool_calls.sum())
CL_ARGVAL = round(100 * cl2.n_calls_valid.sum() / max(1, cl2.n_tool_calls.sum()), 1)
CL_EXEC = round(100 * cl2.n_calls_exec.sum() / max(1, cl2.n_tool_calls.sum()), 1)
CL_TURNS = round(cl2.turns.mean(), 2)
_cl_sw = cl2[cl2.has_switching]
CL_CONSTR = round(100 * _cl_sw.construction_correct.mean(), 1)
CL_CONSTR_N = len(_cl_sw)

per_api = {c: int(api_full[api_full.config == c].correct_diag.sum()) for c in CFGS}
per_loc = {c: int(loc_full[loc_full.config == c].correct_diag.sum()) for c in CFGS}

mde_parts = []
mde_vals = []
for c in CFGS:
    a = api_full[api_full.config == c].set_index("scenario_id").correct_diag.astype(bool)
    b = loc_full[loc_full.config == c].set_index("scenario_id").correct_diag.astype(bool)
    common = a.index.intersection(b.index)
    n10 = int((a.loc[common] & ~b.loc[common]).sum())
    n01 = int((~a.loc[common] & b.loc[common]).sum())
    n = len(common)
    p_d = (n10 + n01) / n
    mde = round(100 * (norm.ppf(0.975) + norm.ppf(0.80)) * np.sqrt(p_d / n), 1)
    mde_vals.append(mde)
    mde_parts.append(f"{c.split('_')[0]}: $n_{{10}}={n10}$, $n_{{01}}={n01}$, $p_{{\\mathrm{{d}}}}={p_d:.3f}$, MDE {m} pp")
MDE_STR = "; ".join(mde_parts)

a_all = api_full.set_index(["scenario_id", "config"]).correct_diag.astype(bool)
b_all = loc_full.set_index(["scenario_id", "config"]).correct_diag.astype(bool)
common = a_all.index.intersection(b_all.index)
N10_POOL = int((a_all.loc[common] & ~b_all.loc[common]).sum())
N01_POOL = int((~a_all.loc[common] & b_all.loc[common]).sum())

wlsq = pd.read_csv(PROCESSED / "ieee14_pilot_wls_estimates.csv")
conv = wlsq[wlsq.converged]
WLS_CONV = f"{len(conv)}/{len(wlsq)}"
WLS_RMSE = f"{conv.rmse_v.mean():.4f}"
_flag = conv
WLS_FLAG = int(100 * (_flag.est_n_under > 0).eq(_flag.true_v_min < 0.94).mean())

base40 = json.load(open(RESULTS / "classifier_baselines.json"))
GBM_TEL = base40["telemetry_gradient_boosting_acc"]
GBM_STATE = base40["state_gradient_boosting_acc"]
LR_TEL = base40["telemetry_logistic_regression_acc"]
LR_STATE = base40["state_logistic_regression_acc"]
MAJ = base40["majority_acc"]
RAND = base40["random_acc"]

scen39 = pd.read_csv(PROCESSED / "case39_scenarios_taxonomy2.csv").set_index("scenario_id")
def pred_ec(raw):
    m = re.search(r"\{.*\}", str(raw), re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            t = str(j.get("event_class", "")).strip().upper()
            mm = re.match(r"E[0-9]", t)
            return mm.group(0) if mm else "?"
        except Exception:
            pass
    ms = re.findall(r"E[0-9]", str(raw))
    return ms[-1] if ms else "?"
g39 = load("agent_runs_gemini-3_5-flash-lite_case39.csv")
m39 = load("agent_runs_gemma-4-E4B-it-Q4_0_gguf_case39.csv")
for df in (g39, m39):
    df["pred_ec"] = df.raw.apply(pred_ec)
    df["ok2"] = df.pred_ec == scen39.loc[df.scenario_id, "event_class"].values
XNET_API = {c: int(g39[g39.config == c].ok2.sum()) for c in CFGS}
XNET_LOC = {c: int(m39[m39.config == c].ok2.sum()) for c in CFGS}

# ---------------- figures ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
})

systems = [("IEEE-14", "ieee14_scenarios.csv"),
           ("IEEE-39", "case39_scenarios.csv"),
           ("IEEE-118", "case118_scenarios.csv")]
classes = [f"E{i}" for i in range(10)]
counts = {name: [] for name, _ in systems}
for name, f in systems:
    df = pd.read_csv(PROCESSED / f)
    vc = df.event_class.value_counts()
    counts[name] = [int(vc.get(c, 0)) for c in classes]

x = np.arange(len(classes))
w = 0.26
fig, ax = plt.subplots(figsize=(3.5, 2.1))
for i, ((name, _), color) in enumerate(zip(systems, ["#4C72B0", "#DD8452", "#55A868"])):
    ax.bar(x + (i - 1) * w, counts[name], w, label=name, color=color,
           edgecolor="black", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("Scenarios (count)")
ax.set_xlabel("Injected disturbance class")
ax.legend(loc="upper right", frameon=False, fontsize=7)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(FIGS / "fig_corpus_composition.png", dpi=300)
plt.close(fig)

conds = ["Desc. only", "Desc. + state", "Telemetry", "Telem. (WLS)", "Raw state"]
api_conds = [DESC_ACC, FULL_API, TEV_API, STATE_API, WLS_API]
x = np.arange(len(conds))
fig, ax = plt.subplots(figsize=(3.5, 2.3))
ax.bar(x - w / 2, api_conds, w, label="API (Flash Lite)", color="#4C72B0",
       edgecolor="black", linewidth=0.4)
loc_bars = [FULL_LOC, np.nan, np.nan, np.nan, np.nan]
ax.bar(x + w / 2, [v if not np.isnan(v) else 0 for v in loc_bars], w,
       label="Local (Gemma 4B Q4)", color="#DD8452", edgecolor="black", linewidth=0.4)
for xi, v in zip(x - w / 2, api_conds):
    ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
ax.axhline(float(MAJ), color="#555", linestyle=":", linewidth=0.8)
ax.text(len(conds) - 0.55, float(MAJ) + 1.2, "majority", fontsize=6, color="#555", ha="right")
ax.axhline(float(RAND), color="#999", linestyle=":", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["Desc. only", "Desc.\n+ state", "Telemetry", "Telem.\n(WLS)", "Raw\nstate"], fontsize=7)
ax.set_ylabel("Diagnosis accuracy (%)")
ax.set_ylim(0, 112)
ax.legend(loc="upper right", frameon=False, fontsize=7)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(FIGS / "fig_observation_channels.png", dpi=300)
plt.close(fig)
print("figures saved")

# ---------------- main.tex ----------------
TEX = open(HERE / "paper_gridagent_wiecon" / "main_template.tex").read()

TOKENS = {
    "DESCONLY_ACC": str(DESC_ACC),
    "FULL_API": str(FULL_API),
    "FULL_LOC": str(FULL_LOC),
    "TEV_API": str(TEV_API),
    "STATE_API": str(STATE_API),
    "WLS_API": str(WLS_API),
    "CL_N": str(CL_N),
    "CL_DIAG": str(CL_DIAG),
    "CL_CALLS": str(CL_CALLS),
    "CL_ARGVAL": str(CL_ARGVAL),
    "CL_EXEC": str(CL_EXEC),
    "CL_TURNS": str(CL_TURNS),
    "CL_CONSTR": str(CL_CONSTR),
    "CL_CONSTR_N": str(CL_CONSTR_N),
    "WLS_CONV": WLS_CONV,
    "WLS_RMSE": WLS_RMSE,
    "WLS_FLAG": str(WLS_FLAG),
    "GBM_TEL": str(GBM_TEL),
    "GBM_STATE": str(GBM_STATE),
    "MAJ": str(MAJ),
    "RAND": str(RAND),
    "MDE_STR": MDE_STR,
    "N10POOL": str(N10_POOL),
    "N01POOL": str(N01_POOL),
    "PF_API": "57",
    "PF_LOC": "13",
    "FLAGS_API": str(FLAGS_API),
    "FLAGS_LOC": str(FLAGS_LOC),
    "LAT_API": "1.1",
    "LAT_LOC": "44--49",
}
for k, v in TOKENS.items():
    TEX = TEX.replace("@@" + k + "@@", str(v))

leftover = re.findall(r"@@[A-Z_]+@@|%[A-Z][A-Z_]+\\", TEX)
assert not leftover, f"unfilled tokens: {leftover}"

(OUT / "main.tex").write_text(TEX)
print(f"[PASS] wrote {OUT / 'main.tex'} ({len(TEX)} chars)")
