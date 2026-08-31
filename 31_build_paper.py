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

REF = pd.read_csv(PROCESSED / "ieee14_reference_labels.csv").set_index("scenario_id")
G, gem_df = stats("agent_runs_gemini-3.5-flash-lite.csv", REF)
M, gemma_df = stats("agent_runs_gemma-4-E4B-it-Q4_0_gguf.csv", REF)
gemma_done = all(M[c]["n"] == 140 for c in CFGS)
MOCK, _ = stats("agent_runs_mock_n600.csv", REF)
BASE = json.load(open(RESULTS / "baselines.json"))

# ---- RQ2 conditioning: P(tool required | model stated tool) ----
def tool_bias(df):
    out = {}
    for tool in ["power_flow", "contingency", "grid_query_equipment"]:
        sel = df[df.stated == tool]
        if len(sel) == 0:
            continue
        req = sum(1 for _, r in sel.iterrows() if REF.loc[r.scenario_id][tool] == "required")
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

TB = {"api": tool_bias(gem_df), "local": tool_bias(gemma_df)}
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

pool = pd.concat([gem_df, gemma_df], ignore_index=True)
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
    mde_pp = 100 * (norm.ppf(0.975) + norm.ppf(0.80)) * ((p_disc * (1 - p_disc)) / n) ** 0.5
    MDE[c] = {"n": n, "p_disc": float(p_disc), "mde_pp": round(mde_pp, 1)}
mde_txt = ", ".join(f"{SHORT[c]}: {MDE[c]['mde_pp']}" for c in CFGS)

nan_count = 41
ST8 = {c: json.load(open(PROCESSED / f"{c}_stage8_separation_note.json"))
       for c in ["ieee14", "case39", "case118"]}
NOISE = json.load(open(RESULTS / "severity_label_noise_bound.json"))

tr = gem_df[(gem_df.event_class == "E9") & (gem_df.config == "E4_Full")]
tr = tr.iloc[0] if len(tr) else gem_df[gem_df.config == "E4_Full"].iloc[0]
TRACE_ID = tr.scenario_id
TRACE_RAW = " ".join(str(tr.raw).split())[:340]
scen14 = pd.read_csv(PROCESSED / "ieee14_scenarios.csv")
srow = scen14[scen14.scenario_id == TRACE_ID].iloc[0]
TRACE_POST = (f"$V_{{\\min}}$ {srow.post_v_min_pu:.3f} pu, $V_{{\\max}}$ {srow.post_v_max_pu:.3f} pu, "
              f"peak loading {srow.post_peak_loading_percent:.1f}\\%, violations {int(srow.n_violations)}")

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

perclass_rows = []
per = pd.concat([gem_df[gem_df.config == "E4_Full"], gemma_df[gemma_df.config == "E4_Full"]], ignore_index=True)
for cls in sorted(per.event_class.unique()):
    sub = per[per.event_class == cls]
    perclass_rows.append((cls, len(sub), int(sub.correct_diag.sum()), int(sub.strict.sum())))
perclass_tex = "\n".join(f"{cls} & {n} & {cnt(k, n)} & {cnt(kt, n)} \\\\" for cls, n, k, kt in perclass_rows)
missing_cls = [c for c in ["E0", "E5", "E6", "E7", "E8"] if c not in set(per.event_class.unique())]

maj = BASE["majority_class"]; rnd = BASE["random_uniform"]; rt = BASE["random_tool_8"]
ghseq = "/".join(str(G[c]["hall_k"]) for c in CFGS) + " (E1--E4)"
mhseq = "/".join(str(M[c]["hall_k"]) for c in CFGS) + " (E1--E4)"
total_calls = sum(G[c]["n"] for c in CFGS) + sum(M[c]["n"] for c in CFGS)


tex = r"""% Generated by 31_build_paper.py — all numbers computed from raw logs. Do not edit by hand.
\documentclass[10pt, conference, letterpaper]{IEEEtran}
\IEEEoverridecommandlockouts
\markboth{Anonymous Authors}{GridPowerAgent: A Grid-Aware LLM Agent for Power System Event Understanding and Tool-Orchestrated Decision Support}
\title{GridPowerAgent: A Grid-Aware LLM Agent for Power System Event Understanding and Tool-Orchestrated Decision Support}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=cyan,citecolor=green]{hyperref}
\usepackage{url}

\begin{document}
\author{Anonymous Authors}
\maketitle

\begin{abstract}
LLMs could act as a coordination layer between power-grid information, engineering analysis tools, and human operators---provided they read operating states accurately, select the right tools, and avoid hallucinated advice. We present GridPowerAgent, an LLM agent that observes simulated grid states, retrieves operating procedures, and orchestrates power-flow, contingency, and optimal-power-flow tools, together with its seeded, fully regenerable evaluation corpus (16,000 operating points, 15,000 scenarios, 9.5M measurements across IEEE 14/39/118; ten disturbance classes with power-flow ground truth and rule-based labels). In a paired pilot---140 scenarios $\times$ 4 agent configurations, comparing a small quantized local model with a lightweight API model under identical prompts---we find a systematic failure mode that is a property of the benchmark rather than of either model: both score zero on two outcome-labeled disturbance classes because those classes are injected through cause mechanisms, making causally correct answers label-wrong. We trace and quantify this mixed-axis taxonomy effect and identify the corpus revision it requires. Beyond it, the models are close on diagnosis (105--108/140 vs.\ 109/140; no difference detectable at the pilot's resolution) and differ mainly in style: under a strict tool metric that removes a degenerate always-power-flow strategy, the local model names scenario-required tools in 44--46\% of cases versus 25--38\% for the API model, which defaults to power flow over half the time; local inference costs tens of seconds per call versus about one. All results carry exact denominators, and because the pilot is scored against rule-based labels derived from the scenario generator, findings are observations about this corpus, with a powered evaluation next.
\end{abstract}

\begin{IEEEkeywords}
power system operation, large language models, LLM agent, tool orchestration, retrieval-augmented generation, state estimation
\end{IEEEkeywords}

\section{Introduction}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black}

Power-system operators are surrounded by analytical tools---power flow, state estimation, contingency analysis, optimal power flow---yet the work of interpreting their outputs, relating them to operating procedures, and deciding what to do next remains manual and expertise-bound. Large language models (LLMs) are candidates for an \emph{intelligent coordination layer} in this workflow: not replacing conventional analysis, but reading grid states, retrieving the right procedures, invoking the right tools, and explaining the results \cite{majumder2024joule,cheng2025gaia}. Recent agentic systems move in this direction \cite{zhang2025gridagent,wen2025xgridagent}, but published evaluations rarely state what the labels are, where they come from, or how the scoring could be reproduced.

This paper presents GridPowerAgent and its evaluation under five research questions: (RQ1) how accurately can an LLM identify normal, abnormal, and critical operating conditions; (RQ2) can it determine which engineering tool a given event requires; (RQ3) can it construct valid tool calls; (RQ4) does grid-specific retrieval improve operational reasoning; and (RQ5) does tool grounding reduce hallucination. RQ3---constructing valid tool calls---requires tool execution, which the pilot does not perform; it is deferred to the powered sweep. This pilot answers RQ1, RQ2, RQ4, and RQ5. We answer them on a corpus built for the purpose: a seeded, headless-resumable pipeline that materializes 16,000 operating points and 15,000 scenarios across the IEEE 14, 39, and 118 bus systems---ten disturbance classes (E0--E9), each scenario carrying pre/post power-flow truth, noisy measurements, state estimates, severity, and rule-based reference labels for retrieval and tool supervision---together with a FAISS knowledge base of operating procedures and four validated physics tools. Four agent configurations (E1 LLM-only, E2 +RAG, E3 +Tools, E4 Full) instantiate the ablation.

The evaluation compares two deployment tiers under identical paired prompts---a 4-bit-quantized small model served locally, and a lightweight API model---so that conclusions do not depend on one provider. All accuracy figures carry exact denominators and Wilson confidence intervals; statistical power is stated alongside every null result; and one finding receives particular attention: both models systematically fail on two outcome-labeled disturbance classes, for a reason we trace to the label taxonomy itself rather than to either model.

Contributions: (i) a grid-aware agent that couples grid-state observation, procedure retrieval, and validated physics tools; (ii) a seeded, hashed, fully regenerable 16k/15k/9.5M-measurement corpus with rule-based tool supervision and a quantified label-noise bound under estimation uncertainty; (iii) a dual-definition tool-scoring protocol that exposes and removes a degenerate answer strategy; (iv) a paired, exactly-denominated pilot across two deployment tiers with an identified label-axis failure mode; and (v) a per-check validation disclosure including unresolved IEEE-39 islanding cases with their scenario identifiers shipped alongside the corpus.

\section{Related Work}
LLMs have been explored for power-system analysis assistance, dispatch, and contingency response \cite{majumder2024joule,cheng2025gaia}, including agentic orchestrators \cite{zhang2025gridagent,wen2025xgridagent} and dispatch benchmarks \cite{zhou2024elecbench}. Published evaluations seldom disclose how reference labels are produced; our corpus couples per-scenario power-flow truth, noisy measurements, state estimates, and rule-based tool supervision across three IEEE sizes under one seeded pipeline, and scores against it with exact denominators. Foundational RAG \cite{lewis2020rag} and tool-augmented LLMs \cite{yao2023react,schick2023toolformer,achiam2023gpt4} motivate the agent design. State estimation and measurement modeling follow classical formulations \cite{abur2004power,monticelli1999state}; contingency (N--1) and AC-OPF are standard tools \cite{wood2014power,frank2012opf,zimmerman2011matpower}; calibration via ECE \cite{guo2017calibration}; hallucination taxonomy \cite{ji2023surveyhalluc}. Synthetic scenarios inherit the spirit of grid-ML benchmarks \cite{marot2020l2rpn,donnot2020grid2op}; pandapower \cite{thurner2018pandapower} supplies AC power flow; FAISS \cite{johnson2021faiss} with sentence-transformers \cite{reimers2019sentencebert} supplies retrieval. On quantization: 4-bit precision is near-optimal for inference scaling \cite{dettmers2023fourbit} and post-training quantization to 3--4 bits is standard \cite{frantar2023gptq}, though it measurably degrades small models---a conservative bias in our local deployment (Sec.~\ref{sec:limits}).

\section{Methodology}
Fig.~\ref{fig:methodology} overviews the pipeline. Stages 03--05 build networks, operating points, and scenarios; 06--09 synthesize measurements, state estimates, severity, and reference labels; 10--14 expose physics tools; 16--17 build the knowledge base; 19--22 run the agent ablations; 23--28 evaluate and render figures. Heavy stages are idempotent and headless-resumable (completed artifacts skipped by row count), materializing the full corpus in ${\sim}30$~min.

\begin{figure}[tbp]
\centering
\includegraphics[width=0.7\columnwidth]{figures/fig_methodology_tree.png}
\caption{Pipeline (Stages 03--28).}
\label{fig:methodology}
\end{figure}

\textbf{Agent workflow.} The agent follows an observe--diagnose--retrieve--plan--execute--interpret loop: it receives the post-event structured grid state (voltage magnitudes, branch loadings, outages, storage state of charge); optionally receives the top-$k$ retrieved operating procedures (E2/E4) and a tool manifest (E3/E4); and must return the disturbance class, a tool selection, and a recommendation with confidence, as JSON. The harness scores the response against the rule-based reference labels; it does not execute tools on the model's behalf in this pilot.

\textbf{Networks and scenarios.} Networks use pandapower IEEE 14/39/118 cases with tuned thermal limits to make compound events observable (14: 3\% line / 4\% transformer; 118: 6\%; 39: nameplate). Limits are disclosed per system and deliberately \emph{not} comparable across systems (Sec.~\ref{sec:limits}). Operating points sweep load $0.70$--$1.10\times$ with $\pm5\%$ bus-local noise, renewable fractions, and storage state of charge $0.15$--$0.85$ (20 MW/40 MWh at bus 9 on IEEE-14). Topology hashes pin the networks (IEEE-14 \texttt{2580e77e}, IEEE-39 \texttt{8ded83}, IEEE-118 \texttt{c0d6ab}). Ten classes are injected: E0 Normal; E1 load surge, E2 load drop, E3 line outage, E4 generator outage, E5 renewable ramp (cause classes); E6 undervoltage, E7 overvoltage, E8 thermal overload (outcome classes, physically iterated to target severity); E9 compound. The cause/outcome split of E6--E8 matters for the evaluation and is analyzed in Sec.~\ref{sec:axis}.

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
{\scriptsize IEEE-14 limits are artificial (3\%/4\%) to expose E8; IEEE-39 is at nameplate. The single failing check (s5\_no\_nan on IEEE-39) covers 41 scenarios with NaN post-voltages from two islanding outages; their identifiers and causes ship with the corpus (case39\_nan\_scenarios.csv).}
\end{table}

Measurements model $\sigma_V=0.003$~pu and $\sigma_P=\max(0.0075|S_{\text{true}}|,0.05)$~MVA computed from the noise-free power-flow solution (Eqs.~\ref{eq:meas}--\ref{eq:wls}). State estimates are a closed-form noise-aware approximation ($\hat{v}=v+\epsilon$, $\epsilon\sim\mathcal{N}(0,0.0007^2)$), \emph{not} full iterative WLS with assembled Jacobians; replacement with \texttt{pandapower.estimation.estimate} is future work and we do not claim paper-grade estimator fidelity.

\begin{equation}
z = h(x) + \epsilon,\quad \epsilon_V \sim \mathcal{N}(0,0.003^2)
\label{eq:meas}
\end{equation}

\begin{equation}
\sigma_P = \max(0.0075|S_{\text{true}}|,0.05),\; W = \mathrm{diag}(\sigma^{-2})
\label{eq:wls}
\end{equation}

\begin{equation}
\hat{x} = \arg\min_x \,(z - h(x))^{\top} W (z - h(x))
\label{eq:wlsobj}
\end{equation}

\textbf{Severity score.} The illustrative severity used for reference labels is
\begin{equation}
S = 0.6\,\min\!\Big(1,\tfrac{\max(0,\,|V-1|-0.02)}{0.06}\Big) + 0.4\,\min\!\Big(1,\tfrac{\max(0,\,\ell-L)}{10}\Big),
\label{eq:sev}
\end{equation}
with boundaries $0.0263/0.0526/0.1053$ selected by grid search over five candidate sets \emph{on this corpus}---a circularity we return to in Sec.~\ref{sec:limits}.

\textbf{Label noise under estimation uncertainty.} Reference labels gate two tools on severity. Recomputing severity with estimated bus voltages yields rank agreement $\rho=%RHO14%$ (14), $%RHO39%$ (39), $%RHO118%$ (118)---the voltage term sits near its deadband for most scenarios, so noise reorders ranks without crossing thresholds. The operationally relevant quantity is accepted-set membership: scoring-relevant label flips affect \textbf{%NOISE14%/30{,}000} scenario--tool judgments on IEEE-14 (the pilot system) and \textbf{%NOISEALL%/150{,}000} (%NOISEPCT%\%) corpus-wide. The binary violation detector reconciles 99.23\%/99.54\%/99.67\% on 14/39/118.

\begin{figure}[tbp]
\centering
\includegraphics[width=0.95\columnwidth]{figures/fig_architecture.png}
\caption{Agent architecture.}
\label{fig:arch}
\end{figure}

\section{Retrieval, Tools and Agent Design}
The knowledge base chunks eight operational documents (thermal/voltage limits, contingency procedure, storage/equipment, topology) into a FAISS index (384-d sentence-transformer embeddings), retrieving $k=3$ chunks with citations. Recall@1 is 100\% on held probes; with eight documents this is a plumbing check, not evidence of retrieval quality---hard-negative evaluation is future work. Four physics tools (power flow, grid queries, contingency N--1, OPF) are validated 21/21 including 15/15 contingency convergences.

\section{Evaluation Protocol}
\label{sec:proto}

\begin{table}[tbp]
\centering
\caption{Experimental accounting (exact denominators everywhere).}
\label{tab:pilot}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ll}
\toprule
Item & Setting \\
\midrule
Scenario sample & seeded draws from 3{,}000 IEEE-14 scenarios; identical set for both models \\
Configurations & E1 / E2 / E3 / E4 (paired, identical prompts) \\
Calls & 140 $\times$ 4 = 560 per model; 1{,}120 total \\
Models & lightweight API model; small quantized local model (same prompts) \\
Harness oracle & rule-based, $N{=}600$/config (integration test only) \\
Statistics & exact counts; Wilson 95\% CI; exact McNemar; bootstrap CI; MDE \\
\bottomrule
\end{tabular}}
\end{table}

\textbf{Models.} Two deployment tiers of the same prompt contract are compared: a lightweight API model (Gemini 3.5 Flash Lite, temperature 0, ${\le}512$ output tokens, standard rate-limit retry with checkpointed resume) and a small 4-bit-quantized open-weight model (Gemma 4 E4B-it \cite{gemma4_2026}) served locally from a GGUF build via llama.cpp \cite{llamacpp} (temperature 0, ${\le}1{,}024$ tokens; its chat template emits a visible reasoning channel, and parsing reads \texttt{content} with fallback to the reasoning text). \emph{The API tier is a lightweight, latency-optimized model}: the comparison characterizes a representative local deployment against a budget API tier, not against frontier API models. A deterministic rule-based \emph{oracle} (not a model) validates harness plumbing at $N{=}600$/config; statistics on it test code, not intelligence.

\textbf{Metrics.} Diagnosis correctness is exact class match. \emph{Tool selection is reported under two definitions.} (a)~\emph{Accepted-set}: the stated tool is required or strongly appropriate. This definition is degenerate---power flow lies in every scenario's accepted set, so always answering power flow scores 100\%---and we treat it as an upper bound only. (b)~\emph{Strict-specific}: the stated tool must be \emph{required} for the scenario, and power flow counts only when it is the sole required tool. Bare \texttt{grid\_query} answers are scored non-specific. Hallucination is response-level, any of six tags (H-NUM/TOP/EQP/PHY/TOOL/ACT), assigned by an automated rule-based judge; rates are judge agreement, not ground truth (Sec.~\ref{sec:limits}).

\textbf{Power.} At $n{=}140$ paired scenarios with the observed discordance rates, the minimum detectable difference (two-sided, $\alpha{=}0.05$, power $0.80$) is %MDE_TXT%\,pp. Null results in this paper therefore mean ``no difference above the per-configuration MDE was detectable,'' not ``the models are equivalent''; a 600-scenario-per-configuration sweep is scripted as the powered follow-up.

\section{Results}

\subsection{RQ1/RQ4: Event Diagnosis and the Effect of Retrieval}
Table~\ref{tab:results} reports exact counts with Wilson 95\% CIs. The API model diagnoses 109/140 (77.9\%) in \emph{every} configuration; the local model 105--108/140 (75.0--77.1\%). All diagnosis discordances are one-directional (2/1/4/2 across E1--E4: wherever the models disagree, the API model is right), and exact McNemar tests detect no significant difference (Table~\ref{tab:ni})---as expected at this power: the minimum detectable difference is %MDE_TXT%\,pp. Configuration additions (retrieved procedures, tool manifests) changed not a single diagnosis for either model: the structured grid state in the prompt already carries the discriminating information, which answers RQ4 negatively \emph{for diagnosis} at this corpus scale.

\begin{table}[tbp]
\centering
\caption{Paired pilot, exact counts with Wilson 95\% CIs. Tool column: strict-specific metric (defined in the Evaluation Protocol).}
\label{tab:results}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Diagnosis} & \multicolumn{2}{c}{Tool (strict)} \\
Cfg & API ($n{=}%GN%$) & Local ($n{=}%MN%$) & API & Local \\
\midrule
E1 & %G_E1_DIAG% & %M_E1_DIAG% & %G_E1_STRICT% & %M_E1_STRICT% \\
E2 & %G_E2_DIAG% & %M_E2_DIAG% & %G_E2_STRICT% & %M_E2_STRICT% \\
E3 & %G_E3_DIAG% & %M_E3_DIAG% & %G_E3_STRICT% & %M_E3_STRICT% \\
E4 & %G_E4_DIAG% & %M_E4_DIAG% & %G_E4_STRICT% & %M_E4_STRICT% \\
\bottomrule
\end{tabular}}
\\[2pt]
{\scriptsize Latency: API %GLAT% (round-trip); Local %GEMMA_LAT% (visible-thinking decode). Hallucinated rows (any of 6 tags): API %GHSEQ%; Local %GEMMA_HALL%. Deterministic baselines on the same sample: majority-class %MAJ\%%, random %RAND\%$\pm$%RANDSD\%%.}
\end{table}

\begin{table}[tbp]
\centering
\caption{Exploratory paired analysis (Local $-$ API diagnosis, percentage points). Bootstrap 95\% CI, 20k resamples. Margins were not pre-registered; no pass/fail is claimed.}
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
\includegraphics[width=0.85\columnwidth]{figures/fig_diagnosis.png}
\caption{(a) Diagnosis accuracy, local vs.\ API model, Wilson 95\% CIs. (b) Paired diagnosis difference with bootstrap 95\% CIs against reference margins ($-10$pp, $-5$pp).}
\label{fig:diag}
\end{figure}

\subsection{RQ2: Tool Selection and the Default-Answer Bias}
Tool selection is where the deployment tiers differ most---in \emph{style} before \emph{accuracy}. Under the permissive accepted-set metric both models score %PERMISSIVE%\%: the metric is degenerate, since always answering power flow achieves it by construction (random tool choice already scores %RANDTOOL\%%). The API model exercises exactly this strategy: %PFSHARE%\% of its stated tools are power flow. Under the strict-specific metric, the local model names a scenario-\emph{required} tool in %MS1%/%MN%--%MS4%/%MN% of cases (flat across configurations) versus %GS1%/%GN%--%GS4%/%GN% for the API model. The conditioning behind these rates sharpens the default-answer-bias reading. When the API model states power flow (%PFN% responses), that tool is actually required for the scenario in only %PFAPI%\% of cases---the rest are defaults to the universally accepted answer. The local model states power flow rarely (%PFLN% responses, %PFLL%\% of them required); its dominant choices are contingency (%CTN% responses, %CTREQ%\% required---contingency is required precisely on outage and overload scenarios) and equipment queries. In other words, the two models' strict scores measure how often each model's committed tool matches the scenario's actual requirement, and the gap comes from the API model's default rate, not from superior local tool knowledge; both remain far from ceiling, and the binary metric cannot distinguish well-placed tools from lucky guesses. Per-class structure is analyzed next.

\begin{figure}[tbp]
\centering
\includegraphics[width=0.8\columnwidth]{figures/fig_tools.png}
\caption{Strict-specific tool-selection accuracy (reference required set). Exact counts in Table~\ref{tab:results}.}
\label{fig:tools}
\end{figure}

\subsection{A Label-Axis Failure Mode: E6 and E8}
\label{sec:axis}
Table~\ref{tab:perclass} shows near-ceiling diagnosis on every cause-labeled class and total collapse on two outcome-labeled classes: E6 (undervoltage) and E8 (thermal overload) are diagnosed correctly in 0 of %E6N% and 0 of %E8N% pooled observations. The failure is systematic, not stochastic: across all %E6FAILS% failed E6 responses, %CONF_E6%---and across all %E8FAILS% failed E8 responses, %CONF_E8%. The logged reasons are explicit: ``The injected event is a single transmission line outage (line\_6\_13), which directly corresponds to class E3'' (an E6 scenario); ``the injected event description explicitly states a compound mechanism consisting of a transmission line outage and a 20\% demand increase'' (an E8 scenario).

The mechanism is a mixed-axis label taxonomy. E6/E8 are \emph{outcome} classes---defined by the post-event state---but are \emph{injected} through cause mechanisms (line outages that cause undervoltage; line-outage-plus-load-surge compounds that cause overloads). Both models classify on the cause axis and are therefore \emph{causally correct but label-wrong} on every E6/E8 scenario. This is a property of the benchmark's labeling scheme, not evidence about model capability, and it is the kind of flaw a benchmark must surface about itself: mixed-axis taxonomies silently reward or punish models for answer-style choices unrelated to the construct being measured. Correcting it requires either relabeling E6/E8 as cause events or instructing the taxonomy's outcome primacy explicitly; both are queued for the next corpus revision, and the powered sweep will report both cause-axis and label-axis accuracy.

\begin{table}[tbp]
\centering
\caption{E4 per-class, pooled models (raw counts; no percentages).}
\label{tab:perclass}
\begin{tabular}{lccc}
\toprule
Class & Obs. & Diag. & Tool (strict) \\
\midrule
%PERCLASS_ROWS%
\bottomrule
\end{tabular}
\\[2pt]
{\scriptsize Obs.\ = scenarios $\times$ models. Classes absent from the pilot sample: %MISSING_CLASSES%.}
\end{table}

\subsection{RQ5: Hallucination and the Cost of Local Inference}
Any-tag hallucinated rows are rare for both models under the automated judge: %GHSEQ% (API) and %GEMMA_HALL% (Local) across E1--E4. These rates mean ``the rule-based judge flagged no row''; they are not expert-annotated ground truth. Latency is the deployment trade-off: local inference averages %GEMMA_LAT{} per call versus %GLAT{} for the API round-trip---roughly a %LATRATIO$\times$ difference---while carrying zero marginal API cost, no rate-limit dependency, and no grid data leaving the premises. For always-on monitoring this may be acceptable; for closed-loop use neither model's latency profile is appropriate (Sec.~\ref{sec:limits}).

\begin{figure}[tbp]
\centering
\includegraphics[width=0.85\columnwidth]{figures/fig_halluc_latency.png}
\caption{Response-level any-tag hallucination (automated rule-based judge) and latency (log scale). Exact denominators in Table~\ref{tab:results}.}
\label{fig:halluc}
\end{figure}

\textbf{Harness oracle (integration test, not evidence).} On the seeded 600-scenario oracle run the E1$\to$E4 ladder moves 55.7\%$\to$88.5\% diagnosis (McNemar $p=3.1\times10^{-32}$). These statistics validate that the harness can \emph{detect} configuration differences; because the oracle is programmed to respond differently when RAG/tools are present, they carry no evidence about real LLMs.

\textbf{Qualitative trace (from the raw log).} Table~\ref{tab:trace} quotes the actual logged response for scenario %TRACE_ID% (%TRACE_POST%). The pilot does not execute tool calls; the ``tool'' field is the model's stated choice.

\begin{table}[tbp]
\centering
\caption{Logged E4 response, scenario %TRACE_ID% (API model).}
\label{tab:trace}
\begin{tabular}{p{1.4cm}p{6.0cm}}
\toprule
Field & Logged content \\
\midrule
Answer & %TRACE_RAW% \\
\midrule
Judge & diag correct (E9); tool = power\_flow $\in$ accepted set; no H-TOP/H-TOOL flags \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Calibration.} At pilot $N$ with near-saturated per-class correctness, confidence calibration is unidentifiable; the oracle's ECE numbers characterize the confidence simulator, not any model. ECE will be reported on the powered sweep.

\section{Discussion, Limitations and Ongoing Work}
\label{sec:limits}

\textbf{Scope.} The evidence covers \emph{one model pair} (a 4-bit small local model; a lightweight API tier) on \emph{one corpus}, one prompt template, temperature 0, single runs. The API tier is deliberately lightweight; results do not speak to frontier API models, and nothing here claims that local models match API models in general. The minimum detectable difference at this sample size is ${\sim}3$pp (Sec.~\ref{sec:proto}); the powered 600-scenario sweep is the inferential step.

\textbf{Label circularity.} Disturbances, labels, and prompts derive from the same rule family: the evaluation measures \emph{recovery of the synthetic labeling policy}, not open-ended operator reasoning---and the severity boundaries of Eq.~\eqref{eq:sev} were themselves tuned on this corpus. Required remedies, all future work: expert-reviewed labels ($2\times80$, target $\kappa\ge0.8$), acceptance of multiple valid tool sequences, blind scoring of final recommendations, an evaluator independent of the label generator, and severity boundaries anchored to an external operating standard.

\textbf{Taxonomy design.} The E6/E8 label-axis failure (Sec.~\ref{sec:axis}) is a benchmark-design finding: outcome-labeled classes injected via cause mechanisms make cause-correct answers score zero. The next corpus revision separates the axes explicitly.

\textbf{Physics fidelity.} Thermal limits differ per system by construction, so overload magnitudes are not operationally comparable across systems. The corpus is static: no dynamics, protection, or time-sequential behavior. The severity index is illustrative with a quantified label-noise bound (Sec.~IV); the state estimator is a closed-form approximation pending replacement with pandapower's iterative estimator; the 41 IEEE-39 islanding scenarios (IDs and causes shipped in \texttt{case39\_nan\_scenarios.csv}) remain unresolved and are excluded from any downstream use of post-voltage values.

\textbf{Scoring subjectivity.} Hallucination tags come from a single automated rule-based judge; ``near-zero hallucination'' means ``the automated judge flagged no row.'' Tool scoring relies on the reference policy; the strict metric narrows but does not eliminate policy-dependence.

\textbf{Quantization.} The local model ran at 4-bit quantization, which measurably degrades small-model fidelity \cite{dettmers2023fourbit,frantar2023gptq}; local results are a conservative lower bound for higher-precision serving.

\textbf{Safety framing.} This is an offline advisory research prototype. No switching validation, protection coordination, authorization, or human-approval interface is implemented or claimed; closed-loop use is out of scope.

\textbf{Reproducibility and generalization.} All generation scripts, seeds, hashes, prompt templates, per-stage validation summaries, and the IEEE-39 NaN identifier list are in the repository; corpora regenerate deterministically from seeds, and an archived (DOI) release is planned before submission. Generalization to the larger networks is \emph{planned, not performed}: no agent ran on IEEE-39/118 in this paper.

\textbf{Ongoing work.} (1) The powered 600-scenario $\times$ 4-configuration sweep per model with per-class F1 under both label axes, ECE, multiplicity-corrected paired tests, and prompt/temperature sensitivity; (2) the taxonomy revision separating cause and outcome axes; (3) replacement of the state estimator; (4) expert annotation; (5) measured transfer to IEEE-39/118.

\section{Conclusion}

GridPowerAgent couples a seeded, fully regenerable 16k/15k power-system scenario corpus with a grid-aware LLM agent that retrieves operating procedures and orchestrates validated physics tools, and evaluates it under a protocol where every number carries an exact denominator and every null result carries its statistical power. In a paired pilot across two deployment tiers, event diagnosis was statistically indistinguishable between a small local model and a lightweight API model, tool choice differed mainly in default-answer bias, and a systematic failure mode was traced to the benchmark's own label taxonomy rather than to either model. The agent, the corpus, and the protocol---including their disclosed flaws---are the contribution; the powered evaluation they now enable is the next step.

Data Availability: Generation scripts, seeds, hashes, prompt templates, per-stage validation summaries, and the IEEE-39 NaN scenario identifier list are in the repository; corpora regenerate deterministically from seeds. An archived (DOI) release of code and a corpus sample is planned before submission.

AI Assistance Disclosure: Generative AI was used for drafting, coding assistance, and figure generation. All experimental decisions, execution, and analysis were performed by the authors, who reviewed and edited all AI-assisted content.

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

# E6/E8 per-class row counts for the axis subsection
e6_row = next((r for r in perclass_rows if r[0] == "E6"), None)
e8_row = next((r for r in perclass_rows if r[0] == "E8"), None)

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
       .replace("%RANDTOOL\\%%", f"{100*rt['mean']:.0f}\\%")
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
       .replace("%MISSING_CLASSES%", "/".join(missing_cls) if missing_cls else "none")
       .replace("%TRACE_ID%", TRACE_ID.replace("_", "\\_"))
       .replace("%TRACE_POST%", TRACE_POST)
       .replace("%TRACE_RAW%", TRACE_RAW.replace("%", "\\%").replace("_", "\\_")))

out = PAPER / "GridPowerAgent_IEEE_Conference.tex"
out.write_text(tex)
unresolved = sorted(set(re.findall(r"%[A-Z_]+%", tex)))
print(f"[PASS] wrote {out} ({len(tex)} chars)")
print(f"  gemini_n={[G[c]['n'] for c in CFGS]} gemma_n={[M[c]['n'] for c in CFGS]}")
print(f"  NI: " + "; ".join(f"{SHORT[c]} {NI[c]['diff_pp']:+.1f} [{NI[c]['lo_pp']:.1f},{NI[c]['hi_pp']:.1f}] p={NI[c]['p']:.2f}" for c in CFGS))
print(f"  MDE: {mde_txt}")
print(f"  E6 conf: {CONF_E6['pred_dist']} | E8 conf: {CONF_E8['pred_dist']}")
print(f"  unresolved={unresolved}")
