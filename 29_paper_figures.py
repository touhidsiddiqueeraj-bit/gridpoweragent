#!/usr/bin/env python3
"""
Stage 29 — Paper figures, drawn at FINAL RENDERED SIZE.

Every figure is drawn ~3.4 in wide (= \\columnwidth for IEEE two-column), so
LaTeX scales it 1:1 and fonts render at their true size. Minimum effective
font: 6.5 pt; axis labels 7-7.5 pt; titles 8 pt.

Evidence-status conventions:
  - "API"   = Gemini 3.5 Flash Lite, real REST calls, N=140/cfg.
  - "Local" = Gemma 4 E4B Q4_0 via llama.cpp, real inference, N=140/cfg.
  - Mock oracle (N=600) never appears in result figures.
  - No projected/simulated series in result figures.
"""
from pathlib import Path
import ast
import json
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("data/results")
PROCESSED = Path("data/processed")
FIGDIR = Path("paper/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
CFGS = ["E1_LLM", "E2_LLM_RAG", "E3_LLM_Tools", "E4_Full"]
CFG_LABELS = ["E1", "E2", "E3", "E4"]
TOOLS8 = ["power_flow", "state_estimation", "contingency", "n1_security", "opf",
          "grid_query_topology", "grid_query_limits", "grid_query_equipment"]
C_API, C_LOC = "#16a085", "#e67e22"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
})

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

REF = pd.read_csv(PROCESSED / "ieee14_reference_labels.csv").set_index("scenario_id")

def strict_flag(sid, tool):
    row = REF.loc[sid]
    req = [t for t in TOOLS8 if row.get(t) == "required"]
    if tool not in req:
        return False
    if tool == "power_flow" and len(req) > 1:
        return False
    return True

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def load(fname):
    df = pd.read_csv(RESULTS / fname).drop_duplicates(subset=["scenario_id", "config"])
    if "raw" in df.columns:
        df["stated"] = df.raw.apply(lambda r: stated_tool(r))
    else:
        df["stated"] = "power_flow"
    df["strict"] = df.apply(lambda r: strict_flag(r.scenario_id, r.stated), axis=1)
    out = {}
    for c in CFGS:
        sub = df[df.config == c]
        out[c] = {
            "n": len(sub),
            "diag": sub.correct_diag.mean(),
            "strict": sub.strict.mean(),
            "halluc": any_halluc(sub.halluc).mean(),
            "lat": sub.latency.mean(),
            "lat_sd": sub.latency.std(),
        }
    return df, out

API_DF, API = load("agent_runs_gemini-3.5-flash-lite.csv")
LOC_DF, LOC = load("agent_runs_gemma-4-E4B-it-Q4_0_gguf.csv")

def ni_stats():
    rows = []
    rng = np.random.default_rng(0)
    for c in CFGS:
        a = API_DF[API_DF.config == c].set_index("scenario_id").correct_diag.astype(bool)
        b = LOC_DF[LOC_DF.config == c].set_index("scenario_id").correct_diag.astype(bool)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        d = b.values.astype(int) - a.values.astype(int)
        boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(20000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"cfg": c, "diff": 100 * d.mean(), "lo": 100 * lo, "hi": 100 * hi})
    return rows

NI = ni_stats()
SERIES = [
    ("API (Gemini 3.5 FL)", API, C_API, ""),
    ("Local (Gemma 4 E4B Q4_0)", LOC, C_LOC, "//"),
]

def bars_with_ci(ax, key, offset, data, color, hatch):
    vals = [data[c][key] for c in CFGS]
    ns = [data[c]["n"] for c in CFGS]
    ks = [round(v * n) for v, n in zip(vals, ns)]
    errs = np.array([[max(0, v - wilson_ci(k, n)[0]) for v, k, n in zip(vals, ks, ns)],
                     [max(0, wilson_ci(k, n)[1] - v) for v, k, n in zip(vals, ks, ns)]])
    w = 0.36
    x = np.arange(len(CFGS))
    offs = (offset - 0.5) * w
    return ax.bar(x + offs, vals, w * 0.92, color=color, hatch=hatch,
                  edgecolor="black", linewidth=0.5, yerr=errs, capsize=1.5, ecolor="#333")

print("Stage 29 — paper figures (native column width, readable fonts)")

# ---------- Fig: diagnosis (bars + paired-difference forest) ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.1), gridspec_kw={"width_ratios": [1.1, 1]})
for i, (name, data, color, hatch) in enumerate(SERIES):
    bars_with_ci(ax1, "diag", i, data, color, hatch)
import matplotlib.patches as mpatches
ax1.legend(handles=[mpatches.Patch(facecolor=c, edgecolor="black", hatch=h, label=n)
                    for n, _, c, h in SERIES],
           loc="lower right", framealpha=0.9, handlelength=1.2, borderpad=0.3)
ax1.set_xticks(range(len(CFGS)), CFG_LABELS)
ax1.set_ylabel("Diagnosis accuracy")
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", alpha=0.25, lw=0.4)
for i, r in enumerate(NI):
    ax2.errorbar(r["diff"], i, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]],
                 fmt="o", color="#2c3e50", capsize=2, lw=1.0, ms=3.5)
ax2.axvline(0, color="#7f8c8d", lw=0.8)
ax2.set_yticks(range(len(CFGS)), CFG_LABELS)
ax2.invert_yaxis()
ax2.set_xlabel("Local $-$ API (pp), 95% CI")
ax2.set_xlim(-8, 3)
ax2.grid(axis="x", alpha=0.25, lw=0.4)
plt.tight_layout(pad=0.4)
plt.savefig(FIGDIR / "fig_diagnosis.png", dpi=300)
plt.close()
print("  fig_diagnosis.png saved")

# ---------- Fig: strict tool selection ----------
plt.figure(figsize=(3.5, 2.2))
for i, (name, data, color, hatch) in enumerate(SERIES):
    bars_with_ci(plt.gca(), "strict", i, data, color, hatch)
plt.axhline(100 * 0.532, color="#7f8c8d", lw=0.7, ls=":")
plt.text(3.55, 100 * 0.532 + 0.012, "random tool 53%", fontsize=6, color="#7f8c8d", ha="right")
plt.xticks(range(len(CFGS)), CFG_LABELS)
plt.ylabel("Strict-specific tool accuracy")
plt.ylim(0, 0.6)
plt.legend(loc="upper left", handlelength=1.2, borderpad=0.3)
plt.grid(axis="y", alpha=0.25, lw=0.4)
plt.tight_layout(pad=0.4)
plt.savefig(FIGDIR / "fig_tools.png", dpi=300)
plt.close()
print("  fig_tools.png saved")

# ---------- Fig: hallucination + latency (log) ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.0))
x = np.arange(len(CFGS))
w = 0.36
for i, (name, data, color, hatch) in enumerate(SERIES):
    off = (i - 0.5) * w
    ax1.bar(x + off, [data[c]["halluc"] * 100 for c in CFGS], w * 0.92, label=name,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.5)
    ax2.bar(x + off, [data[c]["lat"] for c in CFGS], w * 0.92, label=name,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.5,
            yerr=[data[c]["lat_sd"] for c in CFGS], capsize=1.5, ecolor="#333")
ax1.set_xticks(x, CFG_LABELS)
ax1.set_ylabel("Hallucinated rows (%)")
ax1.set_ylim(0, 2.2)
ax1.set_xlabel("(a) Hallucination (rule-based judge)")
ax2.set_xticks(x, CFG_LABELS)
ax2.set_ylabel("Latency (s, log)")
ax2.set_yscale("log")
ax2.set_ylim(0.5, 200)
ax2.set_xlabel("(b) Latency per call")
ax1.legend(fontsize=6, handlelength=1.2, borderpad=0.25)
ax1.grid(axis="y", alpha=0.25, lw=0.4)
ax2.grid(axis="y", alpha=0.25, lw=0.4, which="both")
plt.tight_layout(pad=0.4)
plt.savefig(FIGDIR / "fig_halluc_latency.png", dpi=300)
plt.close()
print("  fig_halluc_latency.png saved")

# ---------- Fig: architecture (no overlap, column-sized) ----------
fig, ax = plt.subplots(figsize=(3.4, 1.35))
boxes = [
    (0.005, 0.26, 0.155, 0.48, "Grid\nstates", "#eaf2f8"),
    (0.20, 0.26, 0.16, 0.48, "Measure\n+ label", "#eaf2f8"),
    (0.40, 0.56, 0.17, 0.38, "RAG\n8 docs", "#fdebd0"),
    (0.40, 0.04, 0.17, 0.38, "Tools:\nPF N-1 OPF", "#fdebd0"),
    (0.61, 0.26, 0.155, 0.48, "LLM agent\nE1-E4", "#e8f8f5"),
    (0.81, 0.26, 0.185, 0.48, "Advice", "#e8f8f5"),
]
for (bx, by, bw, bh, label, fc) in boxes:
    ax.add_patch(plt.Rectangle((bx, by), bw, bh, fc=fc, ec="#333", lw=0.7))
    ax.text(bx + bw / 2, by + bh / 2, label, ha="center", va="center", fontsize=7)
arrows = [(0.18, 0.225, 0.50), (0.39, 0.435, 0.75), (0.39, 0.435, 0.25),
          (0.605, 0.65, 0.50), (0.605, 0.65, 0.25), (0.805, 0.85, 0.50)]
for x0, x1, y0 in arrows:
    ax.annotate("", xy=(x1, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"))
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0, 1)
ax.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig(FIGDIR / "fig_architecture.png", dpi=300)
plt.close()
print("  fig_architecture.png saved (overlap fixed)")

# ---------- Fig: methodology pipeline (de-staled, column width) ----------
fig, ax = plt.subplots(figsize=(3.4, 2.9))
stages = [
    ("03  Networks & hashes", "IEEE 14/39/118 - hash-pinned"),
    ("04  Operating points", "16k - load 0.70-1.10x"),
    ("05  Scenarios", "15k - classes E0-E9"),
    ("06-07  Meas. + SE", "9.5M - closed-form SE"),
    ("08-09  Severity + labels", "10 tools - tier rules"),
    ("10-14  Physics tools", "PF - N-1 - OPF - query"),
    ("16-17  Knowledge base", "FAISS 384-d - 8 docs"),
    ("19-22  Agents E1-E4", "paired pilot 560+560 calls"),
    ("23-28  Evaluation", "counts - CIs - McNemar - MDE"),
]
n = len(stages)
bh = 0.086
gap = (1.0 - n * bh) / (n - 1)
for i, (title, sub) in enumerate(stages):
    y = 1.0 - (i + 1) * bh - i * gap
    accent = "#eaf2f8" if i < 3 else ("#fdebd0" if i < 6 else "#e8f8f5")
    ax.add_patch(plt.Rectangle((0.13, y), 0.83, bh, fc=accent, ec="#333", lw=0.7))
    ax.text(0.155, y + bh / 2, title, ha="left", va="center", fontsize=7.2, fontweight="bold")
    ax.text(0.945, y + bh / 2, sub, ha="right", va="center", fontsize=6.6, color="#444")
    if i < n - 1:
        ax.annotate("", xy=(0.545, y - gap), xytext=(0.545, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#555"))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig(FIGDIR / "fig_methodology_tree.png", dpi=300)
plt.close()
print("  fig_methodology_tree.png saved (de-staled)")

# remove dead files
for dead in ["fig_tradeoff.png", "fig_scenarios.png", "fig_generalization.png"]:
    f = FIGDIR / dead
    if f.exists():
        f.unlink()
        print(f"  removed dead {dead}")
print("[PASS] Stage 29 complete")
