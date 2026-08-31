#!/usr/bin/env python3
"""
Stage 29 — Paper figures: local-vs-API benchmark, evidentiary status baked in.

Conventions:
  - "API"    = Gemini 3.5 Flash Lite, real REST calls, N=140/cfg.
  - "Local"  = Gemma 4 E4B Q4_0 via llama.cpp, real inference, N=140/cfg.
  - Mock oracle (N=600) never appears in result figures (integration test only).
  - Generalization figure marked PROJECTED + IEEE-39 islanding caveat.
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
CFG_LABELS = ["E1\nLLM", "E2\n+RAG", "E3\n+Tools", "E4\nFull"]
TOOLS8 = ["power_flow", "state_estimation", "contingency", "n1_security", "opf",
          "grid_query_topology", "grid_query_limits", "grid_query_equipment"]
C_API, C_LOC = "#16a085", "#e67e22"

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
            "diag_k": int(sub.correct_diag.sum()),
            "strict": sub.strict.mean(),
            "strict_k": int(sub.strict.sum()),
            "halluc": any_halluc(sub.halluc).mean(),
            "hall_k": int(any_halluc(sub.halluc).sum()),
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

print("Stage 29 — paper figures (local vs API)")

# ---------- Figure: diagnosis (paired bars) + NI forest ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(CFGS))
w = 0.36
series = [("API model (Gemini 3.5 FL)", API, C_API, ""),
          ("Local model (Gemma 4 E4B Q4_0)", LOC, C_LOC, "//")]
for i, (name, data, color, hatch) in enumerate(series):
    vals = [data[c]["diag"] for c in CFGS]
    ns = [data[c]["n"] for c in CFGS]
    errs = np.array([[max(0, v - wilson_ci(round(v * n), n)[0]) for v, n in zip(vals, ns)],
                     [max(0, wilson_ci(round(v * n), n)[1] - v) for v, n in zip(vals, ns)]])
    offs = (i - 0.5) * w
    bars = ax1.bar(x + offs, vals, w * 0.92, label=name, color=color, hatch=hatch,
                   edgecolor="black", linewidth=0.6, yerr=errs, capsize=2, ecolor="#333")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, 0.02, f"{v*100:.0f}",
                 ha="center", va="bottom", fontsize=7, rotation=90,
                 color="white" if hatch == "" else "black", fontweight="bold")
ax1.set_xticks(x, ["E1", "E2", "E3", "E4"])
ax1.set_ylabel("Diagnosis accuracy")
ax1.set_ylim(0, 1.0)
ax1.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax1.grid(axis="y", alpha=0.25, lw=0.4)
ax1.set_title("(a) Diagnosis, N=140/cfg/model", fontsize=8)

for i, r in enumerate(NI):
    ax2.errorbar(r["diff"], i, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]],
                 fmt="o", color="#2c3e50", capsize=3, lw=1.2, ms=4)
ax2.axvline(0, color="#7f8c8d", lw=0.8)
ax2.set_yticks(range(len(CFGS)), ["E1", "E2", "E3", "E4"])
ax2.invert_yaxis()
ax2.set_xlabel("Paired diff, Local − API (pp)")
ax2.set_xlim(-12, 3)
ax2.grid(axis="x", alpha=0.25, lw=0.4)
ax2.set_title("(b) Paired difference, Local − API (bootstrap 95% CI)", fontsize=8)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_diagnosis.png", dpi=200)
plt.close()
print("  fig_diagnosis.png saved")

# ---------- Figure: strict tool selection ----------
plt.figure(figsize=(6.4, 3.0))
for i, (name, data, color, hatch) in enumerate(series):
    vals = [data[c]["strict"] for c in CFGS]
    ns = [data[c]["n"] for c in CFGS]
    ks = [data[c]["strict_k"] for c in CFGS]
    errs = np.array([[max(0, v - wilson_ci(k, n)[0]) for v, k, n in zip(vals, ks, ns)],
                     [max(0, wilson_ci(k, n)[1] - v) for v, k, n in zip(vals, ks, ns)]])
    offs = (i - 0.5) * w
    bars = plt.bar(x + offs, vals, w * 0.92, label=name, color=color, hatch=hatch,
                   edgecolor="black", linewidth=0.6, yerr=errs, capsize=2, ecolor="#333")
    for b, k, n in zip(bars, ks, ns):
        plt.text(b.get_x() + b.get_width() / 2, 0.015, f"{k}/{n}",
                 ha="center", va="bottom", fontsize=6.5, rotation=90,
                 color="white" if hatch == "" else "black", fontweight="bold")
plt.axhline(100 * 0.532, color="#7f8c8d", lw=0.8, ls=":")
plt.text(3.55, 100 * 0.532 + 0.01, "random tool choice 53%", fontsize=5.8, color="#7f8c8d", ha="right")
plt.xticks(x, ["E1", "E2", "E3", "E4"])
plt.ylabel("Strict-specific tool accuracy")
plt.ylim(0, 0.6)
plt.legend(fontsize=6.5, loc="upper left")
plt.grid(axis="y", alpha=0.25, lw=0.4)
plt.title("Tool selection, strict metric (permissive metric is degenerate: 100% by always-power-flow)", fontsize=7.5)
plt.tight_layout()
plt.savefig(FIGDIR / "fig_tools.png", dpi=200)
plt.close()
print("  fig_tools.png saved")

# ---------- Figure: hallucination + latency (log) ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
for i, (name, data, color, hatch) in enumerate(series):
    off = (i - 0.5) * w
    ax1.bar(x + off, [data[c]["halluc"] * 100 for c in CFGS], w * 0.92, label=name,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.6)
    ax2.bar(x + off, [data[c]["lat"] for c in CFGS], w * 0.92, label=name,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.6,
            yerr=[data[c]["lat_sd"] for c in CFGS], capsize=2, ecolor="#333")
ax1.set_xticks(x, ["E1", "E2", "E3", "E4"])
ax1.set_ylabel("Hallucinated rows (%)")
ax1.set_title("(a) Hallucination (rule-based judge)", fontsize=8)
ax2.set_xticks(x, ["E1", "E2", "E3", "E4"])
ax2.set_ylabel("Latency (s, log scale)")
ax2.set_yscale("log")
ax2.set_ylim(0.5, 200)
ax2.set_title("(b) Latency per call (log scale)", fontsize=8)
ax1.legend(fontsize=6.5)
ax1.grid(axis="y", alpha=0.25, lw=0.4)
ax2.grid(axis="y", alpha=0.25, lw=0.4, which="both")
ax1.text(0.02, 0.95, "Exact counts in Table III", transform=ax1.transAxes,
         fontsize=6.3, va="top", style="italic", color="#555")
ax2.text(0.02, 0.95, "~40× latency = the privacy/cost trade-off", transform=ax2.transAxes,
         fontsize=6.3, va="top", style="italic", color="#555")
plt.tight_layout()
plt.savefig(FIGDIR / "fig_halluc_latency.png", dpi=200)
plt.close()
print("  fig_halluc_latency.png saved")

# ---------- Figure: architecture (Muse-free redraw, sized for column) ----------
plt.figure(figsize=(3.4, 1.5))
boxes = [
    (0.01, 0.28, 0.155, 0.44, "IEEE 14/39/118\nscenarios", "#eaf2f8"),
    (0.20, 0.28, 0.16, 0.44, "Measurements\n+ labels", "#eaf2f8"),
    (0.405, 0.56, 0.20, 0.36, "RAG: 8 docs", "#fdebd0"),
    (0.405, 0.06, 0.20, 0.36, "Tools: PF\nN-1, OPF", "#fdebd0"),
    (0.655, 0.28, 0.16, 0.44, "LLM agent\nE1-E4", "#e8f8f5"),
    (0.86, 0.28, 0.13, 0.44, "Advice", "#e8f8f5"),
]
for (bx, by, bw, bh, label, fc) in boxes:
    plt.gca().add_patch(plt.Rectangle((bx, by), bw, bh, fc=fc, ec="#333", lw=0.7))
    plt.text(bx + bw/2, by + bh/2, label, ha="center", va="center", fontsize=7.5)
for x0, x1, y0 in [(0.165, 0.20, 0.50), (0.36, 0.405, 0.74), (0.36, 0.405, 0.24),
                   (0.605, 0.655, 0.50), (0.605, 0.655, 0.24), (0.815, 0.86, 0.50)]:
    plt.annotate("", xy=(x1, y0), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"))
plt.xlim(0, 1); plt.ylim(0, 1); plt.axis("off")
plt.tight_layout(pad=0.3)
plt.savefig(FIGDIR / "fig_architecture.png", dpi=200)
plt.close()
print("  fig_architecture.png saved (column-sized)")


