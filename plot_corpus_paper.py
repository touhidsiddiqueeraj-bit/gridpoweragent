#!/usr/bin/env python3
"""Corpus-paper figures, rendered at true column width (3.5 in, IEEEtran)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROCESSED = HERE / "data" / "processed"
FIGDIR = HERE / "paper_gridagent_wiecon" / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
})

# ---- Fig A: corpus composition (class x system, count per class) ----
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
fig.savefig(FIGDIR / "fig_corpus_composition.png", dpi=300)
plt.close(fig)
print("fig_corpus_composition.png saved")

# ---- Fig B: observation channels x tiers (grouped bars, values labeled) ----
conds = ["Desc. only", "Desc. + state", "Telemetry", "Telem. (WLS)", "Raw state"]
api_conds = [100.0, 100.0, 93.9, 8.0, 97.1]
loc_conds = [96.8, np.nan, np.nan, np.nan, np.nan]

x = np.arange(len(conds))
w = 0.36
fig, ax = plt.subplots(figsize=(3.5, 2.3))
ax.bar(x - w / 2, api_conds, w, label="API (Flash Lite)", color="#4C72B0",
       edgecolor="black", linewidth=0.4)
loc_bars = [96.8, np.nan, np.nan, np.nan, np.nan]
ax.bar(x + w / 2, [v if not np.isnan(v) else 0 for v in loc_bars], w,
       label="Local (Gemma 4B Q4)", color="#DD8452", edgecolor="black", linewidth=0.4)
for xi, v in zip(x - w / 2, api_conds):
    ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
ax.axhline(20.0, color="#555", linestyle=":", linewidth=0.8)
ax.text(len(conds) - 0.55, 21.2, "majority 20.0", fontsize=6, color="#555", ha="right")
ax.axhline(12.5, color="#999", linestyle=":", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["Desc. only", "Desc. + state", "Telemetry", "Telem. (WLS)", "Raw state"], fontsize=7)
ax.set_ylabel("Diagnosis accuracy (%)")
ax.set_ylim(0, 112)
ax.legend(loc="upper right", frameon=False, fontsize=7)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(FIGDIR / "fig_observation_channels.png", dpi=300)
plt.close(fig)
print("fig_observation_channels.png saved")
