#!/usr/bin/env python3
"""Corpus-paper figures — final version, clean formatting."""
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

# ---- Fig: corpus composition (class x system) ----
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
ax.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=2,
          columnspacing=0.8, handlelength=1.0)
ax.set_ylim(0, 780)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(FIGDIR / "fig_corpus_composition.png", dpi=300)
plt.close(fig)
print("fig_corpus_composition.png saved")

# ---- Fig: observation channels (grouped bars, values labeled, legend above) ----
conds = ["Desc.\nonly", "Desc.\n+state", "Telem.", "Telem.\n(WLS)", "Raw\nstate"]
api_vals = [100.0, 100.0, 93.9, 97.1, 8.0]
loc_vals = [96.8, np.nan, np.nan, np.nan, np.nan]

x = np.arange(len(conds))
bw = 0.32
fig, ax = plt.subplots(figsize=(3.5, 2.3))
bars1 = ax.bar(x - bw / 2, api_vals, bw, label="API (Flash Lite)", color="#4C72B0",
               edgecolor="black", linewidth=0.4)
loc_clean = [v if not np.isnan(v) else 0 for v in loc_vals]
bars2 = ax.bar(x + bw / 2, loc_vals, bw, label="Local (Gemma 4 E4B)", color="#DD8452",
               edgecolor="black", linewidth=0.4)
for xi, v in zip(x - bw / 2, api_vals):
    ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6)
for xi, v in zip(x + bw / 2, loc_vals):
    if not np.isnan(v):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6)
ax.axhline(20.0, color="#555", linestyle=":", linewidth=0.8)
ax.text(0.22, 21.8, "majority 20", fontsize=6, color="#555", ha="left")
ax.axhline(12.5, color="#999", linestyle=":", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(conds, fontsize=7)
ax.set_ylabel("Diagnosis accuracy (%)")
ax.set_ylim(0, 118)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False,
          fontsize=6.5, columnspacing=1.2, handlelength=1.2)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(FIGDIR / "fig_observation_channels.png", dpi=300)
plt.close(fig)
print("fig_observation_channels.png saved")
