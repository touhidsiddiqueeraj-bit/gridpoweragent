#!/usr/bin/env python3
"""Methodology diagram: corpus generation + evaluation pipeline (true column width)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent / "paper_gridagent_wiecon" / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

W, H = 3.5, 2.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

EC = "#2D3748"
def box(x, y, text, fc="#EBF4FF", fs=6.8):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=EC, lw=0.8))

def arrow(x0, y0, x1, y1, color=EC):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color))

# Row 1: corpus generation (left to right)
box(0.11, 0.87, "Seeded\ndisturbance\ninjections")
box(0.37, 0.87, "pandapower\npower flow\n(ground truth)")
box(0.63, 0.87, "~120 noisy\nmeters /\nscenario")
box(0.89, 0.87, "Iterative\nWLS state\nestimation")
arrow(0.20, 0.87, 0.29, 0.87)
arrow(0.46, 0.87, 0.56, 0.87)
arrow(0.72, 0.87, 0.82, 0.87)

# Row 2: dual-axis keying + observation channels + closed loop
box(0.11, 0.55, "Dual-axis\nkeying\n(cause +\noutcome flags)", fc="#FEB2B2")
box(0.37, 0.55, "Observation\nchannels\n(desc. / telem.\n/ raw state)")
box(0.63, 0.55, "LLM agents\n(API + local)")
box(0.89, 0.55, "Closed-loop\ntool exec.\n(pandapower)")
arrow(0.20, 0.55, 0.29, 0.55)
arrow(0.46, 0.55, 0.56, 0.55)
arrow(0.72, 0.55, 0.82, 0.55)

# Row 3: metrics
box(0.11, 0.20, "Exact-match\ndiagnosis\n+ paired MDE", fc="#F0FFF4")
box(0.37, 0.20, "Tool-call\nvalidity +\nexecution", fc="#F0FFF4")
box(0.63, 0.20, "Element\nconstruction\naccuracy", fc="#F0FFF4")
arrow(0.20, 0.44, 0.11, 0.31)
arrow(0.37, 0.44, 0.335, 0.31)
arrow(0.63, 0.44, 0.605, 0.31)

fig.tight_layout(pad=0.25)
fig.savefig(FIGDIR / "fig_methodology_diagram.png", dpi=300)
print("fig_methodology_diagram.png saved")
