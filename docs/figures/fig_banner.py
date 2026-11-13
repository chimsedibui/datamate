"""README banner: a messy shop spreadsheet, the SQL the model writes, a vote, the project name. Text is saved as paths, so GitHub
renders it the same everywhere. Run: python docs/figures/fig_banner.py -> docs/assets/banner.{svg,png}"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "assets"
INK, GREY, BLUE, GREEN, AMBER, LINE = "#1F2A37", "#6B7280", "#2F5D8A", "#3F7D2A", "#B07A12", "#C9D2DE"
plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans"})
W, H = 12.8, 3.2
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.08, 0.08), W - 0.16, H - 0.16, boxstyle="round,pad=0,rounding_size=0.28", fc="#F6F8FB", ec="#DDE3EC", lw=1.4))

# spreadsheet: a header row and a few uneven rows
x0, y0, cw, rh = 0.7, 0.75, 0.52, 0.34
for c in range(4):
    ax.add_patch(Rectangle((x0 + c * cw, y0 + 4 * rh), cw, rh, fc=GREEN, ec="white", lw=1.2, alpha=0.85))
for r in range(4):
    for c in range(4):
        ax.add_patch(Rectangle((x0 + c * cw, y0 + r * rh), cw, rh, fc="white", ec=LINE, lw=1.0))
for r, c in ((0, 1), (1, 3), (2, 0), (3, 2)):
    ax.plot([x0 + c * cw + 0.1, x0 + c * cw + cw - 0.1], [y0 + r * rh + rh / 2] * 2, color=GREY, lw=2.2, alpha=0.6, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((2.9, 1.6), (3.5, 1.6), arrowstyle="-|>", mutation_scale=18, color=GREY, lw=1.6))

# SQL card: the samples the model writes vote for one answer
ax.add_patch(FancyBboxPatch((3.65, 0.8), 3.0, 1.6, boxstyle="round,pad=0,rounding_size=0.16", fc="white", ec=BLUE, lw=1.6))
for k, (text, col) in enumerate((("SELECT sum(amount)", BLUE), ("FROM v_sales WHERE ...", BLUE), ("vote: 48,250,000", AMBER))):
    ax.text(3.85, 2.07 - k * 0.42, text, ha="left", va="center", fontsize=12, family="DejaVu Sans Mono", color=col)
ax.plot([6.0, 6.12, 6.34], [1.24, 1.1, 1.4], color=GREEN, lw=3.0, solid_capstyle="round")

ax.text(7.05, 2.05, "DataMate", ha="left", va="center", fontsize=44, fontweight="bold", color=INK)
ax.text(7.08, 1.28, "Vietnamese questions over messy shop data,", ha="left", va="center", fontsize=15.5, color=INK)
ax.text(7.08, 0.92, "answered by voted DuckDB SQL", ha="left", va="center", fontsize=15.5, color=INK)
ax.text(7.08, 0.5, "Zalo AI Challenge 2026  |  Track 2", ha="left", va="center", fontsize=11.5, color=GREY)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "banner.svg", transparent=True)
fig.savefig(OUT / "banner.png", dpi=200, transparent=True)
print("saved", OUT / "banner.svg")
