"""Docker packaging (figures4papers style, same look as pipeline.png): image layers on the left, the offline run on the right.
Run: python docs/figures/fig_docker.py"""
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from style import C, arrow, gbox, logo, rich, save, setup

fig, ax = setup(17.4, 5.9)
B = {"weight": "bold"}
GAP = 0.12

# ---- left: image layers (docker build) ---------------------------------------------------------------------------------------------
ax.text(2.75, 5.45, "docker build", ha="center", va="center", fontsize=19, fontweight="bold", color=C["dark"], family="Liberation Mono")
layers = [  # bottom to top
    ("Base image", "vllm-openai v0.32.0, pinned by digest", "#E4E4E4", C["dark"]),
    ("Runtime libraries", "duckdb, pandas, openpyxl, calamine", C["b_fill"], C["b_text"]),
    ("Base model", "Qwen3.5-9B at a fixed revision", C["p_fill"], C["p_text"]),
    ("LoRA adapter", "trained on the training split only", C["g_fill"], C["g_text"]),
    ("Code and example bank", "datamate package and few-shot bank", C["y_fill"], C["y_text"]),
    ("Entry point", "predict.sh", C["r_fill"], C["r_edge"]),
]
for k, (name, desc, fill, tc) in enumerate(layers):
    y = 0.42 + k * 0.78
    gbox(ax, 0.3, y, 4.9, 0.7, fill, tc, lw=1.4, r=0.14)
    ax.text(0.5, y + 0.47, name, ha="left", va="center", fontsize=16, fontweight="bold", color=tc, zorder=5)
    ax.text(0.5, y + 0.19, desc, ha="left", va="center", fontsize=13.5, color=C["dark"], style="italic", zorder=5)
logo(ax, "vllm.png", 4.6, 0.5, 0.52)
logo(ax, "qwen_gh.png", 4.62, 2.07, 0.5)

# ---- right: the run (docker run, no network) --------------------------------------------------------------------------------------
ax.text(11.6, 5.45, "docker run  --network none  --gpus all", ha="center", va="center", fontsize=17.5, fontweight="bold", color=C["dark"], family="Liberation Mono")
arrow(ax, 5.2 + GAP, 2.75, 6.15 - GAP, 2.75, C["dark"])

# /data folder glyph
ax.add_patch(Polygon([[6.2, 2.0], [6.2, 3.5], [6.55, 3.5], [6.7, 3.68], [7.3, 3.68], [7.3, 3.5], [7.8, 3.5], [7.8, 2.0]], closed=True, fc=C["light"], ec=C["grey"], lw=1.5, zorder=3))
ax.text(7.0, 2.75, "/data", ha="center", va="center", fontsize=17, fontweight="bold", color=C["dark"], family="Liberation Mono", zorder=5)
ax.text(7.0, 1.7, "read-only", ha="center", va="center", fontsize=14, color=C["grey"], style="italic")


def gpu_lane(y, k, shops):
    gbox(ax, 8.55, y, 3.75, 1.3, C["b_fill"], C["b_edge"], r=0.25)
    # GPU chip glyph
    ax.add_patch(Rectangle((8.78, y + 0.33), 0.64, 0.64, fc=C["b_edge"], ec="none", zorder=4))
    ax.add_patch(Rectangle((8.93, y + 0.48), 0.34, 0.34, fc=C["b_fill"], ec="none", zorder=5))
    for j in range(4):
        for (dx, dy, w, h) in ((0.09 + j * 0.15, -0.08, 0.05, 0.08), (0.09 + j * 0.15, 0.64, 0.05, 0.08)):
            ax.add_patch(Rectangle((8.78 + dx, y + 0.33 + dy), w, h, fc=C["b_edge"], ec="none", zorder=4))
    rich(ax, 9.6, y + 0.88, [(f"GPU {k}", B), (": predict.py", {"family": "Liberation Mono"})], ha="left", size=15.5, color=C["b_text"])
    rich(ax, 9.6, y + 0.43, [(shops, {"style": "italic"})], ha="left", size=14, color=C["dark"])


gpu_lane(3.15, 0, "shops of shard 0")
gpu_lane(1.05, 1, "shops of shard 1")
ax.add_patch(FancyArrowPatch((7.8 + GAP, 3.0), (8.55 - GAP, 3.75), arrowstyle="-|>", mutation_scale=15, color=C["dark"], lw=1.8, zorder=3))
ax.add_patch(FancyArrowPatch((7.8 + GAP, 2.5), (8.55 - GAP, 1.7), arrowstyle="-|>", mutation_scale=15, color=C["dark"], lw=1.8, zorder=3))

# merge and outputs
gbox(ax, 12.95, 2.1, 1.45, 1.3, C["y_fill"], C["y_edge"], r=0.25)
ax.text(13.675, 2.94, "merge", ha="center", va="center", fontsize=16, fontweight="bold", color=C["y_text"], zorder=5)
ax.text(13.675, 2.53, "shards", ha="center", va="center", fontsize=14, color=C["y_text"], zorder=5)
ax.add_patch(FancyArrowPatch((12.3 + GAP, 3.8), (12.95 - GAP, 3.05), arrowstyle="-|>", mutation_scale=15, color=C["b_edge"], lw=1.8, zorder=3))
ax.add_patch(FancyArrowPatch((12.3 + GAP, 1.7), (12.95 - GAP, 2.45), arrowstyle="-|>", mutation_scale=15, color=C["b_edge"], lw=1.8, zorder=3))
arrow(ax, 14.4 + GAP, 2.75, 15.0 - GAP, 2.75, C["g_edge"])
for k, (name, sub) in enumerate((("submission.jsonl", "one line per question"), ("logs/", "one file per question"))):
    y = 3.0 - k * 1.05
    gbox(ax, 15.0, y, 2.35, 0.85, C["g_fill"] if k == 0 else C["light"], C["g_edge"] if k == 0 else C["grey"], lw=1.5, r=0.18)
    ax.text(15.12, y + 0.57, name, ha="left", va="center", fontsize=14, fontweight="bold", family="Liberation Mono", color=C["g_text"] if k == 0 else C["dark"], zorder=5)
    ax.text(15.12, y + 0.23, sub, ha="left", va="center", fontsize=13, style="italic", color=C["dark"], zorder=5)
ax.text(16.17, 4.15, "/result", ha="center", va="center", fontsize=17, fontweight="bold", family="Liberation Mono", color=C["g_text"])

# time budget along the bottom
cx, cy = 9.0, 0.42
ax.add_patch(Circle((cx, cy), 0.22, fc="white", ec=C["r_edge"], lw=1.8, zorder=4))
ax.plot([cx, cx], [cy, cy + 0.14], color=C["r_edge"], lw=1.8, zorder=5)
ax.plot([cx, cx + 0.1], [cy, cy - 0.05], color=C["r_edge"], lw=1.8, zorder=5)
rich(ax, 9.35, 0.42, [("6 h", {"weight": "bold", "color": C["r_edge"]}), (" in total with model loading, so each process has a timeout and writes after every shop", {})], ha="left", size=14.5)

save(fig, "docker")
