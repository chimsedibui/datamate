"""Inference pipeline (figures4papers style). Running example: train question q_06de33fa21cc (biz_118); all 8 plain samples
returned the gold answer. Run: python docs/figures/fig_pipeline.py"""
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from style import C, arrow, box, flow, gbox, gpoly, logo, mix, rich, save, setup

fig, ax = setup(17.35, 6.5)
B, G, I = {"weight": "bold"}, {"color": C["grey"]}, {"style": "italic"}
GAP = 0.12


def title(x, text, color, sub=None):
    ax.text(x, 0.8, text, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    if sub:
        rich(ax, x, 0.4, sub, size=13.5, color=C["grey"], style="italic")


def mini_table(x, y, w, h, name):
    for i in range(4):
        for j in range(3):
            ax.add_patch(Rectangle((x + j * w / 3, y + h - (i + 1) * h / 4), w / 3, h / 4, fc=C["b_edge"] if i == 0 else "white",
                                   ec=C["b_edge"], lw=0.8, zorder=4))
    ax.text(x + w / 2, y - 0.18, name, ha="center", va="center", fontsize=12.5, color=C["b_text"], fontweight="bold", style="italic", zorder=5)


# ---- inputs: shop files and rulebook --------------------------------------------------------------------------------------------
logo(ax, "excel.png", 0.32, 3.25, 0.62)
ax.add_patch(Polygon([[1.12, 3.25], [1.62, 3.25], [1.62, 3.72], [1.5, 3.87], [1.12, 3.87]], closed=True, fc="white", ec=C["grey"], lw=1.3, zorder=4))
ax.text(1.37, 3.48, "CSV", ha="center", va="center", fontsize=10, fontweight="bold", color=C["g_text"], zorder=5)
logo(ax, "sqlite.png", 0.36, 2.55, 0.5)
ax.text(1.17, 2.2, "Shop files", ha="center", va="center", fontsize=14, fontweight="bold", color=C["dark"])
ax.text(1.17, 1.92, "messy tables and README", ha="center", va="center", fontsize=11, color=C["grey"], style="italic")
gbox(ax, 0.42, 0.62, 0.6, 0.78, C["y_fill"], C["y_edge"], lw=1.3, r=0.06)
ax.add_patch(Rectangle((0.42, 0.62), 0.11, 0.78, fc=C["y_edge"], ec="none", zorder=4))
for k in range(3):
    ax.plot([0.62, 0.92], [1.17 - k * 0.17] * 2, color=C["y_edge"], lw=1.1, zorder=4)
ax.text(1.14, 1.12, "Rulebook", ha="left", va="center", fontsize=14, fontweight="bold", color=C["dark"])
ax.text(1.14, 0.84, "+ calendar", ha="left", va="center", fontsize=11, color=C["grey"], style="italic")

# ---- question (enters the prompt from above) --------------------------------------------------------------------------------------
box(ax, 4.9, 5.3, 3.5, 1.02, "white", C["dark"], lw=1.5, r=0.32)
q = [("“sp bán dc ", I), ("nhiều nhất", {**I, **B, "color": C["p_text"]}), (" ", I), ("tuần rồi", {**I, **B, "color": C["y_text"]}), (" là gì?”", I)]
rich(ax, 6.65, 5.93, q, size=14.5)
rich(ax, 6.65, 5.57, [("as_of_date ", G), ("2026-03-04", {"color": C["y_text"], "weight": "bold"})], size=11.5)

# ---- 1 data layer ----------------------------------------------------------------------------------------------------------------
X1, W1 = 2.7, 2.2
gbox(ax, X1, 1.2, W1, 3.1, C["b_fill"], C["b_edge"], r=0.3)
logo(ax, "duckdb.png", X1 + 0.32, 3.38, 0.62, rounded=0.12)
ax.text(X1 + 1.08, 3.69, "DuckDB", ha="left", va="center", fontsize=15, fontweight="bold", color=C["b_text"], zorder=5)
mini_table(X1 + 0.17, 2.15, 0.82, 0.75, "v_sales")
mini_table(X1 + 1.21, 2.15, 0.82, 0.75, "v_orders")
box(ax, X1 + 0.45, 1.42, 1.3, 0.32, "white", C["b_edge"], lw=1.1, r=0.13, z=4)
ax.text(X1 + 1.1, 1.58, "debt_at(day)", ha="center", va="center", fontsize=12, style="italic", color=C["b_text"], zorder=5)
title(X1 + W1 / 2, "Data layer", C["b_text"], [("views ", {}), ("drafted by the LLM", B), (", ", {}), ("tested", B), (" on data", {})])

# ---- 2 prompt --------------------------------------------------------------------------------------------------------------------
X2, W2 = 5.6, 2.1
ax.add_patch(Polygon([[X2, 1.2], [X2 + W2, 1.2], [X2 + W2, 4.0], [X2 + W2 - 0.28, 4.3], [X2, 4.3]], closed=True, fc="white", ec=C["dark"], lw=1.6, zorder=2))
ax.add_patch(Polygon([[X2 + W2 - 0.28, 4.3], [X2 + W2 - 0.28, 4.0], [X2 + W2, 4.0]], closed=True, fc=C["light"], ec=C["dark"], lw=1.2, zorder=3))
parts = [("rulebook", "#E4E4E4", C["dark"]), ("schema and views", C["b_fill"], C["b_text"]), ("date window and hints", C["y_fill"], C["y_text"]),
         ("few-shot SQL", C["g_fill"], C["g_text"]), ("question", C["p_fill"], C["p_text"])]
for k, (name, fc, tc) in enumerate(parts):
    yy = 3.6 - k * 0.5
    gbox(ax, X2 + 0.18, yy, W2 - 0.36, 0.38, fc, mix(tc, 0.35), lw=0.9, r=0.1, z=3)
    ax.text(X2 + W2 / 2, yy + 0.19, name, ha="center", va="center", fontsize=12, color=tc, zorder=5)
title(X2 + W2 / 2, "Prompt", C["dark"], [("≈ ", {}), ("12k", B), (" tokens", {})])

# ---- 3 LLM ---------------------------------------------------------------------------------------------------------------------
X3, X3e = 8.4, 10.15
gpoly(ax, [[X3, 1.3], [X3, 4.2], [X3e, 3.62], [X3e, 1.88]], C["p_fill"], C["p_edge"])
logo(ax, "qwen_gh.png", X3 + 0.62, 3.22, 0.52)
ax.text(X3 + 0.87, 2.84, "Qwen3.5-9B", ha="center", va="center", fontsize=15, fontweight="bold", color=C["p_text"], zorder=5)
ax.text(X3 + 0.87, 2.5, "+ LoRA", ha="center", va="center", fontsize=14, color=C["p_text"], zorder=5)
w = logo(ax, "vllm.png", X3 + 0.28, 1.9, 0.3)
ax.text(X3 + 0.28 + w + 0.05, 2.05, "vLLM (fp8)", ha="left", va="center", fontsize=11, color=C["p_text"], zorder=5)
title(X3 + 0.87, "LLM", C["p_text"], [("≤ 15B", B), (" parameters, offline", {})])

# ---- 4 SQL samples -------------------------------------------------------------------------------------------------------------
X4, W4 = 10.85, 2.85
for k in (2, 1):
    gbox(ax, X4 + 0.1 * k, 1.42 + 0.1 * k, W4, 2.65, C["g_fill"], C["g_edge"], lw=1.2, r=0.15, z=2)
box(ax, X4, 1.42, W4, 2.65, "white", C["g_edge"], lw=1.6, r=0.15, z=3)
M = {"family": "Liberation Mono"}
lines = [[("Notes: last week = [02-23, 03-02)", {**M, "color": C["grey"]})],
         [("TYPE: ", {**M, "color": C["p_text"]}), ("string", {**M, "color": C["p_text"], "weight": "bold"})],
         [("SELECT product_name", M)],
         [("FROM ", M), ("v_sales", {**M, "color": C["b_text"], "weight": "bold"})],
         [("WHERE order_date >= '2026-02-23'", M)],
         [("  AND order_date <  '2026-03-02'", M)],
         [("GROUP BY product_name", M)],
         [("ORDER BY ", M), ("SUM(qty) DESC", {**M, "color": C["g_text"], "weight": "bold"}), (" LIMIT 1", M)]]
for k, seg in enumerate(lines):
    rich(ax, X4 + 0.13, 3.82 - k * 0.3, seg, ha="left", size=9.4)
ax.text(X4 + W4 + 0.1, 4.42, "× 8", ha="center", va="center", fontsize=17, fontweight="bold", color=C["g_text"])
title(X4 + W4 / 2 + 0.1, "SQL samples", C["g_text"], [("+4 thinking", B), (" if unsure", {})])

# ---- 5 execute and vote --------------------------------------------------------------------------------------------------------
X5, W5 = 14.6, 1.4
c5 = X5 + W5 / 2
gbox(ax, X5, 1.2, W5, 3.1, C["y_fill"], C["y_edge"], r=0.3)
logo(ax, "duckdb.png", X5 + 0.5, 3.72, 0.38, rounded=0.08)
for k in range(8):
    ax.add_patch(Circle((X5 + 0.28 + (k % 4) * 0.28, 3.32 - (k // 4) * 0.3), 0.1, fc=C["g_edge"], ec="white", lw=1.0, zorder=5))
rich(ax, c5, 2.68, [("8 / 8", {"weight": "bold", "color": C["g_text"]}), (" agree", {"color": C["y_text"]})], size=12.5)
ax.add_patch(Rectangle((X5 + 0.15, 2.17), W5 - 0.3, 0.24, fc=C["g_edge"], ec=C["y_edge"], lw=1.0, zorder=5))
ax.plot([X5 + W5 / 2] * 2, [2.08, 2.5], color=C["r_edge"], lw=2.0, zorder=6)
rich(ax, c5, 1.88, [("share ≥ ", {"color": C["y_text"]}), ("τ = 0.5", {"weight": "bold", "color": C["r_edge"]})], size=11.5)
ax.text(c5, 1.5, "keys follow the scorer", ha="center", va="center", fontsize=10.5, color=C["y_text"], style="italic", zorder=5)
title(c5, "Execute and vote", C["y_text"], [("read-only", B), (" SQL", {})])

# ---- output ------------------------------------------------------------------------------------------------------------------------
X6, W6 = 16.35, 0.85
c6 = X6 + W6 / 2
gbox(ax, X6, 2.5, W6, 1.6, C["g_fill"], C["g_edge"], lw=1.8, r=0.2)
ax.plot([c6 - 0.17, c6 - 0.05, c6 + 0.18], [3.82, 3.68, 3.96], color=C["g_edge"], lw=3.2, solid_capstyle="round", solid_joinstyle="round", zorder=5)
ax.text(c6, 3.2, "Xi măng\nHà Tiên\nPCB40…", ha="center", va="center", fontsize=11.5, color=C["g_text"], fontweight="bold", zorder=5, linespacing=1.1)
ax.text(c6, 2.66, "string", ha="center", va="center", fontsize=10, color=C["grey"], family="Liberation Mono", zorder=5)
gbox(ax, X6, 1.42, W6, 0.78, C["r_fill"], C["r_edge"], lw=1.4, r=0.2)
ax.text(c6, 1.93, "share < τ", ha="center", va="center", fontsize=10.5, color=C["r_edge"], zorder=5)
ax.text(c6, 1.64, "abstain", ha="center", va="center", fontsize=12.5, color=C["r_edge"], fontweight="bold", zorder=5)
title(c6, "Answer", C["g_text"])

# ---- arrows (a gap on both sides) ---------------------------------------------------------------------------------------------
arrow(ax, 2.0 + GAP, 3.2, X1 - GAP, 3.2, C["b_edge"])
arrow(ax, 2.05 + GAP, 1.02, X1 - GAP, 1.02, C["y_edge"])
arrow(ax, X1 + W1 + GAP, 2.75, X2 - GAP, 2.75, C["b_edge"])
arrow(ax, 6.65, 5.3 - GAP, 6.65, 4.3 + GAP, C["p_edge"])
arrow(ax, X2 + W2 + GAP, 2.75, X3 - GAP, 2.75, C["dark"])
arrow(ax, X3e + GAP, 2.75, X4 - GAP, 2.75, C["p_edge"])
arrow(ax, X4 + W4 + 0.2 + GAP, 2.75, X5 - GAP, 2.75, C["g_edge"])
arrow(ax, X5 + W5 + GAP, 3.3, X6 - GAP, 3.3, C["y_edge"], width=0.22, head_w=0.44, head_l=0.18)
ax.add_patch(FancyArrowPatch((X5 + W5 + GAP, 1.81), (X6 - GAP, 1.81), arrowstyle="-|>", mutation_scale=12, color=C["r_edge"], lw=1.5, zorder=3))
# retry loop: failed or low-agreement questions go back to the LLM
ax.plot([c5, c5, X3 + 1.35, X3 + 1.35], [4.3 + GAP, 4.72, 4.72, 3.95], color=C["loop"], lw=1.7, ls=(0, (4, 2.5)), zorder=1)
ax.add_patch(FancyArrowPatch((X3 + 1.35, 4.05), (X3 + 1.35, 3.72 + GAP), arrowstyle="-|>", mutation_scale=14, color=C["loop"], lw=1.7, zorder=3))
flow(ax, 12.6, 4.95, [[("SQL error", B)], [("repair turn", {})]], size=13, ha="right", color=C["loop"], style="italic")
flow(ax, 13.1, 4.95, [[("low agreement", B)], [("thinking samples", {})]], size=13, ha="left", color=C["loop"], style="italic")

save(fig, "pipeline")
