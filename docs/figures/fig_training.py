"""Training loop (figures4papers style, same look as pipeline.png): iterative self-training by rejection sampling, with a teacher
added in the last round. Training split only. Run: python docs/figures/fig_training.py"""
from matplotlib.patches import FancyArrowPatch, Rectangle

from style import C, arrow, box, flow, gbox, gpoly, logo, rich, save, setup

fig, ax = setup(17.2, 7.15)
ax.set_ylim(-1.05, 6.1)      # same scale, room below for the optional sources
B = {"weight": "bold"}
GAP = 0.12


def title(x, text, color, y=4.62):
    ax.text(x, y, text, ha="center", va="center", fontsize=17, fontweight="bold", color=color, linespacing=1.05)


def sub(x, parts, y=1.6):
    rich(ax, x, y, parts, size=13.5, color=C["grey"], style="italic")


def tick(x, y, color, s=0.1):
    ax.plot([x - s, x - s * 0.3, x + s], [y, y - s * 0.7, y + s * 0.9], color=color, lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=6)


def cross(x, y, color, s=0.08):
    for a, b in ((-s, s), (s, -s)):
        ax.plot([x - s, x + s], [y + a, y + b], color=color, lw=2.6, solid_capstyle="round", zorder=6)


# ---- labelled training set ----------------------------------------------------------------------------------------------------
for k in (2, 1, 0):
    gbox(ax, 0.3 + 0.12 * k, 2.05 + 0.12 * k, 1.55, 1.75, C["b_fill"] if k else "white", C["b_edge"], lw=1.4, r=0.15, z=2 + (2 - k) * 0.3)
for k in range(4):
    ax.plot([0.5, 1.6 - 0.2 * (k % 2)], [3.45 - k * 0.3] * 2, color=C["b_edge"], lw=1.6, zorder=4, solid_capstyle="round")
gbox(ax, 1.36, 2.17, 0.38, 0.38, C["g_fill"], C["g_edge"], lw=1.2, r=0.19, z=4)
tick(1.55, 2.36, C["g_edge"], 0.09)
title(1.2, "Labelled\ntraining set", C["b_text"])
sub(1.2, [("3,000", B), (" questions", {})])
sub(1.2, [("with gold answers", {})], y=1.28)

# ---- 1 candidate generation (the student) ----------------------------------------------------------------------------------------
XS, XSe = 2.75, 4.45
gpoly(ax, [[XS, 1.95], [XS, 4.0], [XSe, 3.55], [XSe, 2.4]], C["p_fill"], C["p_edge"])
logo(ax, "qwen_gh.png", XS + 0.55, 3.25, 0.4)
ax.text(XS + 0.78, 2.95, "Student 9B", ha="center", va="center", fontsize=14.5, fontweight="bold", color=C["p_text"], zorder=5)
rich(ax, XS + 0.78, 2.6, [("+ LoRA ", {}), ("$v_t$", {})], size=13.5, color=C["p_text"])
title(XS + 0.85, "Candidate\ngeneration", C["p_text"])
sub(XS + 0.85, [("8", B), (" plain and ", {}), ("4", B), (" thinking", {})])
sub(XS + 0.85, [("samples per question", {})], y=1.28)

# ---- 2 rejection sampling ------------------------------------------------------------------------------------------------------------
XF, WF = 5.25, 2.3
gbox(ax, XF, 1.95, WF, 2.05, C["y_fill"], C["y_edge"], r=0.3)
for k, ok in enumerate([True, True, False, True, False]):
    yy = 3.62 - k * 0.36
    box(ax, XF + 0.2, yy - 0.12, 1.35, 0.24, "white", C["grey"], lw=0.9, r=0.1, z=4)
    for j in range(3):
        ax.plot([XF + 0.3 + j * 0.4, XF + 0.6 + j * 0.4], [yy, yy], color="#DDDDDD" if j else C["grey"], lw=2.2, zorder=5, solid_capstyle="round")
    (tick if ok else cross)(XF + 1.85, yy, C["g_edge"] if ok else C["r_edge"])
title(XF + WF / 2, "Rejection\nsampling", C["y_text"])

# ---- 3 training-set construction -------------------------------------------------------------------------------------------------------
XT, WT = 8.3, 3.15
gbox(ax, XT, 1.95, WT, 2.05, C["g_fill"], C["g_edge"], r=0.3)
ax.text(XT + 0.25, 3.7, "prompt (no loss)", ha="left", va="center", fontsize=12, color=C["grey"], style="italic", zorder=5)
ax.text(XT + WT - 0.25, 3.7, "reply", ha="right", va="center", fontsize=12, color=C["g_text"], fontweight="bold", zorder=5)
for k in range(3):
    yy = 3.25 - k * 0.42
    ax.add_patch(Rectangle((XT + 0.25, yy - 0.13), 2.27, 0.26, fc="#D9D9D9", ec="white", lw=0.8, zorder=4))
    ax.add_patch(Rectangle((XT + 2.52, yy - 0.13), 0.38, 0.26, fc=C["g_edge"], ec="white", lw=0.8, zorder=4))
rich(ax, XT + WT / 2, 2.16, [("2,910", B), (" rows in v3, loss on ", {}), ("3.7 %", B), (" of tokens", {})], size=12, color=C["g_text"])
title(XT + WT / 2, "Training-set\nconstruction", C["g_text"])

# ---- optional sources of solutions for hard questions (training split only) ----------------------------------------------------------
PX0, PX1, PY0, PY1 = 4.75, 11.75, -0.98, 1.1
box(ax, PX0, PY0, PX1 - PX0, PY1 - PY0, "#FAFAFA", C["grey"], lw=1.3, r=0.25, z=1, ls=(0, (5, 3)))
rich(ax, (PX0 + PX1) / 2, 0.84, [("Optional: ", {"weight": "bold", "color": C["dark"]}), ("extra solutions for hard questions", {})], size=14,
     color=C["dark"], style="italic")


def source(cx, fill, edge, name, sub, text_c, sub_c, line1, line2, c1):
    x0, x1 = cx - 1.05, cx + 1.05
    gpoly(ax, [[x0, -0.22], [x0, 0.6], [x1, 0.47], [x1, -0.09]], fill, edge, lw=2.0, top=0.3 if fill == "#3A3A3A" else 0.55)
    logo(ax, "qwen_gh.png", x0 + 0.13, 0.0, 0.36)
    ax.text(x0 + 0.6, 0.35, name, ha="left", va="center", fontsize=14, fontweight="bold", color=text_c, zorder=5)
    ax.text(x0 + 0.6, 0.06, sub, ha="left", va="center", fontsize=12, color=sub_c, zorder=5)
    rich(ax, cx, -0.5, line1, size=13.5, color=c1)
    rich(ax, cx, -0.8, line2, size=13, color=C["dark"], style="italic")


source(6.55, C["p_fill"], C["p_edge"], "Student 9B", "thinking mode", C["p_text"], C["p_text"],
       [("Self-distillation", {"weight": "bold"}), (" in v2", {})], [("991", B), (" of the 2,398 rows", {})], C["p_text"])
source(9.95, "#3A3A3A", C["gold"], "Teacher 27B", "training split only", "white", "#DADADA",
       [("Teacher distillation", {"weight": "bold"}), (" in v3", {})], [("solves ", {}), ("26 of 29", B), (" unsolved questions", {})], "#9C7A06")
ax.add_patch(FancyArrowPatch((XF + WF / 2, 1.95 - GAP), (XF + WF / 2, PY1 + GAP), arrowstyle="-|>", mutation_scale=15, color=C["r_edge"], lw=2.2, zorder=3))
rich(ax, XF + WF / 2 + 0.15, 1.53, [("few or no correct samples", {"weight": "bold", "color": C["r_edge"]})], ha="left", size=13)
ax.add_patch(FancyArrowPatch((9.95, PY1 + GAP), (9.95, 1.95 - GAP), arrowstyle="-|>", mutation_scale=15, color=C["gold"], lw=2.2, zorder=3))
rich(ax, 10.1, 1.53, [("correct solutions", {"weight": "bold", "color": "#9C7A06"})], ha="left", size=13)

# ---- 4 parameter-efficient fine-tuning -----------------------------------------------------------------------------------------------------
XQ, XQe = 12.2, 13.8
gpoly(ax, [[XQ, 1.95], [XQ, 4.0], [XQe, 3.55], [XQe, 2.4]], "#EFEAF1", C["p_edge"])
ax.text(XQ + 0.72, 3.2, "Base 9B", ha="center", va="center", fontsize=15, fontweight="bold", color=C["p_text"], zorder=5)
ax.text(XQ + 0.72, 2.85, "4-bit, frozen", ha="center", va="center", fontsize=13, color=C["p_text"], zorder=5)
w = logo(ax, "unsloth.png", XQ + 0.22, 2.22, 0.36, rounded=0.08)
ax.text(XQ + 0.22 + w + 0.06, 2.4, "Unsloth", ha="left", va="center", fontsize=12.5, color=C["p_text"], zorder=5)
title(XQ + 0.8, "Parameter-efficient\nfine-tuning", C["p_text"])
sub(XQ + 0.8, [("QLoRA", B), (", one epoch", {})])
sub(XQ + 0.8, [("about 8 h on 24 GB", {})], y=1.28)

# ---- output ---------------------------------------------------------------------------------------------------------------------------------
XO, WO = 15.55, 1.45
gbox(ax, XO, 2.45, WO, 1.1, C["g_fill"], C["g_edge"], lw=1.9, r=0.3)
ax.text(XO + WO / 2, 3.18, "Updated", ha="center", va="center", fontsize=14, color=C["g_text"], zorder=5)
rich(ax, XO + WO / 2, 2.82, [("adapter ", B), ("$v_{t+1}$", {})], size=14.5, color=C["g_text"])
arrow(ax, XQe + GAP, 3.0, XO - GAP, 3.0, C["g_edge"])
ax.text((XQe + XO) / 2, 3.45, "trains", ha="center", va="center", fontsize=13.5, color=C["g_text"], style="italic")
rich(ax, (XQe + XO) / 2, 2.55, [("rank 16", B)], size=13, color=C["grey"])

# ---- arrows -------------------------------------------------------------------------------------------------------------------------------------
arrow(ax, 2.0 + GAP, 3.0, XS - GAP, 3.0, C["b_edge"])
arrow(ax, XSe + GAP, 3.0, XF - GAP, 3.0, C["p_edge"])
arrow(ax, XF + WF + GAP, 3.0, XT - GAP, 3.0, C["y_edge"])
rich(ax, (XF + WF + XT) / 2, 3.52, [("correct", {"weight": "bold", "color": C["g_text"]})], size=13.5)
arrow(ax, XT + WT + GAP, 3.0, XQ - GAP, 3.0, C["g_edge"])
# next round: the updated adapter samples again
ax.plot([XO + WO / 2, XO + WO / 2, XS + 0.15, XS + 0.15], [3.55 + GAP, 5.42, 5.42, 4.25], color=C["loop"], lw=1.8, ls=(0, (4, 2.5)), zorder=1)
ax.add_patch(FancyArrowPatch((XS + 0.15, 4.35), (XS + 0.15, 3.96 + GAP * 0.6), arrowstyle="-|>", mutation_scale=14, color=C["loop"], lw=1.8, zorder=3))
flow(ax, 9.7, 5.68, [[("Iterative self-training:  ", {"weight": "normal"}), ("v1", {})], [("v2", {})], [("v3", {})]], size=15, color=C["loop"], style="italic", weight="bold")
rich(ax, 9.7, 5.17, [("each round samples again with the newest adapter", {})], size=13, color=C["grey"], style="italic")

save(fig, "training")
