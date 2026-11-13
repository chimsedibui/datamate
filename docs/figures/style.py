"""Shared look of the baseline figures (figures4papers style): Times serif type, pastel modules with soft vertical gradients,
gradient block arrows with a gap on both sides, rich text with highlighted words, tool logos."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrow, FancyBboxPatch, Polygon
from PIL import Image

HERE = Path(__file__).resolve().parent
LOGOS = HERE / "logos"

C = dict(
    g_fill="#DDE8C8", g_edge="#56702F", g_text="#3D5A1E",
    b_fill="#CFE2F3", b_edge="#2F5D8A", b_text="#1F4E79",
    y_fill="#FBECC1", y_edge="#9C7A1C", y_text="#7A5C0F",
    p_fill="#E8DCEB", p_edge="#5E3B6E", p_text="#4B1F5E",
    r_fill="#F6D0CC", r_edge="#A23B36", grey="#7F7F7F", light="#F2F2F2", dark="#262626", loop="#7B1E1E", gold="#C99A06")


def setup(w, h):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold", "svg.fonttype": "none",
        "text.hinting": "no_hinting", "text.hinting_factor": 1})       # unhinted advances: text segments join without pixel-rounding gaps
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)  # fixed before any text is measured
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_autoscale_on(False)
    return fig, ax


def mix(c, f):
    """c blended toward white by f (0 = c, 1 = white)."""
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f)


def _grad(ax, patch, x, y, w, h, c_from, c_to, horizontal=False, z=2):
    ax.add_patch(patch)
    n = np.linspace(0, 1, 256)
    img = n.reshape(1, -1) if horizontal else n.reshape(-1, 1)
    im = ax.imshow(img, extent=[x, x + w, y, y + h], origin="lower", aspect="auto", interpolation="bicubic", zorder=z,
                   cmap=LinearSegmentedColormap.from_list("g", [c_from, c_to]))
    im.set_clip_path(patch)


def gbox(ax, x, y, w, h, fill, edge, lw=1.8, r=0.25, z=2, top=0.55):
    """Rounded module with a vertical gradient: lighter at the top."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec=edge, lw=lw, zorder=z + 0.2)
    _grad(ax, p, x, y, w, h, fill, mix(fill, top), z=z)


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.15, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def gpoly(ax, pts, fill, edge, lw=2.0, z=2, top=0.55):
    p = Polygon(pts, closed=True, fc="none", ec=edge, lw=lw, zorder=z + 0.2)
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    _grad(ax, p, min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), fill, mix(fill, top), z=z)


def arrow(ax, x0, y0, x1, y1, color, width=0.28, head_w=0.56, head_l=0.24, z=3):
    """Gradient block arrow from (x0, y0) to (x1, y1), light at the tail to a mid tone at the head, outlined in `color`.
    Callers leave a gap to the boxes on both sides."""
    dx, dy = x1 - x0, y1 - y0
    p = FancyArrow(x0, y0, dx, dy, width=width, head_width=head_w, head_length=min(head_l, (dx * dx + dy * dy) ** 0.5 * 0.6),
                   length_includes_head=True, fc="none", ec=color, lw=1.3, zorder=z + 0.2)
    lo, hi = mix(color, 0.88), mix(color, 0.35)
    pad = head_w / 2
    if abs(dx) >= abs(dy):
        _grad(ax, p, min(x0, x1), min(y0, y1) - pad, abs(dx), abs(dy) + 2 * pad, *((lo, hi) if dx > 0 else (hi, lo)), horizontal=True, z=z)
    else:
        _grad(ax, p, min(x0, x1) - pad, min(y0, y1), abs(dx) + 2 * pad, abs(dy), *((lo, hi) if dy > 0 else (hi, lo)), z=z)


def rich(ax, x, y, parts, ha="center", size=11, z=6, **common):
    """One line of text in segments, each (text, style dict): e.g. [("8 / 8", {"weight": "bold", "color": ...}), (" agree", {})].
    Each segment is drawn without its edge spaces and placed by advance widths, edge spaces counted at the font's own space width."""
    fig = ax.figure
    dpi0 = fig.dpi
    fig.set_dpi(600)                                   # measure at high resolution
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    to_data = lambda px: inv.transform((px, 0))[0] - inv.transform((0, 0))[0]
    items = []
    for text, st in parts:
        core = text.strip(" ")
        t = ax.text(0, y, core, fontsize=st.get("size", size), color=st.get("color", common.get("color", C["dark"])),
                    fontweight=st.get("weight", common.get("weight", "normal")), fontstyle=st.get("style", common.get("style", "normal")),
                    va="center", ha="left", zorder=z)
        fam = st.get("family", common.get("family"))
        if fam:
            t.set_family(fam)
        prop = t.get_fontproperties()
        math = "$" in core                                  # mathtext ($v_t$) is measured as rendered, not as its source
        ink = lambda s: renderer.get_text_width_height_descent(s, prop, ismath=math)[0]
        width = lambda s: ink(s + "|") - ink("|")          # advance width: the ink box of s alone over-counts slanted and bold overhangs
        space = to_data(ink("| |") - ink("||"))
        lead, trail = (len(text) - len(text.lstrip(" ")), len(text) - len(text.rstrip(" "))) if core else (len(text), 0)
        items.append((t, lead * space, to_data(width(core)) if core else 0.0, trail * space))
    fig.set_dpi(dpi0)
    total = sum(a + b + c for _, a, b, c in items)
    cur = x - total / 2 if ha == "center" else (x - total if ha == "right" else x)
    for t, lead, w, trail in items:
        t.set_x(cur + lead)
        cur += lead + w + trail
    return total


def flow(ax, x, y, segments, size=11, ha="center", arrow_color=None, gap=None, **common):
    """Text segments joined by small solid arrow glyphs instead of a typed "->": segments is a list of rich() part lists."""
    gap = gap if gap is not None else 0.62 * size / 14
    n0 = len(ax.texts)
    widths = [rich(ax, 0, y, seg, ha="left", size=size, **common) for seg in segments]
    for t in ax.texts[n0:]:
        t.remove()
    total = sum(widths) + gap * (len(segments) - 1)
    cur = x - total / 2 if ha == "center" else (x - total if ha == "right" else x)
    color = arrow_color or common.get("color", C["dark"])
    hw = 0.17 * size / 14
    for k, (seg, w) in enumerate(zip(segments, widths)):
        rich(ax, cur, y, seg, ha="left", size=size, **common)
        cur += w
        if k < len(segments) - 1:
            pad = gap * 0.2
            ax.add_patch(FancyArrow(cur + pad, y, gap - 2 * pad, 0, width=hw * 0.36, head_width=hw, head_length=hw * 0.8, length_includes_head=True,
                                    fc=color, ec="none", zorder=6))
            cur += gap
    return total


def logo(ax, name, x, y, h, z=5, rounded=0.0):
    """Paste logos/<name> with its lower-left corner at (x, y), height h (axes units are inches, so the aspect is kept)."""
    im = Image.open(LOGOS / name).convert("RGBA")
    w = h * im.width / im.height
    art = ax.imshow(np.asarray(im), extent=[x, x + w, y, y + h], aspect="auto", zorder=z, interpolation="lanczos")
    if rounded:
        clip = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounded}", fc="none", ec="none", transform=ax.transData)
        ax.add_patch(clip)
        art.set_clip_path(clip)
    return w


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    print("saved", name)
