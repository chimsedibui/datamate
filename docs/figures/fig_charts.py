"""Data charts for the baseline slides, same look as pipeline.png (Times type, the module palette, no top/right spines, values
written on the marks). Numbers come from the repo's runs and docs; run from the repo root: python docs/figures/fig_charts.py"""
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from style import C, mix  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 13,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Serif", "mathtext.bf": "Liberation Serif:bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.4, "axes.edgecolor": C["dark"],
    "axes.labelcolor": C["dark"], "xtick.color": C["dark"], "ytick.color": C["dark"], "legend.frameon": False,
    "svg.fonttype": "none", "text.hinting": "no_hinting"})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved", name)


read = lambda p: [json.loads(l) for l in open(ROOT / p, encoding="utf-8-sig") if l.strip()]


# ---- 1 public score by version ---------------------------------------------------------------------------------------------
steps = [("pandas\nagent", 0.25), ("SQL with\nbase 9B", 0.394), ("LoRA v1", 0.428), ("LoRA v2 and\nfew-shot", 0.491), ("base-unit\nquantities", 0.516),
         ("fresh\nsamples", 0.60), ("notes\nbefore SQL", 0.63), ("question-shape\nhints", 0.696)]
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(len(steps))
y = [v for _, v in steps]
ax.axvspan(0.5, len(steps) - 0.5, color=mix(C["b_fill"], 0.45), zorder=0, lw=0)
ax.text(len(steps) - 0.6, 0.27, "text-to-SQL pipeline", ha="right", va="center", fontsize=12.5, color=C["b_text"], style="italic")
ax.plot(x, y, color=C["b_edge"], lw=2.6, zorder=2)
ax.scatter(x[:-1], y[:-1], s=70, color="white", edgecolor=C["b_edge"], lw=2.2, zorder=3)
ax.scatter([x[-1]], [y[-1]], s=170, color=C["g_edge"], edgecolor="white", lw=2.0, zorder=4)
for i, v in enumerate(y):
    ax.text(i, v + 0.022, (f"{v:.3f}".rstrip("0") if round(v, 2) != v else f"{v:.2f}"), ha="center", va="bottom", fontsize=13,
            fontweight="bold" if i == len(y) - 1 else "normal", color=C["g_text"] if i == len(y) - 1 else C["dark"])
ax.set_xticks(x, [s for s, _ in steps], fontsize=12)
ax.set_ylim(0.2, 0.75); ax.set_ylabel("Public PA")
ax.tick_params(axis="x", length=0)
save(fig, "chart_progress")

# ---- 2 prompt composition (mean tokens, public run of the 0.683 setup) ---------------------------------------------------------
parts = [("SQL rules", 1307, "#BDBDBD", C["dark"]), ("rulebook", 5665, "#E4E4E4", C["dark"]), ("README", 984, mix(C["b_edge"], 0.55), C["b_text"]),
         ("schema", 3919, C["b_fill"], C["b_text"]), ("views and debts", 1065, mix(C["b_edge"], 0.3), "white"),
         ("date window and hints", 757, C["y_fill"], C["y_text"]), ("few-shot", 522, C["g_fill"], C["g_text"]), ("question", 71, C["p_fill"], C["p_text"])]
fig, ax = plt.subplots(figsize=(8.2, 4.3))
names = [p[0] for p in parts][::-1]
vals = [p[1] for p in parts][::-1]
cols = [p[2] for p in parts][::-1]
edges = [mix(p[3], 0.2) if p[3] != "white" else C["b_edge"] for p in parts][::-1]
ax.barh(names, vals, color=cols, edgecolor=edges, lw=1.2, height=0.66)
for j, v in enumerate(vals):
    ax.text(v + 80, j, f"{v:,}", va="center", fontsize=13, fontweight="bold", color=C["dark"])
ax.set_xlim(0, 7000); ax.tick_params(axis="y", length=0, labelsize=13.5); ax.spines["left"].set_visible(False)
ax.set_xlabel("mean tokens per prompt (public run of the 0.683 setup)")
ax.text(6900, 0.2, f"≈ {sum(vals) / 1000:.1f}k tokens", ha="right", va="center", fontsize=16, fontweight="bold", color=C["dark"])
save(fig, "chart_prompt_tokens")

# ---- train pool: vote threshold sweep, error classes, question shapes -----------------------------------------------------------
from datamate import scoring, voting  # noqa: E402
from datamate.hints import DEBT_WORDS, LOT, PROFIT, TRIPS  # noqa: E402
from datamate.ingest import norm  # noqa: E402

gold = {a["id"]: a for a in read("data/train/answers.jsonl")}
trace = {r["id"]: r for r in read("runs/sql_v8_pool/trace.jsonl")}


def replay(r, tau):
    texts, preds = [s["text"] for s in r["samples"]], [s["pred"] for s in r["samples"]]
    pool = voting.debt_pool(texts, 8, bool(DEBT_WORDS.search(norm(r["question"]))))
    ans, conf = voting.vote([preds[j] for j in pool], len(pool))
    return voting.final_answer(ans, conf, tau, r["final"].get("reason") == "period outside the shop's data dates", True)


taus = np.round(np.arange(0.2, 0.96, 0.05), 3)
pa, cov = [], []
for t in taus:
    s = [scoring.score(replay(r, t), gold[i]) for i, r in trace.items()]
    pa.append((s.count(1) - s.count(-1)) / len(s)); cov.append((s.count(1) + s.count(-1)) / len(s))
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.plot(taus, cov, color=C["grey"], lw=2.2, ls=(0, (4, 2)), label="answered")
ax.plot(taus, pa, color=C["b_edge"], lw=2.8, label="PA")
k = int(np.argmin(abs(taus - 0.5)))
ax.axvline(0.5, color=C["r_edge"], lw=1.6, ls=":", zorder=0)
ax.scatter([0.5], [pa[k]], s=90, color=C["r_edge"], zorder=4)
ax.annotate(f"PA {pa[k]:.3f} at τ = 0.5", (0.5, pa[k]), (0.22, 0.83), fontsize=12.5, color=C["r_edge"], fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-|>", color=C["r_edge"], lw=1.3, connectionstyle="arc3,rad=-0.2"))
ax.set_xlabel("vote share needed to answer (τ)"); ax.set_ylim(0.6, 1.02)
ax.legend(loc="lower left", fontsize=12)
save(fig, "chart_threshold")


def kind(r):
    g = gold[r["id"]]
    s = scoring.score(r["final"], g)
    if s == 1:
        return "right"
    if not any(p and scoring.score(p, g) == 1 for p in (x["pred"] for x in r["samples"])):
        return "no sample right"
    if s == -1:
        return "wrong group won"
    return "right group below τ" if r.get("vote") and scoring.score(r["vote"], g) == 1 else "right sample outvoted"


kinds = Counter(kind(r) for r in trace.values())
shape = lambda q: ("debt" if DEBT_WORDS.search(norm(q)) else "profit" if PROFIT.search(norm(q)) else "delivery trips" if TRIPS.search(norm(q))
                   else "lots / expiry" if LOT.search(norm(q)) else "other")
by = {}
for i, r in trace.items():
    by.setdefault(shape(r["question"]), []).append(scoring.score(r["final"], gold[i]))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.7), gridspec_kw={"width_ratios": [1.05, 1]})
order = ["no sample right", "wrong group won", "right group below τ", "right sample outvoted"]
cols = [C["r_edge"], mix(C["r_edge"], 0.45), C["y_edge"], C["grey"]]
vals = [kinds[o] for o in order]
a1.barh(order[::-1], vals[::-1], color=cols[::-1], height=0.6)
for j, v in enumerate(vals[::-1]):
    a1.text(v + 1, j, str(v), va="center", fontsize=12.5, fontweight="bold", color=C["dark"])
a1.set_title(f"a   questions not right ({sum(vals)} of {len(trace)})", loc="left", fontsize=14, fontweight="bold", color=C["dark"])
a1.tick_params(axis="y", length=0); a1.set_xlim(0, max(vals) * 1.2); a1.spines["left"].set_visible(False)
names = ["other", "debt", "profit", "lots / expiry", "delivery trips"]
pas = [sum(by[n]) / len(by[n]) for n in names]
a2.barh(names[::-1], pas[::-1], color=[C["g_edge"] if p > 0.5 else (mix(C["y_edge"], 0.2) if p > 0 else C["r_edge"]) for p in pas[::-1]], height=0.6)
for j, (n, p) in enumerate(zip(names[::-1], pas[::-1])):
    a2.text(max(p, 0) + 0.04, j, f"{p:+.2f}  (n={len(by[n])})", va="center", ha="left", fontsize=12, color=C["dark"])
a2.axvline(0, color=C["dark"], lw=1.2)
a2.set_xlim(-1.25, 1.55); a2.set_title("b   PA by question shape", loc="left", fontsize=14, fontweight="bold", color=C["dark"])
a2.tick_params(axis="y", length=0); a2.spines["left"].set_visible(False)
fig.tight_layout(w_pad=3)
save(fig, "chart_errors")

# ---- training: SFT rows, token share, loss ---------------------------------------------------------------------------------------
rows = read("data/sft/sql_rft_v3.jsonl")
route = Counter((r.get("route"), bool(r.get("thinking"))) for r in rows)
log = (ROOT / "runs/lora/sql_rft_v3.log").read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
losses = [float(ast.literal_eval(l.strip())["loss"]) for l in log if l.strip().startswith("{'loss'")]
tok_line = next(l for l in log if l.startswith("tokens:"))
total, trained = [int(v.replace(",", "")) for v in re.findall(r"([\d,]+) (?:total|trained)", tok_line)]

fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(14, 3.7), gridspec_kw={"width_ratios": [1.15, 0.75, 1.3]})
rts = ["view", "macro", "raw"]
plain = [route[(r, False)] for r in rts]
think = [route[(r, True)] for r in rts]
b1.barh(rts[::-1], plain[::-1], color=C["g_edge"], height=0.6, label="plain")
b1.barh(rts[::-1], think[::-1], left=plain[::-1], color=C["p_edge"], height=0.6, label="thinking")
for j, (p, t) in enumerate(zip(plain[::-1], think[::-1])):
    b1.text(p + t + 25, j, f"{p + t:,}", va="center", fontsize=12.5, fontweight="bold")
b1.legend(loc="lower right", fontsize=11.5); b1.set_xlim(0, max(a + b for a, b in zip(plain, think)) * 1.22)
b1.set_title(f"a   SFT rows by route ({len(rows):,})", loc="left", fontsize=14, fontweight="bold")
b1.tick_params(axis="y", length=0); b1.spines["left"].set_visible(False)
share = trained / total
b2.pie([1 - share, share], colors=["#D9D9D9", C["g_edge"]], startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white", lw=2))
b2.text(0, 0.06, f"{share:.1%}", ha="center", va="center", fontsize=19, fontweight="bold", color=C["g_text"])
b2.text(0, -0.2, "tokens with loss", ha="center", va="center", fontsize=11.5, color=C["grey"])
b2.set_title("b   loss on the reply only", loc="left", fontsize=14, fontweight="bold")
s = np.array(losses)
b3.plot(np.arange(1, len(s) + 1), s, color=mix(C["p_edge"], 0.6), lw=1.0)
w = 15
b3.plot(np.arange(w, len(s) + 1), np.convolve(s, np.ones(w) / w, mode="valid"), color=C["p_edge"], lw=2.6)
b3.set_xlabel("logged step"); b3.set_ylabel("loss")
b3.set_title(f"c   LoRA v3 training loss (from v2, {len(s)} logged steps)", loc="left", fontsize=14, fontweight="bold")
fig.tight_layout(w_pad=2.5)
save(fig, "chart_training")
print(json.dumps({"tau": dict(zip(map(float, taus), map(lambda v: round(v, 4), pa))), "kinds": kinds, "shapes": {n: (round(sum(by[n]) / len(by[n]), 3), len(by[n])) for n in by},
                  "routes": {f"{a}/{b}": v for (a, b), v in route.items()}, "token_share": round(share, 4), "losses": len(losses)}, default=str))
