"""README figures from the eval results in $P2R_WORK/eval.
usage: python scripts/figures.py [out_dir=figures]"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK
from phone2robot.stats import wilson
from phone2robot.teacher import sim_to_cm

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID, ACCENT, CONTEXT = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#a8a7a2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def pooled(labels):
    k = n = 0
    for lab in labels:
        with open(os.path.join(WORK, "eval", f"{lab}.json")) as f:
            d = json.load(f)
        k, n = k + d["k"], n + d["n"]
    return k, n


# 1. main results: success with 95% interval, same 16 held-out layouts for every row
rows = [("Your hand paths, 71 clips", ["final_human", "final_human_s2"], False),
        ("Hand paths smoothed, 70 clips", ["final_smooth10"], False),
        ("40 cleanest clips", ["final_curated"], False),
        ("Robot-style teacher at your 71 layouts", ["final_scripted", "final_scripted_s2"], False),
        ("10 clips, as filmed", ["final_d10x1"], False),
        ("10 clips, each spread to 7 positions", ["final_d10x7", "final_d10x7_s2"], True)]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for i, (name, labels, hi) in enumerate(rows[::-1]):
    k, n = pooled(labels)
    lo, up = wilson(k, n)
    col = ACCENT if hi else CONTEXT
    ax.plot([100 * lo, 100 * up], [i, i], color=col, lw=2, solid_capstyle="round")
    ax.plot(100 * k / n, i, "o", ms=8, color=col, mec="white", mew=2)
    ax.text(100 * up + 2, i, f"{100 * k / n:.0f}%  ({k}/{n})", va="center", color=INK if hi else MUTED, fontsize=9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlim(0, 100)
ax.set_xlabel("success on 16 held-out bowl positions (%), dot = rate, line = 95% interval")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
fig.savefig(os.path.join(OUT, "results.png"), dpi=150, bbox_inches="tight")

# 2. where it works: success over a grid of bowl positions vs where the training clips started
with open(os.path.join(WORK, "grid_layouts.json")) as f:
    grid = {L["tag"]: L["bowl_xy"] for L in json.load(f)}


def train_starts(ds_dir, demo_dirs):
    """Bowl start of every training demo, in phone-view cm."""
    m = os.path.join(WORK, ds_dir, "manifest.json")
    if os.path.exists(m):
        with open(m) as f:
            files = [e["file"] if isinstance(e, dict) else e for e in json.load(f)["entries"]]
    else:                                               # datasets built before the manifest format changed
        with open(os.path.join(WORK, ds_dir, "our_manifest.json")) as f:
            tags = json.load(f)["tags"]
        files = [next(os.path.join(WORK, d, f"clip_{t}.npz") for d in demo_dirs
                      if os.path.exists(os.path.join(WORK, d, f"clip_{t}.npz"))) for t in tags]
    xy = [json.loads(str(np.load(p, allow_pickle=True)["meta"]))["bowl_xy"] for p in files]
    return np.array([sim_to_cm(np.asarray(v)) for v in xy])


panels = [("grid_n25", "ds_n25", ["demos"], "25 clips, mostly filmed far side"),
          ("grid_human", "ds_h_all", ["demos", "demos_b2"], "71 clips, your hand paths"),
          ("grid_scripted", "ds_c_all", ["demos_cm"], "71 clips, robot-style teacher")]
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
for ax, (lab, ds, dirs, title) in zip(axs, panels):
    with open(os.path.join(WORK, "eval", f"{lab}.json")) as f:
        E = json.load(f)
    by = {}
    for e in E["episodes"]:
        by.setdefault(e["tag"], []).append(e["success"])
    uv = np.array([sim_to_cm(grid[t]) for t in by])
    rate = np.array([np.mean(v) for v in by.values()])
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=rate, cmap="Blues", vmin=0, vmax=1, s=620, marker="s", edgecolors="#c9c8c3", linewidths=0.8)
    tr = train_starts(ds, dirs)
    ax.scatter(tr[:, 0], tr[:, 1], s=16, c=INK, edgecolors="white", linewidths=0.6, label="where a training clip started")
    ax.add_patch(plt.Circle((0, 0), 13, fill=False, lw=2, color=MUTED))
    ax.text(0, 0, "plate", ha="center", va="center", color=MUTED)
    ax.set_title(f"{title}\n{100 * E['k'] / E['n']:.0f}% over the grid", fontsize=10, color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-22, 36)
    ax.set_ylim(-45, 40)
    ax.set_xlabel("cm right of the plate (phone view)")
axs[0].set_ylabel("cm away from the phone (robot side up)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=1, fontsize=9, frameon=False, bbox_to_anchor=(0.45, 0.04))
cb = fig.colorbar(sc, ax=axs, fraction=0.02)
cb.set_label("success rate (2 trials per square)")
fig.savefig(os.path.join(OUT, "coverage_heatmap.png"), dpi=150, bbox_inches="tight")
print("saved", OUT)
