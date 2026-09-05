"""
English-labeled recreation of results/figures/architecture_diagram.png
(Figure 1 in the thesis) for the English translation. No generator script
for the original exists in the repo, so this rebuilds the same box/arrow
layout from scratch with matplotlib, translating only the on-figure text.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch

OUT = Path(__file__).parent / "results" / "figures_en" / "architecture_diagram.png"
OUT.parent.mkdir(exist_ok=True, parents=True)

fig, ax = plt.subplots(figsize=(16.3, 11.05))
ax.set_xlim(0, 100)
ax.set_ylim(0, 68)
ax.axis("off")

GRAY_BORDER = "#4d4d4d"
GRAY_FILL = "#eceff1"
GOLD_BORDER = "#b8860b"
CREAM_FILL = "#fdf3d7"
HEUR_FILL = "#eceff1"
HEUR_BORDER = "#7f8c8d"
IQL_FILL = "#fbe4cd"
IQL_BORDER = "#e67e22"
VDN_FILL = "#dbeaf7"
VDN_BORDER = "#2980b9"


def box(x, y, w, h, fill, border, lw=2.2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=lw, edgecolor=border, facecolor=fill,
        mutation_aspect=1,
    )
    ax.add_patch(p)
    return p


# Title
ax.text(50, 66, "Architecture: from highway-env to the trained agents",
        ha="center", va="top", fontsize=22, fontweight="bold", color="#111111")

# Top box: highway-env
top_x, top_y, top_w, top_h = 3, 51, 94, 10.5
box(top_x, top_y, top_w, top_h, GRAY_FILL, GRAY_BORDER)
ax.text(50, top_y + top_h * 0.62, "highway-env   intersection-v2",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#111111")
ax.text(50, top_y + top_h * 0.27, "kinematic bicycle model   +   IDM/MOBIL (human-driven vehicles)",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#111111")

# Middle box: MultiAgentIntersectionEnv
mid_x, mid_y, mid_w, mid_h = 3, 30.5, 94, 16.5
box(mid_x, mid_y, mid_w, mid_h, CREAM_FILL, GOLD_BORDER)
ax.text(50, mid_y + mid_h - 2.6, "MultiAgentIntersectionEnv   (envs/multi_agent_intersection.py)",
        ha="center", va="center", fontsize=15.5, color="#111111")

bullets = [
    "flatten the Kinematics observation per agent   (obs_dim = 42)",
    "per-agent reward via env.unwrapped._agent_reward(a, v)",
    "+ courtesy penalty (penalty for aggressive driving near human vehicles)",
    "active-masking:  0 reward for an agent that has already finished (the bug fix)",
]
by0 = mid_y + mid_h - 6.6
for i, b in enumerate(bullets):
    ax.text(50, by0 - i * 3.05, "• " + b, ha="center", va="center",
            fontsize=13.5, color="#111111")

# Arrow: middle box up into top box, labeled obs, shared_reward, info
arr1 = FancyArrowPatch((50, mid_y + mid_h), (50, top_y),
                        arrowstyle="-|>", mutation_scale=22, linewidth=1.6, color="#333333")
ax.add_patch(arr1)
ax.text(50, (mid_y + mid_h + top_y) / 2, "obs, shared_reward, info",
        ha="center", va="center", fontsize=13, style="italic", color="#333333",
        bbox=dict(facecolor="white", edgecolor="none", pad=2), zorder=5)

# Bottom boxes: Heuristic / IQL / VDN
bot_y, bot_h = 13, 12.5
gap = 3.2
bot_w = (94 - 2 * gap) / 3
xs = [3, 3 + bot_w + gap, 3 + 2 * (bot_w + gap)]

# Heuristic
box(xs[0], bot_y, bot_w, bot_h, HEUR_FILL, HEUR_BORDER)
ax.text(xs[0] + bot_w / 2, bot_y + bot_h - 2.6, "Heuristic\n(rule-based)",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#111111")
ax.text(xs[0] + bot_w / 2, bot_y + 3.6,
        "static distance\nto the nearest vehicle\n(no learning)",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111111")

# IQL
box(xs[1], bot_y, bot_w, bot_h, IQL_FILL, IQL_BORDER)
ax.text(xs[1] + bot_w / 2, bot_y + bot_h - 2.6, "IQL\n(independent agents)",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#111111")
ax.text(xs[1] + bot_w / 2, bot_y + 3.6,
        "n × DQNAgent\n(Double DQN)\neach learns from its own R_i",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111111")

# VDN
box(xs[2], bot_y, bot_w, bot_h, VDN_FILL, VDN_BORDER)
ax.text(xs[2] + bot_w / 2, bot_y + bot_h - 2.6, "VDN   (CTDE)",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#111111")
ax.text(xs[2] + bot_w / 2, bot_y + 3.6,
        "n × Dueling Q-network\nQ_tot = Σ Q_i,  trained\njointly against Σ R_i",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111111")

# Arrows from bottom boxes up into middle box, with o_i / o_i,R_i labels
label_colors = ["#555555", "#e67e22", "#2980b9"]
labels = ["o_i", "o_i, R_i", "o_i, R_i"]
for x, lbl, col in zip(xs, labels, label_colors):
    cx = x + bot_w / 2
    arr = FancyArrowPatch((cx, bot_y + bot_h), (cx, mid_y),
                           arrowstyle="-|>", mutation_scale=20, linewidth=1.6, color=col)
    ax.add_patch(arr)
    ax.text(cx, bot_y + bot_h + 1.0, lbl, ha="center", va="bottom",
            fontsize=12.5, style="italic", color=col, fontweight="bold")

# Bottom feedback line: vertical lines down from each bottom box, joined by
# a horizontal line, curving up into the left side of the top box
feedback_y = 4.2
for x in xs:
    cx = x + bot_w / 2
    ax.plot([cx, cx], [bot_y, feedback_y], color="#888888", linewidth=1.4)
ax.plot([xs[0] + bot_w / 2, xs[2] + bot_w / 2], [feedback_y, feedback_y],
        color="#888888", linewidth=1.4)

# curved arrow up the left margin into the top box
con = ConnectionPatch(
    (1.2, feedback_y), (1.2, top_y + top_h / 2),
    "data", "data", arrowstyle="-|>", mutation_scale=20,
    linewidth=1.5, color="#888888", connectionstyle="arc3,rad=-0.15",
)
ax.add_patch(con)
ax.plot([xs[0] + bot_w / 2, 1.2], [feedback_y, feedback_y], color="#888888", linewidth=1.4)

ax.text(
    50, 0.9,
    "a = (a_1, ..., a_n)   →   fed back into env.step(a)   "
    "(decentralized execution: each agent decides using only o_i)",
    ha="center", va="bottom", fontsize=13.5, style="italic", color="#333333",
)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
