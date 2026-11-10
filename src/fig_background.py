"""Background timeline for the README / blog.  python src/fig_background.py -> docs/figures/research_background.png"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"]})
INK, GREY, LIGHT, AMBER, AMBER_BG = "#333333", "#8a8a8a", "#f2f2f2", "#c47f12", "#fbeedb"
ERAS = [
    ("Image formation", "SAS fundamentals", [
        ("2002", "Micronavigation: μm sway", "Bellettini & Pinto"),
        ("2004", "Stop-and-hop breaks down", "Hawkins & Gough"),
        ("2009", "SAS review", "Hayes & Gough")]),
    ("Autofocus", "metric-based", [
        ("1994", "Phase gradient autofocus", "Wahl et al."),
        ("1999", "Entropy minimisation", "Xi, Liu & Ni"),
        ("—", "Stripmap PGA for SAS", "Callow")]),
    ("Deep autofocus", "learning-based", [
        ("2020", "Deep autofocus", "Gerg & Monga"),
        ("2021", "Real-time deep autofocus", "Gerg & Monga"),
        ("2024", "Simulation fidelity", "Blanford & Williams")]),
]
USED = ["receiver-motion backprojection\nsway / surge sensitivity", "local GPGA\nper-ping entropy", "CNN + transformer autofocus\nPoSSM raw IQ, 11 scenes"]

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")
W, X0 = 4.3, [0.1, 4.85, 9.6]
for x, (title, span, items), used in zip(X0, ERAS, USED):
    ax.add_patch(FancyBboxPatch((x, 1.75), W, 4.2, boxstyle="round,pad=0,rounding_size=0.12", fc=LIGHT, ec=INK, lw=1.6))
    ax.text(x + W / 2, 5.55, title, ha="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(x + W / 2, 5.15, span, ha="center", fontsize=13, color=GREY)
    for k, (yr, what, who) in enumerate(items):
        y = 4.45 - k * 0.9
        ax.text(x + 0.25, y, yr, fontsize=13.5, fontweight="bold", color=INK, va="center")
        ax.text(x + 1.05, y + 0.14, what, fontsize=13.5, color=INK, va="center")
        ax.text(x + 1.05, y - 0.22, who, fontsize=12, color=GREY, va="center", style="italic")
    ax.add_patch(FancyBboxPatch((x, 0.2), W, 1.05, boxstyle="round,pad=0,rounding_size=0.12", fc=AMBER_BG, ec=AMBER, lw=1.6))
    ax.text(x + W / 2, 1.0, "used in this project", ha="center", fontsize=12, color=AMBER, style="italic")
    ax.text(x + W / 2, 0.55, used, ha="center", va="center", fontsize=12.5, color=INK, linespacing=1.3)
    ax.annotate("", xy=(x + W / 2, 1.27), xytext=(x + W / 2, 1.73),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.5, mutation_scale=15))
for x in X0[:2]:
    ax.annotate("", xy=(x + W + 0.43, 3.85), xytext=(x + W + 0.02, 3.85),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=16))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures", "research_background.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
