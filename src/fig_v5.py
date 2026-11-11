"""Figure for the v5 scene cross-validation (README / blog).

Two training settings, same folds and test cases: clean fraction 0.25 (frozen v5 protocol) and 0.1 (post-hoc comparison).
  python src/fig_v5.py   ->  docs/figures/deep_autofocus_cv.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from protocol_v5 import ROOT, OUT

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, GREY, MID, LIGHT, AMBER, AMBER_BG = "#333333", "#8a8a8a", "#b5b5b5", "#e6e6e6", "#c47f12", "#f3d9ae"
ORDER = ['scene02', 'scene03', 'scene07', 'scene08', 'scene05', 'scene10', 'scene04', 'scene09', 'scene01', 'scene06']
FAM = ['rough', 'mixed', 'ripple', 'uniform']


def main():
    A = json.loads((OUT / 'summary.json').read_text())                                       # clean fraction 0.25
    B = json.loads((ROOT / 'results/v5_ablation/summary_clean0.1.json').read_text())         # clean fraction 0.1
    fig = plt.figure(figsize=(15.5, 4.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 0.95], wspace=0.3)
    # (a) paired delta per held-out scene, both settings
    ax = fig.add_subplot(gs[0]); x = np.arange(len(ORDER))
    for i, s in enumerate(ORDER):
        for S, dx, col, fill in [(A, -0.17, GREY, 'white'), (B, 0.17, AMBER, AMBER)]:
            p = S['by_scene'][s]['paired_deep_minus_entropy']
            ax.plot([i + dx] * 2, p['ci95'], color=col, lw=2); ax.plot(i + dx, p['median'], 'o', ms=7, mfc=fill, mec=col, mew=1.4)
    ax.axhline(0, color=INK, lw=0.8); xs = 0
    for f in FAM:
        n = sum(A['by_scene'][s]['family'] == f for s in ORDER)
        ax.text(xs + (n - 1) / 2, 0.071, f, ha='center', fontsize=11.5, color=INK, fontweight='bold')
        if xs: ax.axvline(xs - 0.5, color=LIGHT, lw=1)
        xs += n
    ax.set_xticks(x, [s[-2:] for s in ORDER]); ax.set_xlabel('Held-out scene'); ax.set_ylim(-0.09, 0.08)
    ax.set_ylabel('Δ registered NCC (CNN − entropy)')
    ax.set_title('(a) Each scene tested once, never seen in training', fontsize=13, fontweight='bold', loc='left')
    ax.plot([], [], 'o', mfc='white', mec=GREY, label='clean fraction 25%'); ax.plot([], [], 'o', mfc=AMBER, mec=AMBER, label='clean fraction 10%')
    ax.legend(frameon=False, fontsize=10.5, loc='lower left', ncol=2)
    # (b) by family
    ax = fig.add_subplot(gs[1]); w = 0.27
    for j, (lab, vals, col) in enumerate([('Entropy', [A['by_family'][f]['per_ping_entropy'] for f in FAM], GREY),
                                          ('CNN 25%', [A['by_family'][f]['deep'] for f in FAM], AMBER_BG),
                                          ('CNN 10%', [B['by_family'][f]['deep'] for f in FAM], AMBER)]):
        ax.bar(np.arange(4) + (j - 1) * w, vals, w, color=col, ec=INK, lw=0.8, label=lab)
    ax.set_xticks(range(4), FAM); ax.set_ylim(0.83, 1.0); ax.set_ylabel('Median registered NCC')
    ax.legend(frameon=False, fontsize=10, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax.set_title('(b) By seafloor type, 2 mm sway', fontsize=13, fontweight='bold', loc='left')
    # (c) damage to error-free input
    ax = fig.add_subplot(gs[2])
    v4r = json.loads((ROOT / 'results/v4/test_results.json').read_text())['rows']
    v4c = float(np.median([1 - r['registered_ncc'] for r in v4r if r['method'] == 'deep' and r['case'] == 'clean_control']))
    labs = ['GPGA', 'Entropy', 'CNN\nv4', 'CNN\n25%', 'CNN\n10%']
    vals = [A['overall']['clean_loss']['local_gpga'], A['overall']['clean_loss']['per_ping_entropy'], v4c, A['overall']['clean_loss']['deep'], B['overall']['clean_loss']['deep']]
    ax.bar(range(5), vals, 0.62, color=[MID, GREY, LIGHT, AMBER_BG, AMBER], ec=INK, lw=0.8)
    for i, v in enumerate(vals): ax.text(i, v * 1.15, f'{v:.3f}', ha='center', fontsize=10, color=AMBER if i == 4 else INK, fontweight='bold' if i == 4 else None)
    ax.set_yscale('log'); ax.set_ylim(0.001, 0.4); ax.set_xticks(range(5), labs, fontsize=10.5); ax.set_ylabel('1 − NCC on error-free input')
    ax.set_title('(c) Damage to error-free input', fontsize=13, fontweight='bold', loc='left')
    fig.text(0.5, -0.04, 'v5 leave-two-scenes-out CV · 10 scenes × 25 patches × 5 sway paths · clean fraction = share of error-free training samples · '
             'points/bars = median, lines = patch bootstrap 95% CI · CNN v4 = v4 model on its 4 test scenes', ha='center', fontsize=10, color=INK)
    fig.savefig(ROOT / 'docs/figures/deep_autofocus_cv.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    main()
