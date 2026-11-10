"""Figure for the v5 scene cross-validation (README / blog).

  python src/fig_v5.py   ->  docs/figures/deep_autofocus_cv.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from protocol_v5 import ROOT, OUT

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, GREY, MID, LIGHT, AMBER, AMBER_BG = "#333333", "#8a8a8a", "#b5b5b5", "#e6e6e6", "#c47f12", "#fbeedb"
ORDER = ['scene02', 'scene03', 'scene07', 'scene08', 'scene01', 'scene06', 'scene05', 'scene10', 'scene04', 'scene09']
FAM = ['rough', 'uniform', 'mixed', 'ripple']


def main():
    S = json.loads((OUT / 'summary.json').read_text()); B = S['by_scene']
    fig = plt.figure(figsize=(15.5, 4.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.3)
    # (a) paired delta per held-out scene
    ax = fig.add_subplot(gs[0]); x = np.arange(len(ORDER))
    for i, s in enumerate(ORDER):
        p = B[s]['paired_deep_minus_entropy']; neg = p['ci95'][1] < 0
        ax.plot([i, i], p['ci95'], color=AMBER if neg else GREY, lw=2.2)
        ax.plot(i, p['median'], 'o', ms=8, mfc=AMBER if neg else 'white', mec=AMBER if neg else INK, mew=1.4)
    ax.axhline(0, color=INK, lw=0.8)
    xs = 0
    for f in FAM:
        n = sum(B[s]['family'] == f for s in ORDER)
        ax.text(xs + (n - 1) / 2, 0.052, f, ha='center', fontsize=11.5, color=INK, fontweight='bold')
        if xs: ax.axvline(xs - 0.5, color=LIGHT, lw=1)
        xs += n
    ax.set_xticks(x, [s[-2:] for s in ORDER]); ax.set_xlabel('Held-out scene'); ax.set_ylim(-0.075, 0.062)
    ax.set_ylabel('Δ registered NCC (CNN − entropy)')
    ax.set_title('(a) Each scene tested once, never seen in training', fontsize=13, fontweight='bold', loc='left')
    ax.plot([], [], 'o', mfc=AMBER, mec=AMBER, label='CNN lower (95% CI < 0)'); ax.plot([], [], 'o', mfc='white', mec=INK, label='not distinguishable')
    ax.legend(frameon=False, fontsize=10.5, loc='lower left')
    # (b) by family: NCC and clean-input loss
    ax = fig.add_subplot(gs[1]); F = S['by_family']; w = 0.36
    ax.bar(np.arange(4) - w / 2, [F[f]['per_ping_entropy'] for f in FAM], w, color=GREY, ec=INK, lw=0.8, label='Entropy')
    ax.bar(np.arange(4) + w / 2, [F[f]['deep'] for f in FAM], w, color=AMBER, ec=INK, lw=0.8, label='CNN')
    for i, f in enumerate(FAM):
        ax.text(i + w / 2, F[f]['deep'] + 0.003, f"{F[f]['deep']:.3f}", ha='center', fontsize=10, color=AMBER, fontweight='bold')
        ax.text(i - w / 2, F[f]['per_ping_entropy'] + 0.003, f"{F[f]['per_ping_entropy']:.3f}", ha='center', fontsize=10, color=INK)
    ax.set_xticks(range(4), FAM); ax.set_ylim(0.83, 1.0); ax.set_ylabel('Median registered NCC')
    ax.legend(frameon=False, fontsize=10.5, loc='upper right', ncol=2)
    ax.set_title('(b) By seafloor type, 2 mm sway', fontsize=13, fontweight='bold', loc='left')
    # (c) clean-input loss
    ax = fig.add_subplot(gs[2]); C = S['overall']['clean_loss']; V = S['v4_vs_v5']
    labs = ['GPGA', 'Entropy', 'CNN v4', 'CNN v5']
    v4r = json.loads((ROOT / 'results/v4/test_results.json').read_text())['rows']
    v4c = float(np.median([1 - r['registered_ncc'] for r in v4r if r['method'] == 'deep' and r['case'] == 'clean_control']))
    vals = [C['local_gpga'], C['per_ping_entropy'], v4c, C['deep']]
    ax.bar(range(4), vals, 0.6, color=[MID, GREY, AMBER_BG, AMBER], ec=INK, lw=0.8)
    for i, v in enumerate(vals): ax.text(i, v * 1.15, f'{v:.3f}', ha='center', fontsize=10.5, color=AMBER if i == 3 else INK, fontweight='bold' if i == 3 else None)
    ax.set_yscale('log'); ax.set_ylim(0.001, 0.4); ax.set_xticks(range(4), labs); ax.set_ylabel('1 − NCC on error-free input')
    ax.set_title('(c) Damage to error-free input', fontsize=13, fontweight='bold', loc='left')
    fig.text(0.5, -0.04, 'v5 leave-two-scenes-out CV · 10 scenes × 25 patches × 5 sway paths · bars/points = median, lines = patch bootstrap 95% CI · '
             'CNN v4 = v4 model on its 4 test scenes', ha='center', fontsize=10.5, color=INK)
    fig.savefig(ROOT / 'docs/figures/deep_autofocus_cv.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    main()
