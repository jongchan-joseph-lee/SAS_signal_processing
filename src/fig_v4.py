"""Figures for the v4 results (README / blog).

  python src/fig_v4.py
    docs/figures/registered_ncc_reanalysis.png   v3 point test re-scored with registered NCC
    docs/figures/deep_autofocus_results.png      seabed test: accuracy, per scene, accuracy vs time
    docs/figures/deep_autofocus_example.png      one test patch: reference, sway, entropy, deep
    docs/figures/deep_autofocus_model.png        model schematic
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from protocol_v4 import ROOT, OUT

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, GREY, MID, LIGHT, AMBER, AMBER_BG = "#333333", "#8a8a8a", "#b5b5b5", "#e6e6e6", "#c47f12", "#fbeedb"
FIG = ROOT / 'docs/figures'
M = ['uncorrected', 'local_gpga', 'per_ping_entropy', 'deep', 'deep_entropy', 'oracle']
LAB = {'uncorrected': 'No correction', 'local_gpga': 'GPGA', 'per_ping_entropy': 'Entropy', 'deep': 'Deep (CNN)',
       'deep_entropy': 'Deep → entropy', 'oracle': 'True phase'}
COL = {'uncorrected': LIGHT, 'local_gpga': MID, 'per_ping_entropy': GREY, 'deep': AMBER, 'deep_entropy': AMBER_BG, 'oracle': 'white'}


def reanalysis():
    s = json.loads((OUT / 'v3_registered_reanalysis.json').read_text())['summary']
    ms = ['uncorrected', 'local_gpga', 'per_ping_entropy', 'oracle']; bins = ['0', '1', '2']; bl = ['15–35 m', '35–55 m', '55–80 m']
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True); w = 0.2
    for ax, key, title in zip(axs, ['ncc', 'registered_ncc'], ['(a) v3 metric: unregistered NCC', '(b) Along-track shift aligned (registered NCC)']):
        for j, m in enumerate(ms):
            v = [s[m][key][b] for b in bins]
            ax.bar(np.arange(3) + (j - 1.5) * w, v, w, color=COL[m], ec=INK, lw=0.8, label=LAB[m])
        ax.set_xticks(range(3), bl); ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
        ax.axhline(0.9, color=AMBER, ls=':', lw=1.2)
    h, l = axs[0].get_legend_handles_labels(); h.append(plt.Line2D([], [], color=AMBER, ls=':', lw=1.2)); l.append('v3 target 0.90')
    axs[0].set_ylabel('Median complex NCC vs reference'); axs[0].legend(h, l, frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(1.05, 1.22), fontsize=11.5)
    fig.text(0.5, -0.03, 'scene00 isolated points · 24 test points × 5 sway paths (2 mm RMS) · same estimates, only the metric differs',
             ha='center', fontsize=11, color=INK)
    fig.savefig(FIG / 'registered_ncc_reanalysis.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def results():
    r = json.loads((OUT / 'test_results.json').read_text()); rows = r['rows']; S = r['summary']
    fig = plt.figure(figsize=(15, 4.8)); gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    for i, m in enumerate(M):
        v = [x['registered_ncc'] for x in rows if x['method'] == m and x['case'] != 'clean_control']
        b = ax.boxplot(v, positions=[i], widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.8),
                       whiskerprops=dict(color=INK), capprops=dict(color=INK))
        b['boxes'][0].set(facecolor=COL[m], edgecolor=INK)
        ax.text(i, 1.035, f'{np.median(v):.3f}', ha='center', va='bottom', fontsize=11, color=AMBER if m == 'deep' else INK, fontweight='bold')
    XL = {'uncorrected': 'No\ncorrection', 'local_gpga': 'GPGA', 'per_ping_entropy': 'Entropy', 'deep': 'Deep\n(CNN)', 'deep_entropy': 'Deep\n→ entropy', 'oracle': 'True\nphase'}
    ax.set_xticks(range(len(M)), [XL[m] for m in M], fontsize=11); ax.set_ylim(0.3, 1.07)
    ax.set_ylabel('Registered NCC'); ax.set_title('(a) Test scenes, 2 mm sway · 500 cases (median on top)', fontsize=13, fontweight='bold', loc='left')
    ax = fig.add_subplot(gs[1]); sc = ['scene04', 'scene05', 'scene09', 'scene10']; fam = ['ripple', 'mixed', 'ripple', 'mixed']; w = 0.26
    for j, m in enumerate(['per_ping_entropy', 'deep', 'deep_entropy']):
        ax.bar(np.arange(4) + (j - 1) * w, [S[m]['by_scene'][s] for s in sc], w, color=COL[m], ec=INK, lw=0.8, label=LAB[m])
    ax.set_xticks(range(4), [f'{s[-2:]}\n{f}' for s, f in zip(sc, fam)]); ax.set_ylim(0.8, 1.0); ax.set_ylabel('Median registered NCC')
    ax.legend(frameon=False, fontsize=10.5, loc='upper left', ncol=1); ax.set_title('(b) By test scene', fontsize=13, fontweight='bold', loc='left')
    ax = fig.add_subplot(gs[2])
    for m in ['local_gpga', 'per_ping_entropy', 'deep', 'deep_entropy']:
        ax.scatter(S[m]['seconds_median'] * 1000, S[m]['registered_ncc_q25_50_75'][1], s=130, color=COL[m], ec=INK, zorder=3)
        off, ha = {'local_gpga': ((9, 0), 'left'), 'per_ping_entropy': ((8, 12), 'left'), 'deep': ((8, -14), 'left'), 'deep_entropy': ((-10, 14), 'right')}[m]
        ax.annotate(LAB[m], (S[m]['seconds_median'] * 1000, S[m]['registered_ncc_q25_50_75'][1]), xytext=off, textcoords='offset points', fontsize=11, va='center', ha=ha)
    ax.set_xscale('log'); ax.set_xlim(0.3, 3000); ax.set_ylim(0.7, 1.0)
    ax.axhline(S['oracle']['registered_ncc_q25_50_75'][1], color=GREY, ls='--', lw=1); ax.text(0.35, 0.985, 'true phase', color=GREY, fontsize=10.5)
    ax.set_xlabel('Estimator time per patch (ms, CPU, log)'); ax.set_ylabel('Median registered NCC')
    ax.set_title('(c) Accuracy vs time', fontsize=13, fontweight='bold', loc='left')
    fig.savefig(FIG / 'deep_autofocus_results.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def example():
    from autofocus_v3 import minimum_entropy, image
    from deep_af_eval import load_model
    from deep_af_train import load, predict
    from deep_af_data import registered_ncc
    r = json.loads((OUT / 'test_results.json').read_text())['rows']; it_ = json.loads((OUT / 'selection.json').read_text())['iterations']
    ent = [x for x in r if x['method'] == 'per_ping_entropy' and x['case'] == 'corrupt_7101' and x['scene'] == 'scene05']
    med = np.median([x['registered_ncc'] for x in ent]); pick = min(ent, key=lambda x: abs(x['registered_ncc'] - med))
    it = [x for x in load('test') if x['scene'] == 'scene05' and x['i'] == pick['patch']][0]
    model, _ = load_model(); st = it['corrupt_7101']; ref = it['clean'].sum(0); ids = np.arange(len(st))
    cases = [('Reference (true track)', ref, None), ('2 mm sway, no correction', st.sum(0), np.zeros(len(st)))]
    pe = minimum_entropy(st.astype(np.complex128), 1e-4, 100)[1]; pd = predict(model, st, it_)
    cases += [('Entropy', image(st, pe), pe), ('Deep (CNN)', image(st, pd), pd)]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.3)); vmax = 20 * np.log10(np.abs(ref).max())
    for ax, (t, z, ph) in zip(axs, cases):
        ax.imshow(20 * np.log10(np.abs(z) + 1e-12), cmap='gray', vmin=vmax - 35, vmax=vmax, origin='lower', extent=[-0.2, 0.2, -0.2, 0.2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=13, fontweight='bold')
        if ph is not None: ax.text(0.5, -0.07, f'registered NCC {registered_ncc(ref, st, ids, ph)[0]:.3f}', transform=ax.transAxes, ha='center', va='top', fontsize=11.5)
    axs[0].text(0.5, -0.07, f'scene05 seabed · y = {it["y"]:.0f} m · 0.4 m × 0.4 m', transform=axs[0].transAxes, ha='center', va='top', fontsize=11.5)
    fig.savefig(FIG / 'deep_autofocus_example.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def schematic():
    fig, ax = plt.subplots(figsize=(14, 4.0)); ax.set_xlim(0, 14); ax.set_ylim(-0.5, 3.6); ax.axis('off')
    box = lambda x, w, t, s, fc=LIGHT: (ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.7, boxstyle='round,pad=0,rounding_size=0.1', fc=fc, ec=INK, lw=1.4)),
                                        ax.text(x + w / 2, 2.15, t, ha='center', va='center', fontsize=13.5, fontweight='bold', color=INK),
                                        ax.text(x + w / 2, 1.5, s, ha='center', va='center', fontsize=11.5, color=INK))
    arrow = lambda x0, x1: ax.add_patch(FancyArrowPatch((x0, 1.85), (x1, 1.85), arrowstyle='-|>', mutation_scale=16, color=INK, lw=1.4))
    box(0.1, 2.3, 'Ping stack', 'P pings × 81 × 81\n(complex BP)')
    box(3.0, 2.6, 'Per-ping maps', 'ping vs other pings\nphase · amplitude')
    box(6.2, 2.3, 'CNN', 'shared encoder\n(each ping)', AMBER_BG)
    box(9.1, 2.3, 'Transformer', 'mix across pings', AMBER_BG)
    box(12.0, 1.9, 'Phase', 'update per ping')
    for a, b in [(2.4, 3.0), (5.6, 6.2), (8.5, 9.1), (11.4, 12.0)]: arrow(a, b)
    ax.add_patch(FancyArrowPatch((12.95, 1.0), (4.3, 1.0), connectionstyle='arc3,rad=-0.25', arrowstyle='-|>', mutation_scale=16, color=AMBER, lw=1.6))
    ax.text(8.6, -0.35, 'apply correction and repeat (8 times)', ha='center', fontsize=12, color=AMBER)
    ax.text(7, 3.25, 'Trained on synthetic per-ping phase errors · tested on track sway injection', ha='center', fontsize=12, color=INK)
    fig.savefig(FIG / 'deep_autofocus_model.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    reanalysis(); results(); example(); schematic(); print('figures written')
