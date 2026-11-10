"""Aggregate the v5 scene cross-validation and analyse failure patches.

  python src/deep_af_cv_report.py   ->  results/v5/summary.json
"""
import json
import numpy as np
from scipy.stats import spearmanr
from protocol_v5 import ROOT, OUT, FOLDS, sha
from deep_af_cv import load

FAMILY = {'scene01': 'uniform', 'scene06': 'uniform', 'scene02': 'rough', 'scene03': 'rough', 'scene07': 'rough', 'scene08': 'rough',
          'scene04': 'ripple', 'scene09': 'ripple', 'scene05': 'mixed', 'scene10': 'mixed'}
METHODS = ['uncorrected', 'local_gpga', 'per_ping_entropy', 'deep', 'deep_entropy', 'oracle']


def rows_all():
    rows = []
    for f in FOLDS:
        rows += json.loads((OUT / f / 'rows.json').read_text())['rows']
    return rows


def paired(rows, a='deep', b='per_ping_entropy', scenes=None, n=2000, seed=0):
    A = {(r['scene'], r['patch'], r['case']): r['registered_ncc'] for r in rows if r['method'] == a and r['case'] != 'clean_control' and (scenes is None or r['scene'] in scenes)}
    B = {(r['scene'], r['patch'], r['case']): r['registered_ncc'] for r in rows if r['method'] == b and r['case'] != 'clean_control' and (scenes is None or r['scene'] in scenes)}
    d = {}
    for k in A: d.setdefault(k[:2], []).append(A[k] - B[k])
    ps = sorted(d); rng = np.random.default_rng(seed)
    boots = [np.median([v for i in rng.integers(len(ps), size=len(ps)) for v in d[ps[i]]]) for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'median': float(np.median([v for p in ps for v in d[p]])), 'ci95': [float(lo), float(hi)], 'patches': len(ps),
            'patch_median': {f'{p[0]}:{p[1]}': float(np.median(d[p])) for p in ps}}


def med(rows, m, scenes, key='registered_ncc', clean=False):
    v = [r[key] for r in rows if r['method'] == m and r['scene'] in scenes and ((r['case'] == 'clean_control') == clean)]
    return float(np.median(v) if not clean else np.median([1 - x for x in v]))


def features(items):
    """Patch features from the clean ping stack: speckle contrast of the image and energy-weighted adjacent-ping coherence."""
    out = {}
    for it in items:
        s = it['clean']; z = np.abs(s.sum(0)); e = np.sum(np.abs(s) ** 2, axis=(1, 2))
        coh = [abs(np.vdot(s[k], s[k + 1])) / np.sqrt(e[k] * e[k + 1] + 1e-30) for k in range(len(s) - 1)]
        w = np.sqrt(e[:-1] * e[1:])
        out[f'{it["scene"]}:{it["i"]}'] = {'speckle_contrast': float(z.std() / z.mean()), 'ping_coherence': float(np.sum(w * coh) / np.sum(w)),
                                           'pings': len(s), 'y_m': it['y']}
    return out


def main():
    rows = rows_all(); scenes = sorted({r['scene'] for r in rows}); S = {'protocol_sha256': sha(OUT / 'protocol.json')}
    S['by_scene'] = {s: {**{m: med(rows, m, [s]) for m in METHODS}, 'paired_deep_minus_entropy': {k: v for k, v in paired(rows, scenes=[s]).items() if k != 'patch_median'},
                         'clean_loss': {m: med(rows, m, [s], clean=True) for m in ['local_gpga', 'per_ping_entropy', 'deep', 'deep_entropy']},
                         'family': FAMILY[s]} for s in scenes}
    fams = sorted(set(FAMILY.values()))
    S['by_family'] = {f: {**{m: med(rows, m, [s for s in scenes if FAMILY[s] == f]) for m in METHODS},
                          'paired_deep_minus_entropy': {k: v for k, v in paired(rows, scenes=[s for s in scenes if FAMILY[s] == f]).items() if k != 'patch_median'}} for f in fams}
    allp = paired(rows)
    S['overall'] = {**{m: med(rows, m, scenes) for m in METHODS}, 'paired_deep_minus_entropy': {k: v for k, v in allp.items() if k != 'patch_median'},
                    'paired_deep_entropy_minus_entropy': {k: v for k, v in paired(rows, 'deep_entropy').items() if k != 'patch_median'},
                    'clean_loss': {m: med(rows, m, scenes, clean=True) for m in ['local_gpga', 'per_ping_entropy', 'deep', 'deep_entropy']},
                    'seconds': {m: float(np.median([r['seconds'] for r in rows if r['method'] == m and r['case'] != 'clean_control'])) for m in METHODS}}
    # v4 vs v5 on the v4 test scenes
    v4 = json.loads((ROOT / 'results/v4/test_results.json').read_text())['rows']
    S['v4_vs_v5'] = {s: {'v4_deep': med(v4, 'deep', [s]), 'v5_deep': med(rows, 'deep', [s]), 'entropy': med(rows, 'per_ping_entropy', [s]),
                         'v4_paired': {k: v for k, v in paired(v4, scenes=[s]).items() if k != 'patch_median'},
                         'v5_paired': {k: v for k, v in paired(rows, scenes=[s]).items() if k != 'patch_median'},
                         'v4_clean_loss': med(v4, 'deep', [s], clean=True), 'v5_clean_loss': med(rows, 'deep', [s], clean=True)}
                   for s in ['scene04', 'scene05', 'scene09', 'scene10']}
    # failure analysis
    F = features(load('test', scenes)); pm = allp['patch_median']
    keys = sorted(pm); S['features'] = {k: {**F[k], 'delta': pm[k]} for k in keys}
    S['feature_spearman'] = {f: {'rho': float(spearmanr([F[k][f] for k in keys], [pm[k] for k in keys])[0]),
                                 'rho_abs_deep': float(spearmanr([F[k][f] for k in keys],
                                                                 [np.median([r['registered_ncc'] for r in rows if r['method'] == 'deep' and f'{r["scene"]}:{r["patch"]}' == k and r['case'] != 'clean_control']) for k in keys])[0])}
                             for f in ['speckle_contrast', 'ping_coherence', 'pings', 'y_m']}
    S['failures_lt_-0.05'] = {s: int(sum(pm[k] < -0.05 for k in keys if k.startswith(s))) for s in scenes}
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1) + '\n')
    for s, v in S['by_scene'].items():
        p = v['paired_deep_minus_entropy']
        print(s, v['family'], 'deep %.3f ent %.3f gpga %.3f | d %.3f [%.3f, %.3f] | clean deep %.3f ent %.3f' % (v['deep'], v['per_ping_entropy'], v['local_gpga'], p['median'], *p['ci95'], v['clean_loss']['deep'], v['clean_loss']['per_ping_entropy']))
    print('overall', json.dumps(S['overall'], indent=0)); print('v4_vs_v5', json.dumps(S['v4_vs_v5'], indent=0)); print('features', S['feature_spearman']); print('fail', S['failures_lt_-0.05'])


if __name__ == '__main__':
    main()
