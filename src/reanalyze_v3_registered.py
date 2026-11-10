"""Re-score the frozen v3 point autofocus test (24 points x 5 sway paths) with linear-phase registered NCC.

Same stacks, methods and selected settings as evaluate_autofocus_v3; only the metric is added.
A blind estimator cannot recover constant or linear ping phase (absolute phase, along-track shift),
so the unregistered NCC also scores an unobservable image shift.
  python src/reanalyze_v3_registered.py   ->  results/v4/v3_registered_reanalysis.json
"""
import json
import numpy as np
from autofocus_v3 import gpga, minimum_entropy, image
from deep_af_data import scene, case, registered_ncc, ncc, oracle_phase
from protocol_v3 import ROOT, PROTOCOL as P3

OUT = ROOT / 'results/v4'


def main():
    sel = json.loads((ROOT / 'results/v3/autofocus_selection.json').read_text())['selected']
    q, tx, rx, c, pts = scene('scene00'); s3 = P3['autofocus']; rows = []
    for idx in s3['test_points']:
        c0 = case(q, tx, rx, c, pts[idx], None, 0)
        for seed in s3['test_seeds']:
            d = case(q, tx, rx, c, pts[idx], seed, .002, clean=False); st = d['signal']
            assert np.array_equal(d['ids'], c0['ids'])
            bin_ = int(np.searchsorted(s3['range_edges_m'], pts[idx, 1], side='right') - 1)
            phases = {'uncorrected': np.zeros(len(st)), 'local_gpga': gpga(st, **sel['local_gpga'])[1],
                      'per_ping_entropy': minimum_entropy(st, **sel['per_ping_entropy'])[1], 'oracle': oracle_phase(st, c0['signal'])}
            for m, ph in phases.items():
                rows.append({'point': idx, 'seed': seed, 'range_m': float(pts[idx, 1]), 'range_bin': bin_, 'method': m,
                             'ncc': ncc(c0['clean'], image(st, ph)), 'registered_ncc': registered_ncc(c0['clean'], st, d['ids'], ph)[0]})
        print('point', idx, flush=True)
    summ = {m: {k: {str(b): float(np.median([r[k] for r in rows if r['method'] == m and (b == 'all' or r['range_bin'] == b)])) for b in ['all', 0, 1, 2]}
                for k in ['ncc', 'registered_ncc']} for m in ['uncorrected', 'local_gpga', 'per_ping_entropy', 'oracle']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'v3_registered_reanalysis.json').write_text(json.dumps({'note': __doc__, 'summary': summ, 'rows': rows}, indent=1) + '\n')
    for m, s in summ.items(): print(m, s, flush=True)


if __name__ == '__main__':
    main()
