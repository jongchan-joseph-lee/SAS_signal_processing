"""Validation selection and one-shot test evaluation for the v4 deep autofocus benchmark.

  python src/deep_af_eval.py --stage validation   # picks deep iterations; writes results/v4/selection.json
  python src/deep_af_eval.py --stage test         # writes results/v4/test_results.json (run once)
"""
import argparse, json, time
import numpy as np
import torch
from scipy.optimize import minimize
from autofocus_v3 import gpga, minimum_entropy, objective, gauge_matrix, image
from deep_af_model import DeepAutofocus
from deep_af_data import registered_ncc, ncc, oracle_phase
from deep_af_train import load, predict
from protocol_v4 import OUT, PROTOCOL, sha

ITER_GRID = [2, 4, 6, 8]
RANGE_EDGES = [15, 35, 55, 80]


def entropy_from(stack, start, regularization, maxiter):
    n = len(stack); project = gauge_matrix(n); flat = np.ascontiguousarray(stack.reshape(n, -1))
    flat = flat / max(np.max(np.abs(flat)), 1e-30)
    fit = minimize(objective, start, args=(flat, project, regularization), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 25})
    return project @ fit.x


def run_methods(model, stack, clean_stack, iterations):
    cfg = PROTOCOL['methods']; out = {}; n = len(stack); s128 = stack.astype(np.complex128)
    timed = [('uncorrected', lambda: np.zeros(n)),
             ('local_gpga', lambda: gpga(s128, **cfg['local_gpga'])[1]),
             ('per_ping_entropy', lambda: minimum_entropy(s128, **cfg['per_ping_entropy'])[1]),
             ('deep', lambda: predict(model, stack, iterations))]
    for name, f in timed:
        t = time.perf_counter(); ph = f(); out[name] = (ph, time.perf_counter() - t)
    t = time.perf_counter(); ph = entropy_from(s128, out['deep'][0], **cfg['per_ping_entropy'])
    out['deep_entropy'] = (ph, out['deep'][1] + time.perf_counter() - t)
    out['oracle'] = (oracle_phase(stack, clean_stack), 0.0)
    return out


def evaluate(items, model, iterations, label):
    rows = []; t0 = time.perf_counter()
    for j, it in enumerate(items):
        clean = it['clean']; ref = clean.sum(0); ids = np.arange(len(clean))
        rb = int(np.searchsorted(RANGE_EDGES, it['y'], side='right') - 1)
        cases = [('clean_control', clean)] + [(k, it[k]) for k in sorted(it) if k.startswith('corrupt_')]
        for case, st in cases:
            for m, (ph, sec) in run_methods(model, st, clean, iterations).items():
                rows.append({'scene': it['scene'], 'patch': it['i'], 'y_m': it['y'], 'range_bin': rb, 'pings': len(ids), 'case': case, 'method': m,
                             'registered_ncc': registered_ncc(ref, st, ids, ph)[0], 'ncc': ncc(ref, image(st, ph)), 'seconds': sec})
        if j % 10 == 0: print(label, j, len(items), f'{time.perf_counter() - t0:.0f}s', flush=True)
    return rows


def bootstrap_diff(rows, a, b, metric='registered_ncc', n=2000, seed=0):
    key = lambda r: (r['scene'], r['patch'])
    A = {(*key(r), r['case']): r[metric] for r in rows if r['method'] == a and r['case'] != 'clean_control'}
    B = {(*key(r), r['case']): r[metric] for r in rows if r['method'] == b and r['case'] != 'clean_control'}
    patches = sorted({k[:2] for k in A}); d = {p: [A[k] - B[k] for k in A if k[:2] == p] for p in patches}
    rng = np.random.default_rng(seed); est = float(np.median([x for p in patches for x in d[p]])); boots = []
    for _ in range(n):
        pick = rng.integers(len(patches), size=len(patches)); boots.append(np.median([x for i in pick for x in d[patches[i]]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'median_paired_diff': est, 'ci95': [float(lo), float(hi)], 'patches': len(patches)}


def summarize(rows):
    S = {}
    for m in PROTOCOL['methods']:
        c = [r for r in rows if r['method'] == m and r['case'] != 'clean_control']; k = [r for r in rows if r['method'] == m and r['case'] == 'clean_control']
        q = lambda v: [float(x) for x in np.percentile(v, [25, 50, 75])]
        S[m] = {'cases': len(c), 'registered_ncc_q25_50_75': q([r['registered_ncc'] for r in c]), 'ncc_q25_50_75': q([r['ncc'] for r in c]),
                'clean_max_loss': float(max(1 - r['registered_ncc'] for r in k)), 'seconds_median': float(np.median([r['seconds'] for r in c])),
                'by_scene': {s: float(np.median([r['registered_ncc'] for r in c if r['scene'] == s])) for s in sorted({r['scene'] for r in c})},
                'by_range_bin': {b: float(np.median([r['registered_ncc'] for r in c if r['range_bin'] == b])) for b in sorted({r['range_bin'] for r in c})}}
    return S


def load_model():
    ck = torch.load(OUT / 'deep_af.pt', weights_only=True); m = DeepAutofocus(); m.load_state_dict(ck['state']); m.eval(); torch.set_num_threads(2)
    return m, ck


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['validation', 'test']); a = ap.parse_args()
    model, ck = load_model()
    if a.stage == 'validation':
        val = load('validation'); grid = {}
        for T in ITER_GRID:
            r = [x for x in evaluate(val, model, T, f'val T={T}') if x['method'] == 'deep' and x['case'] != 'clean_control']
            grid[T] = float(np.median([x['registered_ncc'] for x in r]))
        T = max(grid, key=grid.get); rows = evaluate(val, model, T, 'val all')
        sel = {'protocol_sha256': sha(OUT / 'protocol.json'), 'checkpoint_step': ck['step'], 'iterations_grid': grid, 'iterations': T,
               'summary': summarize(rows), 'rows': rows}
        (OUT / 'selection.json').write_text(json.dumps(sel, indent=1) + '\n'); print('SELECTED', T, grid, flush=True)
        for m, s in sel['summary'].items(): print(m, s['registered_ncc_q25_50_75'], s['clean_max_loss'], s['seconds_median'], flush=True)
    else:
        sel = json.loads((OUT / 'selection.json').read_text()); assert sel['protocol_sha256'] == sha(OUT / 'protocol.json')
        path = OUT / 'test_results.json'; assert not path.exists(), 'test already run'
        rows = evaluate(load('test'), model, sel['iterations'], 'test')
        res = {'protocol_sha256': sha(OUT / 'protocol.json'), 'selection_sha256': sha(OUT / 'selection.json'), 'iterations': sel['iterations'],
               'summary': summarize(rows),
               'paired': {'deep_minus_entropy': bootstrap_diff(rows, 'deep', 'per_ping_entropy'),
                          'deep_entropy_minus_entropy': bootstrap_diff(rows, 'deep_entropy', 'per_ping_entropy'),
                          'deep_minus_gpga': bootstrap_diff(rows, 'deep', 'local_gpga')},
               'rows': rows}
        path.write_text(json.dumps(res, indent=1) + '\n')
        for m, s in res['summary'].items(): print(m, s['registered_ncc_q25_50_75'], s['clean_max_loss'], s['seconds_median'], s['by_scene'], flush=True)
        print(res['paired'], flush=True)


if __name__ == '__main__':
    main()
