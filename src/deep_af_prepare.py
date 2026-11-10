"""Build seabed ping-stack caches for the v4 deep autofocus benchmark.

  python src/deep_af_prepare.py --split train|validation|test
Writes data/v4_cache/<split>_<scene>.npz. Clean stacks for every patch; for
validation/test also the geometrically sway-corrupted stacks (v3 injection).
"""
import argparse, time
import numpy as np
from deep_af_data import case
from evaluate_possm import load_scene
from protocol_v4 import ROOT, PROTOCOL, write

CACHE = ROOT / 'data/v4_cache'


def centers(scene, n):
    num = int(scene[-2:]); rng = np.random.default_rng(4000 + num); P = PROTOCOL['patch']
    obj = np.loadtxt(ROOT / 'references/Object_Locations_All_Scenes.csv', delimiter=',', skiprows=1)
    obj = obj[obj[:, 1] == num][:, 3:5]; out = []
    while len(out) < n:
        p = np.array([rng.uniform(*P['center_x_m']), rng.uniform(*P['center_y_m'])])
        if len(obj) and np.min(np.hypot(*(obj - p).T)) < P['min_distance_to_object_m']: continue
        if out and np.min(np.hypot(*(np.array(out) - p).T)) < 0.5: continue
        out.append(p)
    return np.array(out)


def pad(stacks):
    m = max(len(s) for s in stacks)
    a = np.zeros((len(stacks), m, 81, 81), np.complex64)
    for i, s in enumerate(stacks): a[i, :len(s)] = s
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', required=True, choices=['train', 'validation', 'test']); a = ap.parse_args()
    write(); CACHE.mkdir(parents=True, exist_ok=True)
    count = PROTOCOL['patch']['count_per_scene'][a.split]
    seeds = {'train': [], 'validation': [7001, 7002], 'test': [7101, 7102, 7103, 7104, 7105]}[a.split]
    for scene in PROTOCOL['scene_split'][a.split]:
        path = CACHE / f'{a.split}_{scene}.npz'
        if path.exists(): print('exists', path.name, flush=True); continue
        q, tx, rx, c = load_scene(ROOT, scene); cs = centers(scene, count); t0 = time.perf_counter()
        clean, ids, corrupt = [], [], {s: [] for s in seeds}
        for i, p in enumerate(cs):
            d = case(q, tx, rx, c, p, None, 0, clean=False); clean.append(d['signal']); ids.append(d['ids'])
            assert np.all(np.diff(d['ids']) == 1), (scene, i, d['ids'])
            for s in seeds:
                e = case(q, tx, rx, c, p, s, .002, clean=False)
                assert np.array_equal(e['ids'], d['ids']), (scene, i, s)
                corrupt[s].append(e['signal'])
            if i % 10 == 0: print(scene, a.split, i, f'{time.perf_counter() - t0:.0f}s', flush=True)
        n = np.array([len(i) for i in ids]); first = np.array([i[0] for i in ids])
        extra = {f'corrupt_{s}': pad(v) for s, v in corrupt.items()}
        np.savez(path, centers=cs, clean=pad(clean), n_pings=n, first_ping=first, **extra)
        print('saved', path.name, pad(clean).shape, f'{time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
