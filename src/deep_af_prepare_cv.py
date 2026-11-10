"""Build caches for the v5 scene cross-validation.

  python src/deep_af_prepare_cv.py --kind train|test [--scenes scene01 scene04 ...]
Writes data/v5_cache/<kind>_<scene>.npz (same format as v4). v4 caches with the
same sampler, count and seeds are linked instead of recomputed.
"""
import argparse, os, time
import numpy as np
from deep_af_data import case
from deep_af_prepare import centers, pad
from evaluate_possm import load_scene
from protocol_v5 import ROOT, PROTOCOL, SCENES, write

V4 = ROOT / 'data/v4_cache'; CACHE = ROOT / 'data/v5_cache'
REUSE = {('train', s): V4 / f'train_{s}.npz' for s in ['scene02', 'scene03', 'scene07', 'scene08']}
REUSE.update({('test', s): V4 / f'test_{s}.npz' for s in ['scene04', 'scene05', 'scene09', 'scene10']})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--kind', required=True, choices=['train', 'test']); ap.add_argument('--scenes', nargs='*', default=SCENES)
    a = ap.parse_args(); write(); CACHE.mkdir(parents=True, exist_ok=True)
    count = PROTOCOL['patches'][f'{a.kind}_per_scene']; seeds = [] if a.kind == 'train' else [7101, 7102, 7103, 7104, 7105]
    for scene in a.scenes:
        path = CACHE / f'{a.kind}_{scene}.npz'
        if path.exists(): print('exists', path.name, flush=True); continue
        if (a.kind, scene) in REUSE:
            os.symlink(REUSE[(a.kind, scene)], path); print('linked', path.name, flush=True); continue
        q, tx, rx, c = load_scene(ROOT, scene); cs = centers(scene, count); t0 = time.perf_counter()
        clean, ids, corrupt = [], [], {s: [] for s in seeds}
        for i, p in enumerate(cs):
            d = case(q, tx, rx, c, p, None, 0, clean=False); clean.append(d['signal']); ids.append(d['ids'])
            assert np.all(np.diff(d['ids']) == 1), (scene, i)
            for s in seeds:
                e = case(q, tx, rx, c, p, s, .002, clean=False); assert np.array_equal(e['ids'], d['ids']); corrupt[s].append(e['signal'])
            if i % 25 == 0: print(scene, a.kind, i, f'{time.perf_counter() - t0:.0f}s', flush=True)
        np.savez(path, centers=cs, clean=pad(clean), n_pings=np.array([len(i) for i in ids]), first_ping=np.array([i[0] for i in ids]),
                 **{f'corrupt_{s}': pad(v) for s, v in corrupt.items()})
        print('saved', path.name, f'{time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
