"""v5 scene cross-validation of the deep autofocus: train one fold, then test its two held-out scenes.

  python src/deep_af_cv.py --fold F1            # trains results/v5/F1/deep_af.pt, writes results/v5/F1/rows.json
  python src/deep_af_cv.py --fold F1 --eval-only
Settings are fixed in protocol_v5 (no validation selection; last checkpoint).
"""
import argparse, json, time
import numpy as np
import torch
from deep_af_model import DeepAutofocus, d2_loss, n_params
from deep_af_train import to_batch, phase_scale, predict
from deep_af_eval import evaluate
from stress_points_v2 import navigation_path
from protocol_v5 import ROOT, OUT, PROTOCOL, SCENES, FOLDS, write, sha

CACHE = ROOT / 'data/v5_cache'


def load(kind, scenes):
    items = []
    for scene in scenes:
        with np.load(CACHE / f'{kind}_{scene}.npz') as z:
            d = {k: z[k] for k in z.files}
        for i in range(len(d['centers'])):
            n = int(d['n_pings'][i])
            it = {'scene': scene, 'i': i, 'y': float(d['centers'][i, 1]), 'x': float(d['centers'][i, 0]), 'first': int(d['first_ping'][i]),
                  'clean': d['clean'][i, :n].copy()}
            for k in d:
                if k.startswith('corrupt_'): it[k] = d[k][i, :n].copy()
            items.append(it)
        del d
    return items


def train(fold, out):
    T = PROTOCOL['training']; torch.manual_seed(T['seed']); rng = np.random.default_rng(T['seed']); torch.set_num_threads(2)
    scenes = [s for s in SCENES if s not in FOLDS[fold]]; data = load('train', scenes)
    bank = np.stack([navigation_path(20000 + i)[:, 1] for i in range(4000)])
    model = DeepAutofocus(iterations=T['iterations_train'])
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=T['steps'], pct_start=0.05)
    log = {'fold': fold, 'train_scenes': scenes, 'patches': len(data), 'params': n_params(model), 'loss': []}; t0 = time.perf_counter(); run = []
    for step in range(1, T['steps'] + 1):
        stacks, thetas = [], []
        for j in rng.integers(len(data), size=T['batch']):
            it = data[j]; n = len(it['clean'])
            amp = 0.0 if rng.random() < T['clean_fraction'] else rng.uniform(0, .003)
            th = phase_scale(it['y']) * amp * bank[rng.integers(len(bank))][it['first']:it['first'] + n]
            y0, x0 = rng.integers(0, 81 - T['crop'] + 1, size=2)
            stacks.append(it['clean'][:, y0:y0 + T['crop'], x0:x0 + T['crop']] * np.exp(1j * th)[:, None, None].astype(np.complex64)); thetas.append(th)
        s, m, e = to_batch(stacks); th = torch.zeros(m.shape)
        for i, t in enumerate(thetas): th[i, :len(t)] = torch.from_numpy(t)
        outs = model(s, m); K = len(outs)
        loss = sum((k + 1) / K * d2_loss(p, th, m, e) for k, p in enumerate(outs)) / sum((k + 1) / K for k in range(K))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); run.append(float(loss))
        if step % 100 == 0:
            log['loss'].append([step, float(np.mean(run[-100:]))]); print(f'{fold} step {step} loss {np.mean(run[-100:]):.4f} {time.perf_counter() - t0:.0f}s', flush=True)
    log['seconds'] = time.perf_counter() - t0
    torch.save({'state': model.state_dict(), 'step': T['steps']}, out / 'deep_af.pt'); (out / 'train_log.json').write_text(json.dumps(log, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', required=True, choices=list(FOLDS)); ap.add_argument('--eval-only', action='store_true'); a = ap.parse_args()
    write(); out = OUT / a.fold; out.mkdir(parents=True, exist_ok=True)
    if not a.eval_only: train(a.fold, out)
    model = DeepAutofocus(iterations=PROTOCOL['training']['iterations_test']); model.load_state_dict(torch.load(out / 'deep_af.pt', weights_only=True)['state'])
    model.eval(); torch.set_num_threads(2)
    test = load('test', FOLDS[a.fold])
    rows = evaluate(test, model, PROTOCOL['training']['iterations_test'], a.fold)
    xs = {(it['scene'], it['i']): it['x'] for it in test}
    for r in rows: r['x_m'] = xs[(r['scene'], r['patch'])]
    (out / 'rows.json').write_text(json.dumps({'protocol_sha256': sha(OUT / 'protocol.json'), 'fold': a.fold, 'rows': rows}) + '\n')
    print('DONE', a.fold, flush=True)


if __name__ == '__main__':
    main()
