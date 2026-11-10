"""Train the v4 deep autofocus on synthetic phase-only corruption of clean training-scene stacks.

  python src/deep_af_train.py [--steps 4000]
Validation (scene01, scene06; geometric sway, seeds 7001/7002) picks the checkpoint.
Outputs results/v4/deep_af.pt and results/v4/training_log.json.
"""
import argparse, json, time
import numpy as np
import torch
from deep_af_model import DeepAutofocus, d2_loss, n_params
from deep_af_data import registered_ncc
from stress_points_v2 import navigation_path
from protocol_v4 import ROOT, OUT, PROTOCOL, write, sha

CACHE = ROOT / 'data/v4_cache'
LAMBDA = 1500 / 300e3
ALT = 6.0


def load(split):
    items = []
    for scene in PROTOCOL['scene_split'][split]:
        with np.load(CACHE / f'{split}_{scene}.npz') as z:
            d = {k: z[k] for k in z.files}                       # read each array once
        for i in range(len(d['centers'])):
            n = int(d['n_pings'][i]); it = {'scene': scene, 'i': i, 'y': float(d['centers'][i, 1]), 'first': int(d['first_ping'][i]),
                                            'clean': d['clean'][i, :n].copy()}
            for k in d:
                if k.startswith('corrupt_'): it[k] = d[k][i, :n].copy()
            items.append(it)
        del d
    return items


def phase_scale(y): return -4 * np.pi / LAMBDA * y / np.hypot(y, ALT)


def to_batch(stacks):
    B = len(stacks); P = max(len(s) for s in stacks); H, W = stacks[0].shape[1:]
    s = np.zeros((B, P, H, W), np.complex64); m = np.zeros((B, P), bool)
    for i, a in enumerate(stacks):
        r = np.sqrt(np.mean(np.abs(a.sum(0)) ** 2)) + 1e-12
        s[i, :len(a)] = a / r; m[i, :len(a)] = True
    e = (np.abs(s) ** 2).sum((2, 3)); e = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    return torch.from_numpy(s), torch.from_numpy(m), torch.from_numpy(e.astype(np.float32))


def predict(model, stack, iterations=None):
    s, m, _ = to_batch([stack])
    with torch.no_grad(): return model(s, m, iterations)[-1][0].numpy().astype(np.float64)


def validate(model, items, iterations=None):
    out = []
    for it in items:
        clean = it['clean'].sum(0)
        for k in [k for k in it if k.startswith('corrupt_')]:
            ph = predict(model, it[k], iterations)
            out.append(registered_ncc(clean, it[k], np.arange(len(ph)), ph)[0])
    return float(np.median(out)), float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=4000); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--crop', type=int, default=64); ap.add_argument('--eval_every', type=int, default=250); a = ap.parse_args()
    write(); torch.manual_seed(0); rng = np.random.default_rng(0); torch.set_num_threads(2)
    train, val = load('train'), load('validation'); print('train', len(train), 'validation', len(val), flush=True)
    bank = np.stack([navigation_path(20000 + i)[:, 1] for i in range(4000)])
    model = DeepAutofocus(); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
    log = {'protocol_sha256': sha(OUT / 'protocol.json'), 'params': n_params(model), 'args': vars(a), 'evals': []}
    best = -1; t0 = time.perf_counter(); run = []
    for step in range(1, a.steps + 1):
        idx = rng.integers(len(train), size=a.batch); stacks, thetas = [], []
        for j in idx:
            it = train[j]; n = len(it['clean'])
            amp = 0.0 if rng.random() < 0.1 else rng.uniform(0, .003)
            th = phase_scale(it['y']) * amp * bank[rng.integers(len(bank))][it['first']:it['first'] + n]
            y0, x0 = rng.integers(0, 81 - a.crop + 1, size=2)
            stacks.append(it['clean'][:, y0:y0 + a.crop, x0:x0 + a.crop] * np.exp(1j * th)[:, None, None].astype(np.complex64))
            thetas.append(th)
        s, m, e = to_batch(stacks)
        th = torch.zeros(m.shape); [th[i, :len(t)].copy_(torch.from_numpy(t)) for i, t in enumerate(thetas)]
        outs = model(s, m); T = len(outs)
        loss = sum((t + 1) / T * d2_loss(p, th, m, e) for t, p in enumerate(outs)) / sum((t + 1) / T for t in range(T))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run.append(float(loss))
        if step % 50 == 0: print(f'step {step} loss {np.mean(run[-50:]):.4f} {time.perf_counter() - t0:.0f}s', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            model.eval(); med, mean = validate(model, val); model.train()
            log['evals'].append({'step': step, 'train_loss': float(np.mean(run[-a.eval_every:])), 'val_median_registered_ncc': med, 'val_mean': mean,
                                 'seconds': time.perf_counter() - t0})
            print('VAL', log['evals'][-1], flush=True)
            if med > best:
                best = med; torch.save({'state': model.state_dict(), 'step': step, 'val_median': med}, OUT / 'deep_af.pt')
            (OUT / 'training_log.json').write_text(json.dumps(log, indent=1) + '\n')
    print('BEST', best, flush=True)


if __name__ == '__main__':
    main()
