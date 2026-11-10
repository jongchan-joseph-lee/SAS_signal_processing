"""Write-once protocol for the v5 scene cross-validation of the deep autofocus. Run before any v5 output."""
from pathlib import Path
import datetime, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v5'
SCENES = [f'scene{i:02d}' for i in range(1, 11)]
FOLDS = {'F1': ['scene04', 'scene02'], 'F2': ['scene09', 'scene03'], 'F3': ['scene05', 'scene07'],
         'F4': ['scene10', 'scene08'], 'F5': ['scene01', 'scene06']}
PROTOCOL = {
 'version': 5,
 'question': 'Is the v4 scene04 deficit of the deep autofocus tied to the seafloor type, and does it persist after fixing the train/test iteration mismatch?',
 'why_v5': ['v4 trained with 4 iterations but tested with 8; step embeddings 5-8 were never trained',
            'v4 scene04 (ripple) deficit did not appear on scene09 (ripple); v4 had no ripple scene in training and none left out for checking'],
 'design': 'leave-two-scenes-out cross-validation over the 10 seabed scenes; each scene is tested exactly once by a model that never saw it',
 'folds': FOLDS,
 'fold_rule': 'each ripple and mixed scene is paired with a rough or uniform scene, so every fold trains on at least one ripple and one mixed scene',
 'patches': {'train_per_scene': 150, 'test_per_scene': 25, 'centers': 'same sampler as v4 (rng 4000 + scene number, >= 4 m from objects)'},
 'test_corruption': 'geometric sway injection as v3/v4: 2 mm RMS, seeds 7101..7105, plus clean control',
 'training': {'corruption': 'phase-only on clean stacks, as v4', 'clean_fraction': 0.25, 'iterations_train': 8, 'iterations_test': 8,
              'steps': 2400, 'batch': 12, 'crop': 64, 'optimizer': 'AdamW 1e-3 one-cycle, wd 1e-4', 'seed': 0,
              'checkpoint': 'last step (no validation selection)'},
 'baselines': {'local_gpga': {'iterations': 1}, 'per_ping_entropy': {'regularization': 0.0001, 'maxiter': 100}},
 'metrics': ['registered complex NCC (primary)', 'paired delta = deep - entropy per patch and path', 'clean-control loss', 'estimator seconds'],
 'report': ['per scene and per seafloor family', 'patch-cluster bootstrap 95% CI (2000, seed 0)',
            'patch features vs delta: speckle contrast of the clean image, mean adjacent-ping coherence, ping count, range (Spearman)'],
 'disclosure': 'scenes 04, 05, 09, 10 were the v4 test scenes; the v5 changes (iteration fix, clean fraction) were chosen after seeing v4 test results. No v5 setting is tuned on any v5 test output.',
}


def write():
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / 'protocol.json'
    if p.exists():
        old = json.loads(p.read_text()); assert old['protocol'] == PROTOCOL, 'Existing protocol differs; do not overwrite'
        return old
    rec = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'protocol': PROTOCOL}
    p.write_text(json.dumps(rec, indent=2) + '\n'); return rec


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__': write(); print('Frozen protocol:', sha(OUT / 'protocol.json'))
