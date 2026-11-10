"""v5 ablation: one fold retrained with a single training setting changed (post-hoc, reported as ablation, not as a v5 result).

  python src/deep_af_cv_ablation.py --fold F4 --clean-fraction 0.1   ->  results/v5_ablation/F4_clean0.1/
"""
import argparse, copy, json
import torch
import deep_af_cv as cv
import protocol_v5 as p5
from deep_af_model import DeepAutofocus
from deep_af_eval import evaluate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', required=True, choices=list(p5.FOLDS))
    ap.add_argument('--clean-fraction', type=float, required=True); a = ap.parse_args()
    P = copy.deepcopy(p5.PROTOCOL); P['training']['clean_fraction'] = a.clean_fraction; cv.PROTOCOL = P
    out = p5.ROOT / f'results/v5_ablation/{a.fold}_clean{a.clean_fraction}'; out.mkdir(parents=True, exist_ok=True)
    (out / 'ablation.json').write_text(json.dumps({'base_protocol_sha256': p5.sha(p5.OUT / 'protocol.json'), 'fold': a.fold,
                                                   'changed': {'clean_fraction': [p5.PROTOCOL['training']['clean_fraction'], a.clean_fraction]},
                                                   'note': 'post-hoc ablation after v5 test results; everything else as v5'}, indent=1) + '\n')
    cv.train(a.fold, out)
    model = DeepAutofocus(iterations=P['training']['iterations_test']); model.load_state_dict(torch.load(out / 'deep_af.pt', weights_only=True)['state'])
    model.eval(); torch.set_num_threads(2)
    rows = evaluate(cv.load('test', p5.FOLDS[a.fold]), model, P['training']['iterations_test'], a.fold)
    (out / 'rows.json').write_text(json.dumps({'fold': a.fold, 'rows': rows}) + '\n'); print('DONE', flush=True)


if __name__ == '__main__':
    main()
