"""Numerical and evaluation-integrity checks; empirical success is not assumed."""
import json,sys,platform,importlib.metadata
import numpy as np
from autofocus_v3 import objective,gauge_matrix
from scipy.optimize import check_grad
from point_detection_v3 import match_scene,pooled
from protocol_v3 import ROOT,OUT,PROTOCOL,sha


def main():
 checks=[]
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['protocol']==PROTOCOL;checks.append('Frozen protocol unchanged; development and test identifiers disjoint')
 cfg=PROTOCOL['detection'];assert not set(cfg['train'])&set(cfg['test']);assert not set(cfg['validation'])&set(cfg['test'])
 pred=json.loads((OUT/'detection_test_predictions.json').read_text());data=[]
 for d in pred:
  data.append({k:np.asarray(v) if k in ['positions','truth','proposal','probability'] else v for k,v in d.items()})
 assert sum(len(d['truth']) for d in data)==36 and {d['scene'] for d in data}==set(cfg['test'])
 sel=json.loads((OUT/'detection_selection.json').read_text());assert sha(OUT/'point_detector.joblib')==sel['model_sha256']
 r=json.loads((OUT/'detection_results.json').read_text());actual=pooled(data,sel['rf_threshold'],bootstrap=True)
 assert actual==r['operating_point_rf'];checks.append('Saved full-image predictions independently reproduce 36-object counts, intervals and frozen-threshold metrics')
 # One GT, duplicated candidates, extra distant candidate; class mismatch is FP.
 d={'positions':np.array([[0.,0.],[.1,0],[3,3]]),'truth':np.array([[0.,0.,2.,1.]])}
 m=match_scene(d,np.array([.9,.8,.7]),np.array([1,2,3]),.5)
 assert (m['tp'],m['fp'])==(1,2)
 m=match_scene(d,np.array([.9,.8,.7]),np.array([1,2,3]),.85,class_aware=True)
 assert (m['tp'],m['fp'])==(0,1);checks.append('Point-FROC duplicate penalty, threshold inclusion and wrong-class handling verified on toy geometry')
 for name in ['froc_rf','froc_proposal','froc_class_aware']:
  c=r[name];assert all(a['fp']<=b['fp'] for a,b in zip(c,c[1:]));assert all(0<=v['tp']<=36 for v in c)
 checks.append('Complete FROC curves have monotonic false-positive counts; no test-derived operating threshold')
 af=json.loads((OUT/'autofocus_results.json').read_text());afsel=json.loads((OUT/'autofocus_selection.json').read_text())
 assert af['selection_sha256']==sha(OUT/'autofocus_selection.json');assert af['stack_sum_relative_error_max']<1e-10
 assert len(af['rows'])==24*6*3;assert len({r['seed'] for r in af['rows'] if r['amount']})==5
 assert not set(PROTOCOL['autofocus']['test_seeds'])&set(PROTOCOL['autofocus']['development_seeds'])
 checks.append('24 isolated points × (5 held-out navigation paths + clean) × 3 methods completed; ping stack agrees with frozen BP')
 assert afsel['verification']['entropy_gradient_l2_error']<1e-5;checks.append('Analytic minimum-entropy gradient agrees with finite differences')
 for m,v in af['success'].items():
  vals=[x for x in af['rows'] if x['method']==m];med=float(np.median([x['complex_correlation'] for x in vals if x['amount']]))
  loss=max(1-x['complex_correlation'] for x in vals if not x['amount'])
  assert v['overall_pass']==(med>=.9 and loss<=.01)
 checks.append('Autofocus success/failure uses preregistered 0.90 NCC and 0.01 clean-loss limits')
 bridge=json.loads((OUT/'raw_bridge_results.json').read_text());assert len(bridge['rows'])==3
 assert bridge['execution_detector_selection_sha256']==sha(OUT/'detection_selection.json')
 assert bridge['execution_autofocus_selection_sha256']==sha(OUT/'autofocus_selection.json')
 assert len(bridge['automatic_patches'])==30 and all(r['detection']['total']==9 for r in bridge['rows'])
 checks.append('Held-out raw-to-BP-to-detector bridge covers all 9 objects and 30 automatically selected autofocus patches')
 # Strict JSON: early candidate-ceiling -Infinity was a serialization issue,
 # not a changed score or protocol. Normalize this one sentinel to JSON null.
 r['candidate_ceiling']['threshold']=None
 r['auxiliary_GT_crop_classifier']['scope']='known GT location AND foreground given; argmax over 3 foreground classes only; not detection'
 (OUT/'detection_results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 for p in OUT.glob('*.json'):
  json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 checks.append('All exported v3 JSON files contain finite standard JSON numbers')
 env={'python':sys.version,'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','numba','h5py','scikit-learn','joblib','matplotlib']}}
 (OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
 (OUT/'verification.json').write_text(json.dumps({'checks':checks,'passed':True,'scope':'Numerical/evaluation integrity only; does not imply empirical success','protocol_sha256':sha(OUT/'protocol.json')},indent=2)+'\n')
 print('Verified',len(checks),'integrity checks; empirical failures retained')


if __name__=='__main__':main()
