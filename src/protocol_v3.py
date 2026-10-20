"""Write-once evaluation protocol. Run before development and held-out work."""
from pathlib import Path
import hashlib,json,datetime

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v3'
POINTS=[6,8,16,17,20,21,23,25,26,31,33,35,37,38,40,41,42,43,47,48,49,55,56,59]
PROTOCOL={
 'version':3,'data':'PoSSM synthetic, noiseless underwater IQ; no actual mine ground truth',
 'autofocus':{
  'development_points':[1,2,4],'development_seeds':[7001,7002],
  'test_points':POINTS,'test_seeds':[7101,7102,7103,7104,7105],
  'navigation_error':'sway only, 2 mm RMS over 100 pings, low frequency + 0.25 white; IQ unchanged',
  'clean_control':'all 24 points; no navigation error',
  'range_edges_m':[15,35,55,80],
  'primary_success':{'median_unregistered_complex_NCC_min':0.90,'clean_NCC_loss_max':0.01},
  'primary_scope':'blind correction inside GT-selected isolated-point ROI; not full-scene automatic point discovery',
  'gauge':'constant and linear ping phase removed; no truth registration or truth phase used by estimator',
  'methods':['uncorrected','local_gpga','per_ping_entropy'],
  'gpga':'BP contributions; strongest image pixel selected without GT; adjacent-ping conjugate products; iterative phase correction',
  'gpga_iterations_grid':[1,3], 'entropy_regularization_grid':[0.0001,0.001],
  'entropy_maxiter':100,'entropy_starts':['zero','gpga'],
  'selection':'development mean unregistered NCC, subject to clean worst loss <=0.01; fallback best clean loss then corrupted NCC',
  'extra':'common phase pooled across all points vs within fixed range bands is a secondary diagnostic, not used to choose primary methods'
 },
 'detection':{
  'train':[2,3,7,8],'validation':[1,6],'test':[4,5,9,10],
  'classes':{'1':'trap','2':'rock','3':'cylinder'},
  'imagery':'official complex SLC intensity, 5 cm grid; frozen detector also tested on own raw-to-BP reconstruction',
  'roi':'entire geometrically valid SLC area, margin 3 m for features; no GT-limited search windows',
  'proposal':'local log-power contrast at 0.15/0.3/0.6 m; NMS 0.65 m; top 600 per scene; fixed before test',
  'classifier':'RandomForest, 300 trees, class balanced, seed731; no CNN or pretrained model',
  'rf_min_samples_leaf_grid':[2,5],
  'training_labels':'candidate within 1 m of insertion point is positive; >2 m background; middle ignored. GT-jittered train crops permitted only for classifier training',
  'primary_match_radius_m':1.0,'secondary_match_radii_m':[0.5,1.5],
  'matching':'descending score, one-to-one nearest unmatched insertion point; duplicate candidates false positives',
  'operating_point':'validation maximum recall at <=5 false positives per scene, then fewer FP, then higher threshold',
  'report':['point FROC','counts','Wilson 95% interval','scene cluster bootstrap 95% interval','class recall','candidate ceiling','GT-crop classifier auxiliary'],
  'success':'test recall >=0.8 at frozen validation operating threshold, <=5 FP per scene; exploratory small-sample criterion',
  'no_claim':'no bounding-box mAP, no real-mine performance, no independent prototype generalization'
 },
 'provenance':'All thresholds and candidate grids above written before new development/test outputs. Development and validation selections separately hashed before held-out runs.'
}

def write():
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'protocol.json'
 if p.exists():
  old=json.loads(p.read_text());assert old['protocol']==PROTOCOL,'Existing protocol differs; do not overwrite'
  return old
 rec={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':PROTOCOL}
 p.write_text(json.dumps(rec,indent=2)+'\n');return rec

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__=='__main__':write();print('Frozen protocol:',sha(OUT/'protocol.json'))
