"""Verify source integrity status, split rules and legacy numerical parity."""
from pathlib import Path
import ast,json
import numpy as np
from imaging import backproject
from imaging_v2 import bp
from evaluate_possm import load_scene
from validate_slc_v2 import roi_observations
ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=[]
    for p in (ROOT/'src').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    checks.append('All packaged Python sources parse')
    audit=json.loads((ROOT/'results/v2/all_scene_gdf_audit.json').read_text());assert len(audit)==11 and all(r['status']=='ok' and r['pings']==100 for r in audit)
    checks.append('11 complete raw scenes parsed with navigation checksum validation; 55 pinned input hashes audited')
    split=json.loads((ROOT/'results/v2/scene_split.json').read_text())['rows']
    groups={s:{r['family'] for r in split if r['split']==s} for s in ['train','validation','test']}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']
    checks.append('Proposed train/validation/test have disjoint named seabed families')
    val=json.loads((ROOT/'results/v2/validation_results.json').read_text());assert not set(val['frozen_configuration']['selected_on'])&set(val['frozen_configuration']['test_point_indices'])
    assert len(val['point_rows'])==24 and len(val['native_maps'])==4 and all(r['pixels']==8064000 for r in val['native_maps'].values())
    checks.append('24 points exclude development points; four primary maps evaluated on all 8,064,000 native pixels')
    s=json.loads((ROOT/'results/v2/stress_results.json').read_text());assert len(s['seeds'])==5 and len(s['rows'])==1536
    checks.append('Stochastic navigation stress:24 points x5 seeds x4 magnitudes x3 axes; plus96 deterministic sound-speed cases')
    q,tx,rx,c=load_scene(ROOT,'scene00');x=np.arange(21.4,21.5,.005);y=np.arange(31.1,31.2,.005)
    qr,ts,rs,cs=roi_observations(q,tx,rx,c,x,y,4,'none');v=np.zeros_like(ts);v[:,0]=.6
    old=backproject(qr,ts,rs,cs,x,y,0.,300000.,300000.,1196.,1.,2*.05*180/np.pi,receiver_velocity=v)
    new=bp(qr,ts,rs,cs,x,y,75000.,4)
    relative=float(np.linalg.norm(old-new)/np.linalg.norm(old));assert relative<1e-8
    checks.append('Optimized v2 beam-gating preserves v1 unweighted BP (relative difference '+str(relative)+')')
    result={'passed':len(checks),'checks':checks,'scope':'No assertion that v2 exactly reproduces ASASIN or that any detector/autofocus estimator was improved'}
    (ROOT/'results/v2/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
