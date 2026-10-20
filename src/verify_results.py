"""Meaningful regression/integrity checks on the executed experiment."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from focus_learning import PhaseNet,phase_apply


def main():
    root=Path(__file__).resolve().parents[1];checks=[]
    d=json.loads((root/'results'/'possm'/'possm_results.json').read_text())
    rows=d['heldout_points']['rows'];ids=[r['index_0based'] for r in rows]
    assert len(rows)==24 and not set(ids)&set(d['configuration']['development_indices_0based'])
    assert all(r['continuous_receiver']['localization_m']<.008 for r in rows)
    assert d['heldout_points']['continuous_receiver']['localization_m']<d['heldout_points']['stop_and_hop']['localization_m']
    checks.append('Public raw-IQ held-out point localization, dev exclusion, fixed world grid')
    tr=json.loads((root/'results'/'transfer'/'transfer_results.json').read_text())
    pts=np.loadtxt(root/'references'/'Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
    train=pts[tr['train_point_indices_0based']];val=pts[tr['validation_point_indices_0based']]
    assert train[:,0].max()+1.6<val[:,0].min()
    assert tr['training_scene']!=tr['test_scene'] and tr['test_cases']==96 and len(tr['test_manifest'])==24
    checks.append('Fine-tune spatial separation and scene-disjoint test protocol')
    torch.set_num_threads(4);torch.manual_seed(7)
    z=torch.complex(torch.randn(3,64,64),torch.randn(3,64,64));c=torch.tensor([[2.,-1.],[-4.,2.],[0.,0.]])
    recovered=phase_apply(phase_apply(z,c),c,-1)
    assert float((recovered-z).abs().max())<2e-6
    assert torch.allclose(z.abs().square().sum((-2,-1)),phase_apply(z,c).abs().square().sum((-2,-1)),rtol=1e-6)
    checks.append('Known inverse-phase recovery and energy conservation')
    m=PhaseNet();m.load_state_dict(torch.load(root/'results/learning/phase_net.pt',weights_only=True)['state_dict']);m.eval()
    m2=PhaseNet();m2.load_state_dict(torch.load(root/'results/learning/phase_net.pt',weights_only=True)['state_dict']);m2.eval()
    with torch.no_grad():assert torch.equal(m(torch.ones(1,3,64,64)),m2(torch.ones(1,3,64,64)))
    checks.append('Checkpoint reload gives identical CPU inference')
    for item in json.loads((root/'references/download_manifest.json').read_text())['files']:
        p=root/item['path']
        if p.exists():
            assert p.stat().st_size==item['bytes']
            with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    checks.append('All locally available public inputs match pinned SHA-256')
    result={'passed':len(checks),'checks':checks,'note':'No test asserts that the learned model must beat the classical baseline.'}
    (root/'results/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
