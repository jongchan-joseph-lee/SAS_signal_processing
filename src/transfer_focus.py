"""Frozen procedural model, scene00-only fine-tuning, scene01-only test.

Known phase errors are injected into public model-derived SLC patches.
This tests relative phase restoration, not absolute focus of the source SLC.
It is separate from the raw-IQ navigation/sound-speed experiment.
"""
from pathlib import Path
import argparse,json,time
import h5py
import numpy as np
import torch

from focus_learning import PhaseNet,LIMITS,features,phase_apply,blind_entropy,metrics


def read_slc(root,scene):
    with h5py.File(next((root/'data'/'possm_reference').glob(f'*{scene}*hdf5'))) as f:
        z=(f['Data/Real'][:,:,0]+1j*f['Data/Imag'][:,:,0]).T
        offset=f['ImagingGrid/Offset'][0];step=f['ImagingGrid/Step'][0]
        return z,(float(offset[0]),float(offset[1])),(float(step[0]),float(step[1]))


def patch(z,offset,step,xy):
    ix=int(np.rint((xy[0]-offset[0])/step[0]));iy=int(np.rint((xy[1]-offset[1])/step[1]))
    assert 32<=ix<z.shape[1]-32 and 32<=iy<z.shape[0]-32
    return z[iy-32:iy+32,ix-32:ix+32].copy(),[ix-32,ix+32,iy-32,iy+32]


def augment(clean,count,seed):
    rng=np.random.default_rng(seed)
    ids=rng.integers(len(clean),size=count)
    z=torch.from_numpy(clean[ids].copy())
    z=z/z.abs().amax((-2,-1),keepdim=True).clamp_min(1e-12)
    c=torch.from_numpy(rng.uniform(-1,1,(count,2)).astype(np.float32))*LIMITS
    return z,phase_apply(z,c),c


def evaluate(model,z):
    # Feature extraction, CPU forward and correction all included.
    start=time.perf_counter()
    with torch.no_grad():c=model(features(z));restored=phase_apply(z,c,-1)
    return restored,c,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    root=a.root;out=root/'results'/'transfer';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(52)
    source_checkpoint=root/'results'/'learning'/'phase_net.pt'
    base=PhaseNet();base.load_state_dict(torch.load(source_checkpoint,weights_only=True)['state_dict']);base.eval()
    points=np.loadtxt(root/'references'/'Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
    dist=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=-1);np.fill_diagonal(dist,np.inf)
    z0,offset,step=read_slc(root,'scene00')
    # Entire source patch footprints separated along track by >3 metres.
    eligible=(dist.min(1)>2.3)&(points[:,1]>3)&(points[:,1]<87)
    train_ids=np.flatnonzero(eligible&(points[:,0]>7)&(points[:,0]<24))
    val_ids=np.flatnonzero(eligible&(points[:,0]>28)&(points[:,0]<33))
    train_ids=np.array([i for i in train_ids if i not in [1,2,4]])
    val_ids=np.array([i for i in val_ids if i not in [1,2,4]])
    train_patches=np.stack([patch(z0,offset,step,points[i])[0] for i in train_ids]).astype(np.complex64)
    val_patches=np.stack([patch(z0,offset,step,points[i])[0] for i in val_ids]).astype(np.complex64)
    assert len(train_patches)>5 and len(val_patches)>2
    tr_clean,tr_bad,tr_c=augment(train_patches,1024,4401)
    va_clean,va_bad,va_c=augment(val_patches,256,4402)
    tr_x=features(tr_bad);va_x=features(va_bad)
    ft=PhaseNet();ft.load_state_dict(base.state_dict())
    opt=torch.optim.AdamW(ft.parameters(),lr=1e-4,weight_decay=1e-4)
    best=float('inf');best_epoch=0;history=[];start=time.perf_counter()
    for epoch in range(30):
        ft.train();losses=[]
        for ids in torch.randperm(len(tr_x)).split(64):
            opt.zero_grad();loss=((ft(tr_x[ids])-tr_c[ids])/LIMITS).square().mean()
            loss.backward();opt.step();losses.append(float(loss.detach()))
        ft.eval()
        with torch.no_grad():vl=float(((ft(va_x)-va_c)/LIMITS).square().mean())
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':vl})
        if vl<best:
            best=vl;best_epoch=epoch+1
            torch.save({'state_dict':ft.state_dict(),'epoch':best_epoch,'validation_loss':best,
                        'model':'PhaseNet-scene00-ft','source_checkpoint':'../learning/phase_net.pt'},out/'phase_net_finetuned.pt')
        if epoch+1-best_epoch>=8:break
    finetuning_seconds=time.perf_counter()-start
    ft.load_state_dict(torch.load(out/'phase_net_finetuned.pt',weights_only=True)['state_dict']);ft.eval()
    # Only now read scene01 test patches. Nine annotated objects plus a fixed
    # grid of 15 background locations; no performance-based patch filtering.
    z1,offset1,step1=read_slc(root,'scene01')
    objects=np.loadtxt(root/'references'/'Object_Locations_All_Scenes.csv',delimiter=',',skiprows=1)
    locs=[(f'object_{int(r[0])}_type_{int(r[2])}',r[3:5]) for r in objects if r[1]==1]
    locs += [(f'grid_{i}',np.array([x,y])) for i,(x,y) in enumerate(( (x,y) for x in [12.,28.,44.] for y in [12.,26.,44.,62.,78.] ))]
    patches=[];manifest=[]
    for name,xy in locs:
        zz,bounds=patch(z1,offset1,step1,xy);patches.append(zz)
        manifest.append({'name':name,'xy_m':xy.tolist(),'pixel_bounds_xy':bounds})
    patches=np.stack(patches).astype(np.complex64)
    rng=np.random.default_rng(4404)
    clean=torch.from_numpy(np.repeat(patches,4,axis=0))
    clean=clean/clean.abs().amax((-2,-1),keepdim=True).clamp_min(1e-12)
    coeff=torch.from_numpy(rng.uniform(-1,1,(len(clean),2)).astype(np.float32))*LIMITS
    bad=phase_apply(clean,coeff)
    learned,base_c,base_seconds=evaluate(base,bad)
    adapted,ft_c,ft_seconds=evaluate(ft,bad)
    start=time.perf_counter();classic,classic_c=blind_entropy(bad);classic_seconds=time.perf_counter()-start
    arrays={};summary={}
    for name,zz in [('corrupted',bad),('entropy',classic),('procedural_model',learned),('finetuned',adapted)]:
        stats=metrics(clean,zz.detach())
        arrays.update({name+'_'+k:v for k,v in stats.items()})
        summary[name]={k:{'mean':float(v.mean()),'median':float(np.median(v))} for k,v in stats.items()}
    # Already-focused control detects networks that impose unnecessary phase.
    control={}
    for name,model in [('procedural_model',base),('finetuned',ft)]:
        restored,_,_=evaluate(model,clean[::4]);m=metrics(clean[::4],restored)
        control[name]={k:float(v.mean()) for k,v in m.items()}
    restored,_=blind_entropy(clean[::4]);m=metrics(clean[::4],restored)
    control['entropy']={k:float(v.mean()) for k,v in m.items()}
    summary.update({'training_scene':'scene00','train_point_indices_0based':train_ids.tolist(),
        'validation_point_indices_0based':val_ids.tolist(),'train_source_patches':len(train_ids),
        'validation_source_patches':len(val_ids),'train_augmented':1024,'validation_augmented':256,
        'split_seeds':{'train':4401,'validation':4402,'test':4404},
        'test_scene':'scene01','test_source_patches':len(patches),'injections_per_patch':4,'test_cases':len(clean),
        'test_manifest':manifest,'fine_tuning':{'lr':1e-4,'max_epochs':30,'patience':8,'best_epoch':best_epoch,
        'best_validation_loss':best,'seconds':finetuning_seconds,'optimizer':'AdamW; weight decay 1e-4'},
        'runtime_seconds':{'procedural_model_including_features':base_seconds,'finetuned_including_features':ft_seconds,'entropy':classic_seconds},
        'already_focused_control':control,'scope':'Model-derived public SLC with injected quadratic/cubic along-track spectral phase. Single held-out scene; no mine detection or measured underwater accuracy claim.'})
    (out/'transfer_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    np.savez_compressed(out/'transfer_metrics.npz',**arrays,true_coeff=coeff.numpy(),base_coeff=base_c.numpy(),
                        finetuned_coeff=ft_c.numpy(),classic_coeff=classic_c.numpy())
    # First object and first fixed-grid patch, fixed before comparison.
    ids=np.array([0,1,36,37])
    np.savez_compressed(out/'transfer_examples.npz',clean=clean[ids].numpy(),corrupted=bad[ids].numpy(),
        entropy=classic[ids].numpy(),procedural_model=learned[ids].numpy(),finetuned=adapted[ids].numpy(),case_ids=ids)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['test_manifest','train_point_indices_0based','validation_point_indices_0based']},indent=2),flush=True)


if __name__=='__main__':main()
