"""Frozen development-selected v2 settings; native SLC and subpixel tests."""
from pathlib import Path
import json,time
import h5py,numpy as np
from evaluate_possm import load_scene,reference
from imaging import resample_complex,complex_metrics
from imaging_v2 import bp,psf_metrics,quadratic_peak
from diagnose_slc import native_roi
from diagnose_taper import taper

ROOT=Path(__file__).resolve().parents[1]


def describe(values):
    v=np.asarray(values);return {'mean':float(v.mean()),'median':float(np.median(v)),'q25':float(np.quantile(v,.25)),'q75':float(np.quantile(v,.75)),'min':float(v.min()),'max':float(v.max())}


def roi_observations(q,tx,rx,c,x,y,ratio,window):
    selected=(tx[:,0]>=x.min()-y.max()*np.tan(.05)-.1)&(tx[:,0]<=x.max()+y.max()*np.tan(.05)+.1)
    return resample_complex(taper(q[selected],window),ratio),tx[selected],rx[selected],c[selected]


def main():
    out=ROOT/'results/v2';out.mkdir(parents=True,exist_ok=True)
    prev=json.loads((ROOT/'results/possm/possm_results.json').read_text())
    pts=np.loadtxt(ROOT/'references/Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
    indices=[r['index_0based'] for r in prev['heldout_points']['rows']]
    result={'frozen_configuration':{'selected_on':[1,2,4],'test_point_indices':indices,'window':'hann60','ratio':8,'beam':'Gaussian horizontal Tx/Rx','half_cap_rad':.05,
         'statement':'Empirical development-selected processing, not verified ASASIN configuration. No per-test spatial/phase-ramp fitting.'},'point_rows':[],'native_maps':{}}
    q,tx,rx,c=load_scene(ROOT,'scene00');f=h5py.File(next((ROOT/'data/possm_reference').glob('*scene00*hdf5')));examples={}
    for index in indices:
        point=pts[index];ref,x,y=native_roi(f,point,12)
        q4,ts,rs,cs=roi_observations(q,tx,rx,c,x,y,4,'none')
        q8,_,_,_=roi_observations(q,tx,rx,c,x,y,8,'hann60')
        legacy=bp(q4,ts,rs,cs,x,y,75000.,4)
        updated=bp(q8,ts,rs,cs,x,y,75000.,8,1)
        row={'index':index,'native_legacy':complex_metrics(ref,legacy),'native_v2':complex_metrics(ref,updated),'reference_psf':psf_metrics(ref,x,y)}
        for step in [.005,.001]:
            # Same world lattice for all methods; ROI uses GT to select a point,
            # but the peak estimator is not given GT coordinates.
            n=int(round(.12/step));xx=(np.floor(point[0]/step)+np.arange(-n,n+1))*step;yy=(np.floor(point[1]/step)+np.arange(-n,n+1))*step
            for name,qr,ratio,beam,moving in [('stop',q4,4,0,False),('moving',q4,4,0,True),('v2',q8,8,1,True)]:
                z=bp(qr,ts,rs,cs,xx,yy,75000.,ratio,beam,continuous=moving)
                stats=psf_metrics(z,xx,yy);peak=stats['peak']
                stats.update({'grid_error_m':float(np.hypot(peak['grid_x']-point[0],peak['grid_y']-point[1])),
                    'subpixel_error_m':float(np.hypot(peak['subpixel_x']-point[0],peak['subpixel_y']-point[1]))})
                row[f'{name}_{step:g}']=stats
                if index==indices[0] and step==.001:examples[name+'_dense']=z.astype(np.complex64);examples['dense_x']=xx;examples['dense_y']=yy
        result['point_rows'].append(row)
        if index in indices[:3]:
            for name,z in [('reference',ref),('legacy',legacy),('v2',updated)]:examples[f'p{index}_{name}']=z.astype(np.complex64)
            examples[f'p{index}_x']=x;examples[f'p{index}_y']=y
        del q4,q8
        print('heldout',index,row['native_legacy']['amplitude_correlation'],row['native_v2']['amplitude_correlation'],flush=True)
    result['point_summary']={name:{metric:describe([r[name][metric] for r in result['point_rows']]) for metric in ['complex_correlation','amplitude_correlation','aligned_nrmse']} for name in ['native_legacy','native_v2']}
    result['localization_summary']={name:{metric:describe([r[name][metric] for r in result['point_rows']]) for metric in ['grid_error_m','subpixel_error_m']} for name in ['stop_0.005','moving_0.005','v2_0.005','stop_0.001','moving_0.001','v2_0.001']}
    np.savez_compressed(out/'point_validation_examples.npz',**examples)
    (out/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    f.close()
    # Primary whole-scene comparison at exact stored Offset+i*Step grid.
    for scene in ['scene00','scene01']:
        ref,x,y=reference(next((ROOT/'data/possm_reference').glob(f'*{scene}*hdf5')),stride=1)
        q,tx,rx,c=load_scene(ROOT,scene);preview={'reference':ref[::4,::4],'x':x[::4],'y':y[::4]}
        for name,ratio,window,beam in [('legacy4',4,'none',0),('v2_hann_gauss8',8,'hann60',1)]:
            start=time.perf_counter();qr=resample_complex(taper(q,window),ratio);prep=time.perf_counter()-start
            start=time.perf_counter();z=bp(qr,tx,rx,c,x,y,75000.,ratio,beam)
            result['native_maps'][scene+'_'+name]={'shape':list(z.shape),'pixels':int(z.size),'sample_spacing_m':.025,
                'preparation_seconds':prep,'backprojection_seconds':time.perf_counter()-start,**complex_metrics(ref,z)}
            preview[name]=z[::4,::4].astype(np.complex64)
            print('native',scene,name,result['native_maps'][scene+'_'+name],flush=True);del qr,z
        np.savez_compressed(out/(scene+'_native_map_preview.npz'),**preview)
        (out/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
