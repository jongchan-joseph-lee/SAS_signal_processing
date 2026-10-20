"""Frozen-configuration PoSSM beamforming and controlled navigation study."""
from pathlib import Path
import argparse
import json
import time

import h5py
import numpy as np

from imaging import backproject, resample_complex, complex_metrics

FOV=2*.05*180/np.pi
DEV=[1,2,4]  # 0-based indices chosen before held-out evaluation


def reference(path,stride=4):
    with h5py.File(path) as f:
        z=f['Data/Real'][::stride,::stride,0]+1j*f['Data/Imag'][::stride,::stride,0]
        offset=f['ImagingGrid/Offset'][0]
        step=f['ImagingGrid/Step'][0]
        x=float(offset[0])+np.arange(z.shape[0])*float(step[0])*stride
        y=float(offset[1])+np.arange(z.shape[1])*float(step[1])*stride
    return z.T,x,y


def load_scene(root,scene):
    cache=root/'data'/f'{scene}_cache'
    q=np.load(cache/'range_compressed.npy',mmap_mode='r')
    g=np.load(cache/'geometry.npz')
    return q,g['tx'],g['rx'],g['speeds']


def observations(q,tx,rx,c,x,y,ratio=4):
    # Conservative beam-overlap subset, followed by the same per-pixel gate.
    overlap=(tx[:,0]>=x.min()-y.max()*np.tan(.05)-.1)&(tx[:,0]<=x.max()+y.max()*np.tan(.05)+.1)
    return resample_complex(np.asarray(q[overlap]),ratio),tx[overlap],rx[overlap],c[overlap],np.nonzero(overlap)[0]


def form(q,tx,rx,c,x,y,moving=True):
    v=np.zeros_like(tx);v[:,0]=.6
    return backproject(q,tx,rx,c,x,y,0.,300000.,300000.,1196.,1.,FOV,
                       receiver_velocity=v if moving else None)


def peak_stats(z,x,y,truth):
    iy,ix=np.unravel_index(np.abs(z).argmax(),z.shape)
    amplitude=np.abs(z)
    threshold=amplitude[iy,ix]/np.sqrt(2)
    def width(line,index,grid):
        lo=hi=index
        while lo>0 and line[lo-1]>=threshold:lo-=1
        while hi<len(line)-1 and line[hi+1]>=threshold:hi+=1
        return float(grid[hi]-grid[lo]+np.median(np.diff(grid)))
    return {'peak_x':float(x[ix]),'peak_y':float(y[iy]),
            'dx_m':float(x[ix]-truth[0]),'dy_m':float(y[iy]-truth[1]),
            'localization_m':float(np.hypot(x[ix]-truth[0],y[iy]-truth[1])),
            'peak_amplitude':float(amplitude[iy,ix]),
            'width_3db_x_m':width(amplitude[iy],ix,x),
            'width_3db_y_m':width(amplitude[:,ix],iy,y)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--reuse-maps',action='store_true');a=p.parse_args()
    out=a.root/'results'/'possm';out.mkdir(parents=True,exist_ok=True)
    summary={'configuration':{'compression':'replica correlation','offset_samples_input_rate':299,
        'phase_sign':1,'resample_ratio':4,'half_aperture_radians':.05,
        'motion':'receiver moves at 0.6 m/s during acoustic flight; 3 fixed-point iterations',
        'reference_stride':4,'map_grid_m':.1,'development_indices_0based':DEV}}
    for scene in ['scene00','scene01']:
        if a.reuse_maps and (out/'possm_results.json').exists():
            summary[scene]=json.loads((out/'possm_results.json').read_text())[scene]
            continue
        refpath=next((a.root/'data'/'possm_reference').glob(f'*{scene}*hdf5'))
        ref,x,y=reference(refpath)
        q,tx,rx,c=load_scene(a.root,scene)
        # Resampling all observations is ~1.3 GB; one scene at a time.
        t=time.perf_counter();qr=resample_complex(q,4);resampling=time.perf_counter()-t
        t=time.perf_counter();z=form(qr,tx,rx,c,x,y);seconds=time.perf_counter()-t
        summary[scene]={'map':complex_metrics(ref,z),'resampling_seconds':resampling,
                        'bp_seconds_including_possible_jit':seconds,'shape':list(z.shape)}
        np.savez_compressed(out/f'{scene}_map.npz',image=z.astype(np.complex64),reference=ref,x=x,y=y)
        print(scene,summary[scene],flush=True)
        del qr
    # Held-out point locations, selected solely by geometry, before reading IQ.
    points=np.loadtxt(a.root/'references'/'Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
    distance=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=-1);np.fill_diagonal(distance,np.inf)
    candidates=np.flatnonzero((distance.min(1)>1)&(points[:,0]>10)&(points[:,0]<50)&(points[:,1]>15)&(points[:,1]<80))
    candidates=np.array([i for i in candidates if i not in DEV])[:24]
    assert len(candidates)==24
    q,tx,rx,c=load_scene(a.root,'scene00');rows=[];examples=[]
    for index in candidates:
        point=points[index]
        # Fixed world-coordinate grid, independent of the truth's subpixel offset.
        x=np.arange(np.floor(point[0]/.005)-40,np.floor(point[0]/.005)+41)*.005
        y=np.arange(np.floor(point[1]/.005)-40,np.floor(point[1]/.005)+41)*.005
        qs,ts,rs,cs,_=observations(q,tx,rx,c,x,y)
        stop=form(qs,ts,rs,cs,x,y,False);moving=form(qs,ts,rs,cs,x,y,True)
        row={'index_0based':int(index),'truth':point.tolist(),'stop_and_hop':peak_stats(stop,x,y,point),
             'continuous_receiver':peak_stats(moving,x,y,point)}
        rows.append(row)
        if len(examples)<4:examples.append((x,y,stop,moving))
    summary['heldout_points']={'count':len(rows),'selection':'first 24 CSV rows after excluding development; NN>1m, 10<x<50,15<y<80',
        'grid_m':.005,'roi_width_m':.4,'rows':rows}
    for label in ['stop_and_hop','continuous_receiver']:
        summary['heldout_points'][label]={k:float(np.median([r[label][k] for r in rows]))
            for k in ['localization_m','width_3db_x_m','width_3db_y_m']}
    np.savez_compressed(out/'point_examples.npz',x=np.stack([e[0] for e in examples]),y=np.stack([e[1] for e in examples]),
                       stop=np.stack([e[2] for e in examples]),moving=np.stack([e[3] for e in examples]))
    # Fixed held-out point, not used to tune conventions.
    index=int(candidates[len(candidates)//2]);point=points[index]
    x=np.arange(np.floor(point[0]/.005)-60,np.floor(point[0]/.005)+61)*.005
    y=np.arange(np.floor(point[1]/.005)-60,np.floor(point[1]/.005)+61)*.005
    qs,ts,rs,cs,obs=observations(q,tx,rx,c,x,y)
    ideal=form(qs,ts,rs,cs,x,y);base=peak_stats(ideal,x,y,point)
    cases=[];images=[]
    # Smooth assumed cross-track error; raw signals are held fixed.
    ping=(obs//36).astype(float)
    shape=np.sin(2*np.pi*ping/31)+.5*np.sin(2*np.pi*ping/13+.4)
    shape=shape/np.sqrt(np.mean(shape**2))
    for kind,values in [('cross_track_rms_m',[0,.0005,.001,.002,.005]),('sound_speed_bias_m_s',[.25,.5,1.,2.])]:
        for amount in values:
            te=ts.copy();re=rs.copy();ce=cs.copy()
            if kind=='cross_track_rms_m':te[:,1]+=amount*shape;re[:,1]+=amount*shape
            else:ce+=amount
            z=form(qs,te,re,ce,x,y)
            stats=peak_stats(z,x,y,point)
            stats.update(complex_metrics(ideal,z))
            stats['peak_change_db']=float(20*np.log10(stats['peak_amplitude']/base['peak_amplitude']))
            cases.append({'kind':kind,'amount':amount,**stats});images.append(z.astype(np.complex64))
    summary['physical_error']={'point_index_0based':index,'truth':point.tolist(),
        'interpretation':'Incorrect assumed geometry or c applied to unchanged public IQ, not new environmental measurements',
        'rows':cases}
    np.savez_compressed(out/'physical_errors.npz',images=np.stack(images),ideal=ideal,x=x,y=y)
    (out/'possm_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['configuration','physical_error']},indent=2),flush=True)


if __name__=='__main__':main()
