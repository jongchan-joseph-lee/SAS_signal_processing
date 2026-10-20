"""24 points x 5 random navigation paths, plus deterministic c biases.

Corrupt assumed raw-stage Tx/Rx navigation; leave public IQ unchanged.
No SLC polynomial phase injection. This is a sensitivity study, not autofocus.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from evaluate_possm import load_scene
from imaging import complex_metrics,resample_complex
from imaging_v2 import bp,psf_metrics
from diagnose_taper import taper
from validate_slc_v2 import describe
ROOT=Path(__file__).resolve().parents[1]


def navigation_path(seed):
    rng=np.random.default_rng(seed);v=rng.standard_normal((100,2))
    low=gaussian_filter1d(v,5,axis=0,mode='reflect');low/=low.std(0)
    high=rng.standard_normal((100,2));shape=low+.25*high;shape-=shape.mean(0)
    return shape/np.sqrt(np.mean(shape*shape,axis=0))


def entropy(z):
    p=np.abs(z)**2;p=p/(p.sum()+1e-30)
    return float(-np.sum(p*np.log(np.maximum(p,1e-30))))


def main():
    prev=json.loads((ROOT/'results/possm/possm_results.json').read_text());indices=[r['index_0based'] for r in prev['heldout_points']['rows']]
    points=np.loadtxt(ROOT/'references/Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
    q,tx,rx,c=load_scene(ROOT,'scene00');rows=[];seeds=[6101,6102,6103,6104,6105]
    paths={s:navigation_path(s) for s in seeds};arrays={}
    for index in indices:
        point=points[index];x=(np.floor(point[0]/.005)+np.arange(-40,41))*.005;y=(np.floor(point[1]/.005)+np.arange(-40,41))*.005
        selected=(tx[:,0]>=x.min()-y.max()*np.tan(.05)-.1)&(tx[:,0]<=x.max()+y.max()*np.tan(.05)+.1)
        obs=np.flatnonzero(selected);ts=tx[selected];rs=rx[selected];cs=c[selected];pings=obs//36
        start=time.perf_counter();qr=resample_complex(taper(q[selected],'hann60'),8);preparation=time.perf_counter()-start
        clean=bp(qr,ts,rs,cs,x,y,75000.,8,1);peak=abs(clean).max()
        if index==indices[0]:arrays.update({'clean':clean,'x':x,'y':y})
        for kind in ['surge','sway','joint','sound_speed_bias']:
            for amount in ([.25,.5,1.,2.] if kind=='sound_speed_bias' else [.0005,.001,.002,.005]):
                for seed in ([None] if kind=='sound_speed_bias' else seeds):
                    te=ts.copy();re=rs.copy();ce=cs.copy()
                    if kind=='sound_speed_bias':ce+=amount
                    else:
                        for axis in ([0,1] if kind=='joint' else [0] if kind=='surge' else [1]):
                            error=amount*paths[seed][pings,axis];te[:,axis]+=error;re[:,axis]+=error
                    start=time.perf_counter();z=bp(qr,te,re,ce,x,y,75000.,8,1);elapsed=time.perf_counter()-start
                    psf=psf_metrics(z,x,y);pk=psf['peak']
                    row={'point':index,'kind':kind,'amount':amount,'seed':seed,'seconds_bp':elapsed,
                         **complex_metrics(clean,z),'entropy':entropy(z),'peak_change_db':float(20*np.log10(abs(z).max()/peak)),
                         'width_x_m':psf['x']['width_3db_interpolated_m'],'width_y_m':psf['y']['width_3db_interpolated_m'],
                         'subpixel_error_m':float(np.hypot(pk['subpixel_x']-point[0],pk['subpixel_y']-point[1]))}
                    rows.append(row)
                    if index==indices[0] and seed==seeds[0] and kind=='joint':arrays[f'joint_{amount:g}']=z.astype(np.complex64)
        print('stress point',index,'cases',len(rows),'preparation_s',preparation,flush=True);del qr
    groups=[]
    for kind in ['surge','sway','joint','sound_speed_bias']:
        for amount in sorted(set(r['amount'] for r in rows if r['kind']==kind)):
            subset=[r for r in rows if r['kind']==kind and r['amount']==amount]
            groups.append({'kind':kind,'amount':amount,'cases':len(subset),**{k:describe([r[k] for r in subset]) for k in ['complex_correlation','aligned_nrmse','entropy','peak_change_db','width_x_m','width_y_m','subpixel_error_m','seconds_bp']}})
    result={'seeds':seeds,'points':indices,'rows':rows,'summary':groups,
        'source':'Unchanged public model-derived IQ; corrupted assumed raw navigation or sound speed',
        'path':'Gaussian-filtered random low-frequency path sigma5 pings + 0.25 white component, de-meaned, each axis unit RMS over the full 100-ping track',
        'baseline':'Frozen Hann60 + Gaussian horizontal + 8x interpolation; same clean-geometry BP is comparison reference',
        'caution':'120 point/path cases per navigation setting share 24 points and 5 paths; not 120 independent scenes. Fixed c biases have 24 cases and no random repetitions. No estimator corrected these errors in this study.'}
    (ROOT/'results/v2/stress_results.json').write_text(json.dumps(result,indent=2)+'\n')
    # Only the JSON summaries are used in the report; preserve coherent examples.
    np.savez_compressed(ROOT/'results/v2/stress_examples.npz',**arrays)


if __name__=='__main__':main()
