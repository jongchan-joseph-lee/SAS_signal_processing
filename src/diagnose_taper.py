"""Development-only test of range spectral taper, separate from beam shape."""
from pathlib import Path
import json,time
import h5py,numpy as np
from scipy import fft
from scipy.signal.windows import tukey
from evaluate_possm import load_scene
from imaging import resample_complex,complex_metrics
from imaging_v2 import bp
from diagnose_slc import native_roi
ROOT=Path(__file__).resolve().parents[1]


def taper(q,kind):
    if kind=='none':return np.asarray(q)
    n=fft.next_fast_len(q.shape[-1]);f=fft.fftfreq(n,1/75000.);u=np.abs(f)/30000.
    if kind=='hann60':w=np.where(u<=1,.5+.5*np.cos(np.pi*u),0.)
    elif kind=='hamming60':w=np.where(u<=1,.54+.46*np.cos(np.pi*u),0.)
    elif kind=='hann75':w=.5+.5*np.cos(np.pi*f/37500.)
    elif kind=='tukey60':w=np.where(u<=.5,1.,np.where(u<=1,.5+.5*np.cos(2*np.pi*(u-.5)),0.))
    elif kind=='kaiser60':w=np.where(u<=1,np.i0(6*np.sqrt(np.maximum(1-u*u,0)))/np.i0(6),0.)
    else:raise ValueError(kind)
    return np.ascontiguousarray(fft.ifft(fft.fft(q,n,axis=-1)*w.astype(np.float32),axis=-1)[...,:q.shape[-1]],dtype=np.complex64)


def main():
    out=ROOT/'results/v2';q,tx,rx,c=load_scene(ROOT,'scene00');pts=np.loadtxt(ROOT/'references/Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
    f=h5py.File(next((ROOT/'data/possm_reference').glob('*scene00*hdf5')));rows=[];examples={}
    for idx in [1,2,4]:
        ref,x,y=native_roi(f,pts[idx],12)
        selected=(tx[:,0]>=x.min()-y.max()*np.tan(.05)-.1)&(tx[:,0]<=x.max()+y.max()*np.tan(.05)+.1)
        for kind in ['none','hann60','hamming60','hann75','tukey60','kaiser60']:
            start=time.perf_counter();qr=resample_complex(taper(q[selected],kind),8)
            for beam in [0,1]:
                z=bp(qr,tx[selected],rx[selected],c[selected],x,y,75000.,8,beam,.05)
                row={'point':idx,'taper':kind,'beam':beam,**complex_metrics(ref,z),'seconds_including_resampling_shared':time.perf_counter()-start}
                rows.append(row);print(row,flush=True);examples[f'p{idx}_{kind}_b{beam}']=z.astype(np.complex64)
            del qr
        examples[f'p{idx}_reference']=ref;examples[f'p{idx}_x']=x;examples[f'p{idx}_y']=y
    (out/'taper_ablation.json').write_text(json.dumps({'development_only':True,'rows':rows},indent=2)+'\n')
    np.savez_compressed(out/'taper_ablation.npz',**examples)


if __name__=='__main__':main()
