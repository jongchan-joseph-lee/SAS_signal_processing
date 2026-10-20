"""Native SLC-grid ablations on v1 development points only."""
from pathlib import Path
import json,time
import h5py,numpy as np
from evaluate_possm import load_scene
from imaging import resample_complex,complex_metrics
from imaging_v2 import bp,psf_metrics

ROOT=Path(__file__).resolve().parents[1]
CONFIGS=[('legacy4',4,0,.05,False,False),('uniform8',8,0,.05,False,False),
 ('uniform16',16,0,.05,False,False),('gauss_same8',8,1,.05,False,False),
 ('gauss_wide8',8,1,.15,False,False),('uniform_wide8',8,0,.15,False,False),
 ('gauss_3d8',8,2,.15,False,False),('gauss_norm8',8,2,.15,True,False),
 ('gauss_path_norm8',8,2,.15,True,True)]


def native_roi(f,point,half=20):
    off=f['ImagingGrid/Offset'][0];step=f['ImagingGrid/Step'][0]
    ix=int(np.rint((point[0]-float(off[0]))/float(step[0])));iy=int(np.rint((point[1]-float(off[1]))/float(step[1])))
    ii=np.arange(ix-half,ix+half+1);jj=np.arange(iy-half,iy+half+1)
    x=float(off[0])+ii*float(step[0]);y=float(off[1])+jj*float(step[1])
    z=(f['Data/Real'][ii[0]:ii[-1]+1,jj[0]:jj[-1]+1,0]+1j*f['Data/Imag'][ii[0]:ii[-1]+1,jj[0]:jj[-1]+1,0]).T
    return z,x,y


def main():
    out=ROOT/'results'/'v2';out.mkdir(parents=True,exist_ok=True)
    q,tx,rx,c=load_scene(ROOT,'scene00');points=np.loadtxt(ROOT/'references/Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
    f=h5py.File(next((ROOT/'data/possm_reference').glob('*scene00*hdf5')))
    metadata={}
    for name in ['ImagingGrid/Offset','ImagingGrid/Step','Parameters/AzLimit','Parameters/CenterFrequency','Parameters/IntermediateFrequency']:
        metadata[name]={'value':str(f[name][...]),'attributes':{k:str(v) for k,v in f[name].attrs.items()}}
    rows=[];examples={}
    for index in [1,2,4]:
        ref,x,y=native_roi(f,points[index]);examples[f'p{index}_reference']=ref;examples[f'p{index}_x']=x;examples[f'p{index}_y']=y
        for name,ratio,beam,cap,norm,spread in CONFIGS:
            selected=(tx[:,0]>=x.min()-y.max()*np.tan(cap)-.1)&(tx[:,0]<=x.max()+y.max()*np.tan(cap)+.1)
            start=time.perf_counter();qr=resample_complex(np.asarray(q[selected]),ratio)
            z=bp(qr,tx[selected],rx[selected],c[selected],x,y,75000.,ratio,beam,cap,norm,True,spread)
            row={'point_0based':index,'method':name,'ratio':ratio,'beam':beam,'cap_rad':cap,'normalize':norm,'spreading':spread,
                 'seconds':time.perf_counter()-start,**complex_metrics(ref,z),'psf':psf_metrics(z,x,y),'reference_psf':psf_metrics(ref,x,y)}
            rows.append(row);examples[f'p{index}_{name}']=z.astype(np.complex64)
            print({k:v for k,v in row.items() if k not in ['psf','reference_psf']},flush=True);del qr
    (out/'native_ablation.json').write_text(json.dumps({'development_indices':[1,2,4],'metadata':metadata,'rows':rows,
        'caution':'Native 2.5cm SLC grid undersamples the ~1cm range PSF; interpolated reference widths/PSLR are sampling-limited.'},indent=2)+'\n')
    np.savez_compressed(out/'native_ablation.npz',**examples)


if __name__=='__main__':main()
