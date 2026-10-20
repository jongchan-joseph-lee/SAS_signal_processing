"""Secondary fixed spherical-spreading diagnostic; no fitted range exponent.

This hypothesis was added after native-grid validation. It is not included
in the primary frozen v2 result, and is not an independently held-out result.
"""
from pathlib import Path
import json,time
import numpy as np
from evaluate_possm import load_scene,reference
from imaging import complex_metrics,resample_complex
from imaging_v2 import bp
from diagnose_taper import taper
ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[];profiles={}
    for scene in ['scene00','scene01']:
        ref,x,y=reference(next((ROOT/'data/possm_reference').glob(f'*{scene}*hdf5')),1)
        q,tx,rx,c=load_scene(ROOT,scene);qr=resample_complex(taper(q,'hann60'),8)
        z=bp(qr,tx,rx,c,x,y,75000.,8,1);del qr
        # Broadside slant-range squared. This approximates Rt*Rr and ignores
        # along-track offsets; explicitly a pixel-domain diagnostic.
        gain=y[:,None]**2+36.
        rows.append({'scene':scene,'uncompensated':complex_metrics(ref,z),'fixed_r2':complex_metrics(ref,z*gain),
                     'status':'post-hoc diagnostic, not primary frozen validation','gain':'y^2 + h^2, h=6m, exponent fixed=1'})
        profiles[scene+'_y']=y
        profiles[scene+'_ref_rms']=np.sqrt(np.mean(abs(ref)**2,axis=1))
        profiles[scene+'_bp_rms']=np.sqrt(np.mean(abs(z)**2,axis=1))
        profiles[scene+'_r2_rms']=profiles[scene+'_bp_rms']*gain[:,0]
        print(rows[-1],flush=True)
    (ROOT/'results/v2/gain_diagnostic.json').write_text(json.dumps({'rows':rows,'interpretation':'Compatible with missing amplitude normalization, but does not establish the exact ASASIN processing recipe'},indent=2)+'\n')
    np.savez_compressed(ROOT/'results/v2/gain_profiles.npz',**profiles)


if __name__=='__main__':main()
