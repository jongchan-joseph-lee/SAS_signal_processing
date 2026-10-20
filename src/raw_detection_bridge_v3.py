"""Held-out scene04: raw -> own BP -> frozen point detector, controlled error.

Automatic bright-candidate autofocus is deliberately tested separately from
the optimistic GT-selected isolated-point experiment. No GT in phase fitting.
"""
import json,time,datetime,joblib
import numpy as np
from evaluate_possm import load_scene,reference
from imaging import resample_complex,complex_metrics
from imaging_v2 import bp
from diagnose_taper import taper
from stress_points_v2 import navigation_path
from autofocus_v3 import ping_stack,active_stack,minimum_entropy,gauge_matrix
from point_detection_v3 import candidates,features,truth,pooled
from protocol_v3 import ROOT,OUT,sha

ADDENDUM={'scene':4,'navigation':'sway 2mm RMS, seed7101, unchanged IQ',
 'beamformer':'frozen v2 Hann60 / horizontal Gaussian / 8x; native 2.5cm, intensity 2x2 mean to 5cm',
 'autofocus':'top30 corrupted-image contrast candidates; separation 2m; no GT; 0.4m square, 5mm grid',
 'phase_estimator':'same fixed per-ping entropy setting selected on development points; no object-specific retuning',
 'pooling':'circular average adjacent-ping phase gradients, weighted by patch contribution energy; 3 fixed range bands [0,35,55,90.1]; integrate and remove constant/linear phase',
 'detector':'frozen official-SLC-trained RF and frozen validation threshold; no training on own BP',
 'evaluation':'whole valid image, point match radius1m, exact same mask as primary; no GT crops for detector or correction',
 'status':'single held-out scene controlled-error handoff diagnostic, not 4-scene generalization or primary adoption evidence'}


def record():
 p=OUT/'raw_bridge_protocol.json'
 if p.exists():assert json.loads(p.read_text())['protocol']==ADDENDUM
 else:p.write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':ADDENDUM,
  'detector_selection_sha256':sha(OUT/'detection_selection.json'),'autofocus_selection_sha256':sha(OUT/'autofocus_selection.json')},indent=2)+'\n')


def downsample(z,x,y):
 p=abs(z)**2;ny=p.shape[0]//2*2;nx=p.shape[1]//2*2
 return p[:ny,:nx].reshape(ny//2,2,nx//2,2).mean((1,3)),x[:nx].reshape(-1,2).mean(1),y[:ny].reshape(-1,2).mean(1)


def main():
 record();startall=time.perf_counter();sel=json.loads((OUT/'detection_selection.json').read_text())
 af=json.loads((OUT/'autofocus_selection.json').read_text());model=joblib.load(OUT/'point_detector.joblib')
 assert sha(OUT/'point_detector.joblib')==sel['model_sha256']
 ref,x,y=reference(next((ROOT/'data/possm_reference').glob('*scene04*hdf5')),stride=1)
 q,tx,rx,c=load_scene(ROOT,'scene04');start=time.perf_counter();qr=resample_complex(taper(q,'hann60'),8);prep=time.perf_counter()-start
 images={};times={};start=time.perf_counter();images['clean_bp']=bp(qr,tx,rx,c,x,y,75000.,8,1);times['clean_bp']=time.perf_counter()-start
 te=tx.copy();re=rx.copy();d=.002*navigation_path(7101)[np.arange(len(tx))//36,1];te[:,1]+=d;re[:,1]+=d
 start=time.perf_counter();images['corrupted_bp']=bp(qr,te,re,c,x,y,75000.,8,1);times['corrupted_bp']=time.perf_counter()-start
 print('raw bridge clean/corrupted formed',times,flush=True)
 power,xx,yy=downsample(images['corrupted_bp'],x,y);cand,_=candidates(power,xx,yy)
 chosen=[]
 for pos in cand[:,:2]:
  if chosen and np.min(np.linalg.norm(np.array(chosen)-pos,axis=1))<2:continue
  chosen.append(pos)
  if len(chosen)==30:break
 edges=[0,35,55,90.1];grads=np.zeros((3,99),complex);weights=np.zeros((3,99));patchlog=[]
 start=time.perf_counter()
 for i,pos in enumerate(chosen):
  gx=(np.floor(pos[0]/.005)+np.arange(-40,41))*.005;gy=(np.floor(pos[1]/.005)+np.arange(-40,41))*.005
  mask=(te[:,0]>=gx.min()-gy.max()*np.tan(.05)-.11)&(te[:,0]<=gx.max()+gy.max()*np.tan(.05)+.11)
  obs=np.flatnonzero(mask);p=obs//36
  s=ping_stack(qr[mask],te[mask],re[mask],c[mask],p-p.min(),gx,gy)
  s,ids=active_stack(s);ids+=p.min()
  _,ph,info=minimum_entropy(s,**af['selected']['per_ping_entropy'])
  band=min(2,int(np.searchsorted(edges,pos[1],side='right')-1));energy=np.sum(abs(s)**2,axis=(1,2))
  for k in range(len(ids)-1):
   if ids[k+1]==ids[k]+1:
    w=np.sqrt(energy[k]*energy[k+1]);grads[band,ids[k]]+=w*np.exp(1j*(ph[k+1]-ph[k]));weights[band,ids[k]]+=w
  patchlog.append({'position':pos.tolist(),'range_band':band,'active_pings':ids.tolist(),'phase':ph.tolist(),'optimizer':info})
  print('raw bridge automatic patch',i+1,flush=True)
 phase=np.array([gauge_matrix(100)@np.r_[0,np.cumsum(np.angle(g))] for g in grads])
 phase_seconds=time.perf_counter()-start
 # Modulate range-compressed samples only; IQ and assumed geometry remain fixed.
 corrected=np.zeros_like(images['corrupted_bp']);start=time.perf_counter()
 for band in range(3):
  iy=np.flatnonzero((y>=edges[band])&(y<edges[band+1]));rot=np.exp(-1j*phase[band,np.arange(len(tx))//36]).astype(np.complex64)
  qr*=rot[:,None];corrected[iy]=bp(qr,te,re,c,x,y[iy],75000.,8,1);qr*=np.conj(rot[:,None])
 images['automatic_entropy_bp']=corrected;times['automatic_entropy_bp']=time.perf_counter()-start
 rows=[];arrays={'x':xx,'y':yy,'truth':truth(4),'auto_patch_positions':np.array(chosen),'phase':phase,'phase_weights':weights}
 for name,z in images.items():
  power,xx,yy=downsample(z,x,y);cand,area=candidates(power,xx,yy);f=features(power,xx,yy,cand[:,:2]);pr=model.predict_proba(f)
  rec={'scene':4,'positions':cand[:,:2],'proposal':cand[:,2],'truth':truth(4),'probability':pr,'area_m2':area}
  rows.append({'image':name,'clean_bp_complex':complex_metrics(images['clean_bp'],z),'official_reference_complex':complex_metrics(ref,z),
   'detection':pooled([rec],sel['rf_threshold']),'bp_seconds':times[name]})
  arrays[name+'_power']=power[::2,::2];arrays[name+'_positions']=cand[:,:2];arrays[name+'_probability']=pr
 arrays['x']=xx[::2];arrays['y']=yy[::2]
 result={'protocol_sha256':sha(OUT/'raw_bridge_protocol.json'),
  'execution_detector_selection_sha256':sha(OUT/'detection_selection.json'),
  'execution_autofocus_selection_sha256':sha(OUT/'autofocus_selection.json'),
  'rows':rows,'automatic_patches':patchlog,
  'range_edges':edges,'preparation_seconds':prep,'autofocus_seconds':phase_seconds,'total_seconds':time.perf_counter()-startall,
  'status':ADDENDUM['status'],'warning':'Phase-only local-to-band pooling can fail on distributed speckle. Results kept even when entropy improves but NCC or detection worsens.'}
 (OUT/'raw_bridge_results.json').write_text(json.dumps(result,indent=2)+'\n')
 np.savez_compressed(OUT/'raw_bridge_examples.npz',**arrays)
 print('RAW BRIDGE RESULTS',[(r['image'],r['detection']['tp'],r['detection']['fp'],r['clean_bp_complex']['complex_correlation']) for r in rows],flush=True)


if __name__=='__main__':main()
