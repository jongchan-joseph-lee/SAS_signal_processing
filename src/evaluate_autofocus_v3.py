"""Executed development selection followed by frozen 24-point evaluation."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.optimize import check_grad
from evaluate_possm import load_scene
from imaging import resample_complex,complex_metrics
from imaging_v2 import bp,psf_metrics
from diagnose_taper import taper
from stress_points_v2 import navigation_path
from validate_slc_v2 import describe
from autofocus_v3 import ping_stack,active_stack,gpga,minimum_entropy,entropy,objective,gauge_matrix,image
from protocol_v3 import ROOT,OUT,PROTOCOL,write,sha


def stacks(q,tx,rx,c,point,seed,amount):
 x=(np.floor(point[0]/.005)+np.arange(-40,41))*.005
 y=(np.floor(point[1]/.005)+np.arange(-40,41))*.005
 select=(tx[:,0]>=x.min()-y.max()*np.tan(.05)-.11)&(tx[:,0]<=x.max()+y.max()*np.tan(.05)+.11)
 obs=np.flatnonzero(select);p=obs//36;ts=tx[select].copy();rs=rx[select].copy();cs=c[select]
 qr=resample_complex(taper(q[select],'hann60'),8)
 clean=bp(qr,ts,rs,cs,x,y,75000.,8,1)
 if amount:
  d=amount*navigation_path(seed)[p,1];ts[:,1]+=d;rs[:,1]+=d
 stack=ping_stack(qr,ts,rs,cs,p-p.min(),x,y)
 direct=bp(qr,ts,rs,cs,x,y,75000.,8,1)
 parity=np.linalg.norm(stack.sum(0)-direct)/max(np.linalg.norm(direct),1e-30)
 stack,ids=active_stack(stack);ids=ids+p.min()
 return clean,stack,ids,x,y,float(parity)


def scores(clean,z,x,y,point):
 ps=psf_metrics(z,x,y);pk=ps['peak']
 return {**complex_metrics(clean,z),'entropy':entropy(z),
  'width_x_m':ps['x']['width_3db_interpolated_m'],'width_y_m':ps['y']['width_3db_interpolated_m'],
  'position_error_m':float(np.hypot(pk['subpixel_x']-point[0],pk['subpixel_y']-point[1]))}


def develop():
 write();settings=PROTOCOL['autofocus'];q,tx,rx,c=load_scene(ROOT,'scene00')
 pts=np.loadtxt(ROOT/'references/Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
 rows=[];parities=[]
 configs=[('local_gpga',{'iterations':v}) for v in settings['gpga_iterations_grid']]
 configs += [('per_ping_entropy',{'regularization':v,'maxiter':settings['entropy_maxiter']}) for v in settings['entropy_regularization_grid']]
 for index in settings['development_points']:
  for seed,amount in [(None,0)]+[(s,.002) for s in settings['development_seeds']]:
   clean,stack,ids,x,y,parity=stacks(q,tx,rx,c,pts[index],seed,amount);parities.append(parity)
   for method,config in configs:
    start=time.perf_counter()
    if method=='local_gpga':z,ph=gpga(stack,**config)
    else:z,ph,_=minimum_entropy(stack,**config)
    rows.append({'point':index,'seed':seed,'amount':amount,'method':method,'config':config,
     'seconds':time.perf_counter()-start,**scores(clean,z,x,y,pts[index])})
   print('autofocus dev',index,seed,flush=True)
 selected={};summary=[]
 for method,config in configs:
  subset=[r for r in rows if r['method']==method and r['config']==config]
  cleanloss=max(1-r['complex_correlation'] for r in subset if not r['amount'])
  mean=float(np.mean([r['complex_correlation'] for r in subset if r['amount']]))
  summary.append({'method':method,'config':config,'clean_worst_loss':cleanloss,'corrupt_mean_ncc':mean})
 for method in settings['methods'][1:]:
  sub=[r for r in summary if r['method']==method];passing=[r for r in sub if r['clean_worst_loss']<=.01]
  chosen=max(passing,key=lambda r:r['corrupt_mean_ncc']) if passing else min(sub,key=lambda r:(r['clean_worst_loss'],-r['corrupt_mean_ncc']))
  selected[method]=chosen['config']
 # Finite difference analytic-gradient verification on independent complex toy data.
 rng=np.random.default_rng(761);f=rng.normal(size=(6,50))+1j*rng.normal(size=(6,50));pr=gauge_matrix(6);v=rng.normal(size=6)
 err=check_grad(lambda a:objective(a,f,pr,.001)[0],lambda a:objective(a,f,pr,.001)[1],v)
 assert err<1e-5 and max(parities)<1e-10,(err,max(parities))
 result={'protocol_sha256':sha(OUT/'protocol.json'),'selected':selected,'grid_summary':summary,'rows':rows,
  'verification':{'stack_sum_relative_error_max':max(parities),'entropy_gradient_l2_error':float(err)},
  'criterion':'No held-out results used. See protocol.json; clean worst loss constraint first.'}
 path=OUT/'autofocus_selection.json'
 if path.exists():assert json.loads(path.read_text())['selected']==selected
 path.write_text(json.dumps(result,indent=2)+'\n');print('AUTOF0CUS SELECTED',selected,summary,flush=True)


def evaluate():
 settings=PROTOCOL['autofocus'];selection=json.loads((OUT/'autofocus_selection.json').read_text())
 assert selection['protocol_sha256']==sha(OUT/'protocol.json')
 pts=np.loadtxt(ROOT/'references/Scatterer_Locations_Scene00.csv',delimiter=',',skiprows=1)
 q,tx,rx,c=load_scene(ROOT,'scene00');rows=[];examples={};parities=[];phase_records=[]
 # Each case is stored once for the separate common-phase diagnostic.
 cache=ROOT/'data/v3_point_contributions';cache.mkdir(exist_ok=True)
 for index in settings['test_points']:
  for seed,amount in [(None,0)]+[(s,.002) for s in settings['test_seeds']]:
   clean,stack,ids,x,y,parity=stacks(q,tx,rx,c,pts[index],seed,amount);parities.append(parity)
   raw=stack.sum(0);iy,ix=np.unravel_index(abs(raw).argmax(),raw.shape)
   binid=int(np.searchsorted(settings['range_edges_m'],pts[index,1],side='right')-1)
   for method in settings['methods']:
    start=time.perf_counter();info={}
    if method=='uncorrected':z=raw;ph=np.zeros(len(stack))
    elif method=='local_gpga':z,ph=gpga(stack,**selection['selected'][method])
    else:z,ph,info=minimum_entropy(stack,**selection['selected'][method])
    row={'point':index,'seed':seed,'amount':amount,'range_m':float(pts[index,1]),'range_bin':binid,
     'active_pings':len(ids),'method':method,'seconds_estimator':time.perf_counter()-start,
     **scores(clean,z,x,y,pts[index]),'optimizer':info}
    rows.append(row)
    if index==settings['test_points'][0] and seed in [None,settings['test_seeds'][0]]:
     examples[f'{"clean_control" if seed is None else "sway2mm"}_{method}']=z
   if amount:
    u=stack[:,iy,ix];g=np.conj(u[:-1])*u[1:]
    phase_records.append({'point':index,'seed':seed,'range_bin':binid,'ids':ids.tolist(),
      'grad_real':g.real.tolist(),'grad_imag':g.imag.tolist()})
    np.savez_compressed(cache/f'p{index}_s{seed}.npz',stack=stack.astype(np.complex64),ids=ids,clean=clean.astype(np.complex64),x=x,y=y)
   if index==settings['test_points'][0]:examples.update({'reference':clean,'x':x,'y':y})
  print('autofocus test point',index,'rows',len(rows),flush=True)
 groups=[]
 for amount in [0,.002]:
  for method in settings['methods']:
   for band in [None,0,1,2]:
    sub=[r for r in rows if r['amount']==amount and r['method']==method and (band is None or r['range_bin']==band)]
    if sub:groups.append({'amount':amount,'method':method,'range_bin':band,'cases':len(sub),
     **{k:describe([r[k] for r in sub]) for k in ['complex_correlation','amplitude_correlation','entropy','width_x_m','width_y_m','position_error_m','seconds_estimator']}})
 success={}
 for method in settings['methods'][1:]:
  corrupt=[r for r in rows if r['method']==method and r['amount']];controls=[r for r in rows if r['method']==method and not r['amount']]
  med=float(np.median([r['complex_correlation'] for r in corrupt]));loss=max(1-r['complex_correlation'] for r in controls)
  success[method]={'corrupt_median_ncc':med,'clean_max_ncc_loss':loss,'corrupt_pass':med>=.9,'clean_pass':loss<=.01,'overall_pass':med>=.9 and loss<=.01,
   'cases_ncc_ge_0_9':sum(r['complex_correlation']>=.9 for r in corrupt),'total':len(corrupt)}
 result={'protocol_sha256':sha(OUT/'protocol.json'),'selection_sha256':sha(OUT/'autofocus_selection.json'),
  'rows':rows,'summary':groups,'success':success,'stack_sum_relative_error_max':max(parities),
  'scope':settings['primary_scope'],'gauge':settings['gauge'],'ncc_reference':'same frozen BP with clean navigation, no spatial registration',
  'limitations':'Point ROIs are GT-selected; estimator is blind within each ROI. 24 points and 5 paths share one synthetic scene. Phase-only correction does not re-estimate geometric migration or absolute position.'}
 (OUT/'autofocus_results.json').write_text(json.dumps(result,indent=2)+'\n')
 np.savez_compressed(OUT/'autofocus_examples.npz',**examples)
 # Phase screen estimated from pooled point peaks: global vs fixed range bands.
 diag=[]
 for seed in settings['test_seeds']:
  curves={}
  for band in [None,0,1,2]:
   gradient=np.zeros(99,complex)
   for r in phase_records:
    if r['seed']!=seed or (band is not None and r['range_bin']!=band):continue
    ids=np.asarray(r['ids']);g=np.asarray(r['grad_real'])+1j*np.asarray(r['grad_imag'])
    for k in range(len(g)):
     if ids[k+1]==ids[k]+1:gradient[ids[k]]+=g[k]
   ph=np.r_[0,np.cumsum(np.angle(gradient))];ph=gauge_matrix(100)@ph;curves[band]=ph
  for index in settings['test_points']:
   d=np.load(cache/f'p{index}_s{seed}.npz');band=int(np.searchsorted(settings['range_edges_m'],pts[index,1],side='right')-1)
   for label,b in [('common_all_ranges',None),('common_within_range_band',band)]:
    z=image(d['stack'],curves[b][d['ids']]);diag.append({'point':index,'seed':seed,'range_bin':band,'method':label,
     **scores(d['clean'],z,d['x'],d['y'],pts[index])})
 (OUT/'common_phase_diagnostic.json').write_text(json.dumps({'rows':diag,'status':'Secondary diagnostic, not used for primary selection','selection':'known isolated-point ROIs, strongest pixel selected blindly; each point contributes to its phase estimate'},indent=2)+'\n')
 print('AUTOF0CUS SUCCESS',success,flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['develop','test'],required=True);a=p.parse_args()
 develop() if a.stage=='develop' else evaluate()
