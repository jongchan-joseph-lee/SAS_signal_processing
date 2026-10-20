"""Full-image candidate detection + random-forest classification, point GT.

Original transparent low-data baseline. Not YOLO, not a paper reproduction,
and not box-mAP. No absolute coordinates or range are classifier features.
"""
from pathlib import Path
import argparse,json,time,joblib
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter,map_coordinates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from evaluate_possm import reference
from protocol_v3 import ROOT,OUT,PROTOCOL,write,sha

NAMES=['background','trap','rock','cylinder']


def truth(scene):
 a=np.loadtxt(ROOT/'references/Object_Locations_All_Scenes.csv',delimiter=',',skiprows=1)
 a=a[a[:,1]==scene]
 return np.column_stack([a[:,3],a[:,4],a[:,2],a[:,0]])


def official(scene):
 z,x,y=reference(next((ROOT/'data/possm_reference').glob(f'*scene{scene:02d}*hdf5')),stride=1)
 power=np.abs(z)**2;ny=power.shape[0]//2*2;nx=power.shape[1]//2*2
 power=power[:ny,:nx].reshape(ny//2,2,nx//2,2).mean((1,3))
 return power,x[:nx].reshape(-1,2).mean(1),y[:ny].reshape(-1,2).mean(1)


def features(power,x,y,positions):
 """6x6 m, 12x12 cells: log power, variation, highlight & shadow fractions.

 Patch-local normalization: no scene label, coordinates, range or GT class.
 72x72 samples at 8.45cm; mean/std on each 6x6-sample cell. Cells have
 fixed acoustic range direction; no 90-degree rotations of shadows.
 """
 out=[];stepx=float(np.median(np.diff(x)));stepy=float(np.median(np.diff(y)))
 offset=np.linspace(-3,3,72);oy,ox=np.meshgrid(offset,offset,indexing='ij')
 eps=max(float(np.quantile(power,.01)),1e-25)
 lp=np.log(np.maximum(power,eps))
 for px,py in positions:
  grid=np.array([(py+oy-y[0])/stepy,(px+ox-x[0])/stepx])
  p=map_coordinates(lp,grid,order=1,mode='nearest')
  med=np.median(p);scale=max(float(np.quantile(p,.9)-np.quantile(p,.1)),.5)
  p=np.clip((p-med)/scale,-3,3);cells=p.reshape(12,6,12,6)
  vec=[cells.mean((1,3)).ravel(),cells.std((1,3)).ravel(),
       (cells>.5).mean((1,3)).ravel(),(cells<-.5).mean((1,3)).ravel(),
       np.quantile(p,[0,.05,.1,.25,.75,.9,.95,1]),p.mean(0)[::3],p.mean(1)[::3]]
  out.append(np.concatenate(vec))
 return np.asarray(out,np.float32)


def candidates(power,x,y):
 eps=max(float(np.quantile(power,.01)),1e-25);lp=np.log(np.maximum(power,eps))
 broad=gaussian_filter(lp,40,mode='reflect')
 score=np.maximum.reduce([gaussian_filter(lp,s,mode='reflect')-broad for s in [3,6,12]])
 valid=(y[:,None]>=y[0]+3)&(y[:,None]<=y[-1]-3)&(x[None,:]>=x[0]+3)&(x[None,:]<=x[-1]-3)
 # Restrict to actual insonification support using geometry only; no GT.
 valid &= (x[None,:] >= .675-y[:,None]*np.tan(.05)) & (x[None,:]<=60.075+y[:,None]*np.tan(.05))
 local=(score==maximum_filter(score,size=13))&valid
 iy,ix=np.nonzero(local);rank=np.argsort(-score[iy,ix],kind='stable');picked=[]
 for k in rank:
  pos=np.array([x[ix[k]],y[iy[k]]])
  if picked and np.min(np.linalg.norm(np.array([p[:2] for p in picked])-pos,axis=1))<.65:continue
  picked.append([float(pos[0]),float(pos[1]),float(score[iy[k],ix[k]])])
  if len(picked)==600:break
 return np.array(picked),float(valid.sum()*np.median(np.diff(x))*np.median(np.diff(y)))


def dataset(scene):
 t=time.perf_counter();power,x,y=official(scene);cand,area=candidates(power,x,y);f=features(power,x,y,cand[:,:2])
 return {'scene':scene,'positions':cand[:,:2],'proposal':cand[:,2],'features':f,'truth':truth(scene),'area_m2':area,
   'seconds_proposals_features':time.perf_counter()-t},power,x,y


def match_scene(d,score,classes,threshold,radius=1.,class_aware=False):
 gt=d['truth'];used=set();matches=[];fp=0
 for k in np.argsort(-score,kind='stable'):
  if score[k]<threshold:continue
  dist=np.linalg.norm(gt[:,:2]-d['positions'][k],axis=1)
  allowed=[j for j in range(len(gt)) if j not in used and dist[j]<=radius and (not class_aware or classes[k]==int(gt[j,2]))]
  if allowed:
   j=min(allowed,key=lambda j:dist[j]);used.add(j);matches.append([int(k),int(j),float(dist[j])])
  else:fp+=1
 return {'tp':len(used),'fp':fp,'total':len(gt),'matches':matches}


def wilson(k,n):
 if not n:return [0.,1.]
 z=1.95996398454;p=k/n;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(mid-half),float(mid+half)]


def pooled(data,threshold,kind='rf',radius=1.,class_aware=False,bootstrap=False):
 rows=[];ct=np.zeros((3,2),int);cm=np.zeros((3,4),int)
 for d in data:
  if kind=='proposal':s=d['proposal'];cl=np.ones(len(s),int)
  else:s=1-d['probability'][:,0];cl=np.argmax(d['probability'][:,1:],axis=1)+1
  m=match_scene(d,s,cl,threshold,radius,class_aware);rows.append({'scene':d['scene'],**m})
  seen={j:(k,dist) for k,j,dist in m['matches']}
  for j,g in enumerate(d['truth']):
   label=int(g[2]);ct[label-1,1]+=1
   if j in seen:
    k,_=seen[j];ct[label-1,0]+=1;cm[label-1,cl[k]]+=1
   else:cm[label-1,0]+=1
 tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);n=sum(r['total'] for r in rows)
 out={'threshold':float(threshold) if np.isfinite(threshold) else None,'tp':tp,'fp':fp,'total':n,'recall':tp/n,'fp_per_scene':fp/len(rows),
  'fp_per_km2':fp/sum(d['area_m2'] for d in data)*1e6,'wilson95':wilson(tp,n),'scenes':rows,
  'class_recall':[{'class':NAMES[i+1],'tp':int(c[0]),'total':int(c[1]),'wilson95':wilson(*c)} for i,c in enumerate(ct)],
  'confusion_rows_truth_columns_background_trap_rock_cylinder':cm.tolist()}
 if bootstrap:
  rng=np.random.default_rng(731);vals=[]
  for _ in range(5000):
   sample=rng.integers(len(rows),size=len(rows));vals.append(sum(rows[i]['tp'] for i in sample)/sum(rows[i]['total'] for i in sample))
  out['scene_bootstrap95']=[float(v) for v in np.quantile(vals,[.025,.975])]
 return out


def operating_point(data,kind='rf'):
 scores=np.concatenate([d['proposal'] if kind=='proposal' else 1-d['probability'][:,0] for d in data])
 thresholds=np.r_[np.nextafter(scores.max(),np.inf),np.unique(scores)[::-1]]
 best=None
 for t in thresholds:
  r=pooled(data,t,kind)
  if r['fp_per_scene']<=5 and (best is None or (r['tp'],-r['fp'],r['threshold'])>(best['tp'],-best['fp'],best['threshold'])):best=r
 return best


def curve(data,kind='rf',class_aware=False,radius=1.):
 s=np.concatenate([d['proposal'] if kind=='proposal' else 1-d['probability'][:,0] for d in data])
 # Complete unique-score curve; no threshold is chosen using test results.
 return [{k:r[k] for k in ['threshold','tp','fp','total','recall','fp_per_scene']} for r in
  [pooled(data,t,kind,radius,class_aware) for t in np.r_[np.nextafter(s.max(),np.inf),np.unique(s)[::-1]]]]


def train():
 write();cfg=PROTOCOL['detection'];training=[];validation=[];datasets={};trainrows=[]
 for scene in cfg['train']+cfg['validation']:
  d,power,x,y=dataset(scene);datasets[scene]=d
  print('detector source',scene,'candidates',len(d['positions']),flush=True)
  if scene in cfg['train']:
   dist=np.linalg.norm(d['positions'][:,None,:]-d['truth'][None,:,:2],axis=-1);near=dist.argmin(1);md=dist.min(1)
   labels=np.where(md<=1,d['truth'][near,2],0).astype(int);keep=(md<=1)|(md>2)
   training.append((d['features'][keep],labels[keep]))
   # Augment positive training locations only; not extra independent objects.
   pos=[];lab=[]
   for g in d['truth']:
    for dx,dy in [(0,0),(-.35,0),(.35,0),(0,-.35),(0,.35),(-.25,-.25),(.25,.25),(-.25,.25),(.25,-.25)]:
     pos.append(g[:2]+[dx,dy]);lab.append(int(g[2]))
   training.append((features(power,x,y,pos),np.array(lab)))
   trainrows.append({'scene':scene,'positive_candidates':int((labels[keep]>0).sum()),'background_candidates':int((labels[keep]==0).sum()),'GT_jitter_crops':len(pos),'unique_objects':9})
  else:validation.append(d)
 X=np.concatenate([v[0] for v in training]);Y=np.concatenate([v[1] for v in training]);grid=[];fits=[]
 for leaf in cfg['rf_min_samples_leaf_grid']:
  model=RandomForestClassifier(n_estimators=300,min_samples_leaf=leaf,max_features='sqrt',class_weight='balanced_subsample',random_state=731,n_jobs=4)
  start=time.perf_counter();model.fit(X,Y);seconds=time.perf_counter()-start
  for d in validation:d['probability']=model.predict_proba(d['features'])
  op=operating_point(validation);grid.append({'min_samples_leaf':leaf,'validation':op,'fit_seconds':seconds});fits.append(model)
 pick=max(range(len(grid)),key=lambda i:(grid[i]['validation']['tp'],-grid[i]['validation']['fp'],grid[i]['min_samples_leaf']))
 model=fits[pick]
 for d in validation:d['probability']=model.predict_proba(d['features'])
 prop=operating_point(validation,'proposal')
 selection={'protocol_sha256':sha(OUT/'protocol.json'),'selected_min_samples_leaf':grid[pick]['min_samples_leaf'],
  'rf_threshold':grid[pick]['validation']['threshold'],'proposal_threshold':prop['threshold'],'grid':grid,'proposal_validation':prop,
  'training':trainrows,'feature_count':int(X.shape[1]),'training_samples':int(len(Y)),
  'class_order':model.classes_.tolist(),'model':'Original local-contrast + patch statistics RandomForest baseline, no paper reproduction or pretrained weights',
  'test_not_used':True}
 joblib.dump(model,OUT/'point_detector.joblib',compress=3)
 selection['model_sha256']=sha(OUT/'point_detector.joblib')
 (OUT/'detection_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 np.savez_compressed(OUT/'detector_training_features.npz',X=X,y=Y)
 print('DETECTOR SELECTED',selection['selected_min_samples_leaf'],selection['rf_threshold'],grid[pick]['validation']['tp'],flush=True)


def test():
 cfg=PROTOCOL['detection'];sel=json.loads((OUT/'detection_selection.json').read_text())
 assert sel['protocol_sha256']==sha(OUT/'protocol.json') and sel['model_sha256']==sha(OUT/'point_detector.joblib')
 model=joblib.load(OUT/'point_detector.joblib');data=[];aux_y=[];aux_pred=[];example={}
 for scene in cfg['test']:
  d,power,x,y=dataset(scene);start=time.perf_counter();d['probability']=model.predict_proba(d['features']);d['seconds_predict']=time.perf_counter()-start;data.append(d)
  # Auxiliary GT-crop classifier, explicitly NOT end-to-end detection.
  f=features(power,x,y,d['truth'][:,:2]);aux_y.extend(d['truth'][:,2].astype(int));aux_pred.extend(np.argmax(model.predict_proba(f)[:,1:],axis=1)+1)
  if scene==cfg['test'][0]:example={'power':power[::2,::2],'x':x[::2],'y':y[::2],'positions':d['positions'],'truth':d['truth'],'probability':d['probability']}
  print('detector test scene',scene,flush=True)
 result={'protocol_sha256':sha(OUT/'protocol.json'),'selection_sha256':sha(OUT/'detection_selection.json'),
  'operating_point_rf':pooled(data,sel['rf_threshold'],bootstrap=True),
  'operating_point_proposal':pooled(data,sel['proposal_threshold'],'proposal',bootstrap=True),
  'class_aware_rf':pooled(data,sel['rf_threshold'],class_aware=True,bootstrap=True),
  'candidate_ceiling':pooled(data,-np.inf,'proposal'),
  'matching_radius_diagnostic':[{'radius_m':r,'result':pooled(data,sel['rf_threshold'],radius=r)} for r in cfg['secondary_match_radii_m']],
  'froc_rf':curve(data),'froc_proposal':curve(data,'proposal'),'froc_class_aware':curve(data,class_aware=True),
  'auxiliary_GT_crop_classifier':{'correct':int(np.sum(np.array(aux_y)==aux_pred)),'total':len(aux_y),
   'confusion':confusion_matrix(aux_y,aux_pred,labels=[1,2,3]).tolist(),'scope':'known GT location AND foreground given; argmax over 3 foreground classes only; not detection'},
  'test_scenes':cfg['test'],'train_unique_objects':36,'validation_unique_objects':18,'test_unique_objects':36,
  'interval_caution':'Wilson assumes independent placements; repeated mesh designs violate independence. Scene bootstrap (4 scenes only) is separately reported and remains unstable.',
  'timing':[{'scene':d['scene'],'seconds_proposals_features':d['seconds_proposals_features'],'seconds_predict':d['seconds_predict'],'search_area_m2':d['area_m2']} for d in data]}
 op=result['operating_point_rf'];result['success']={'recall_pass':op['recall']>=.8,'fp_pass':op['fp_per_scene']<=5,'overall_pass':op['recall']>=.8 and op['fp_per_scene']<=5}
 (OUT/'detection_results.json').write_text(json.dumps(result,indent=2)+'\n')
 # Complete candidates, scores and GT permit independent FROC recomputation.
 rec=[]
 for d in data:rec.append({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items() if k!='features'})
 (OUT/'detection_test_predictions.json').write_text(json.dumps(rec,indent=2)+'\n')
 np.savez_compressed(OUT/'detection_example.npz',**example)
 print('DETECTION TEST',op['tp'],op['total'],'FP',op['fp'],'success',result['success'],flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','test'],required=True);a=p.parse_args()
 train() if a.stage=='train' else test()
