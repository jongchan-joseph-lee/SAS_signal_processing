"""Input integrity and explicit seabed-family split for later model work."""
from pathlib import Path
import csv,hashlib,json
import h5py,numpy as np
from gdf import read_gdf
ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=[]
    objects=np.loadtxt(ROOT/'references/Object_Locations_All_Scenes.csv',delimiter=',',skiprows=1)
    for i in range(11):
        scene=f'scene{i:02d}';config,kernel,pings,manifest=read_gdf((ROOT/'data'/('possm_'+scene)).glob('*.gdf'))
        slc=next((ROOT/'data/possm_reference').glob(f'*{scene}*hdf5'))
        with h5py.File(slc) as f:
            shape=list(f['Data/Real'].shape);off=f['ImagingGrid/Offset'][0];step=f['ImagingGrid/Step'][0]
            bounds=[float(off[0]),float(off[0])+float(step[0])*(shape[0]-1),float(off[1]),float(off[1])+float(step[1])*(shape[1]-1)]
        placement=objects[objects[:,1]==i]
        in_frame=(placement[:,3]>=bounds[0])&(placement[:,3]<=bounds[1])&(placement[:,4]>=bounds[2])&(placement[:,4]<=bounds[3])
        row={'scene':scene,'status':'ok','pings':len(pings),'channels':config['channels'],'samples':sorted(set(p['iq'].shape[1] for p in pings)),
             'ping_min':min(p['number'] for p in pings),'ping_max':max(p['number'] for p in pings),'gdf_files':len(manifest),'slc_shape':shape,
             'image_bounds_xy_m':bounds,'in_frame_insertion_points':int(in_frame.sum())}
        assert in_frame.sum()==(0 if i==0 else 9)
        rows.append(row);del pings
    for item in json.loads((ROOT/'references/download_manifest_v2.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes']
        with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    (ROOT/'results/v2/all_scene_gdf_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Distinct family labels across all three sets. Uniform constituents still
    # occur inside mixed test terrain; this is not independence of all physics.
    families={0:'isolated_points',1:'uniform',2:'rough_uniform',3:'rough_uniform',4:'ripple',5:'mixed_uniform_ripple',6:'uniform',7:'rough_uniform',8:'rough_uniform',9:'ripple',10:'mixed_uniform_ripple'}
    sets={0:'beamformer_development',1:'validation',2:'train',3:'train',4:'test',5:'test',6:'validation',7:'train',8:'train',9:'test',10:'test'}
    objects=np.loadtxt(ROOT/'references/Object_Locations_All_Scenes.csv',delimiter=',',skiprows=1)
    split=[]
    for i in range(11):
        count=int(np.sum(objects[:,1]==i));split.append({'scene':f'scene{i:02d}','family':families[i],'split':sets[i],'object_placements':count})
        assert count==(0 if i==0 else 9)
    family_sets={s:{r['family'] for r in split if r['split']==s} for s in ['train','validation','test']}
    assert not family_sets['train']&family_sets['validation'] and not family_sets['train']&family_sets['test'] and not family_sets['validation']&family_sets['test']
    (ROOT/'results/v2/scene_split.json').write_text(json.dumps({'rows':split,'counts':{'train':36,'validation':18,'test':36},'status':'Frozen plan for subsequent object model experiments; no detector trained in this checkpoint',
        'caveats':['Only 90 placements, not 90 proven independent target designs','Mixed terrain shares a uniform/ripple constituent; family-disjoint is not independence of every physical factor','CSV gives insertion positions and Euler rotations, not box/mask/visible-shadow extents','Point-distance FROC can be defined; bbox mAP requires independently checked box annotations']},indent=2)+'\n')
    with (ROOT/'references/scene_split_v2.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(split[0]));w.writeheader();w.writerows(split)
    print('55 pinned products verified; 11 complete GDF scenes parsed; split train36/val18/test36 placements')


if __name__=='__main__':main()
