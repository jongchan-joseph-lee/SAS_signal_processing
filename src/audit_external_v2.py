"""Read-only format/download checks; no author contact, no new training."""
from pathlib import Path
import argparse,csv,io,json
import requests


def main():
    p=argparse.ArgumentParser();p.add_argument('--airsas',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    air={'source':'https://www.nature.com/articles/s41597-024-04050-0','format':'HDF5 with real-voltage time series, not GDF','local_file_checked':False}
    if a.airsas:
        import h5py
        from gdf import read_gdf
        with h5py.File(a.airsas) as f:air.update({'local_file_checked':True,'groups':list(f),'ts_shape':list(f['ch1/ts'].shape)})
        try:read_gdf([a.airsas]);air['gdf_parser']='unexpected acceptance'
        except ValueError as e:air['gdf_parser']='Rejected as expected: '+str(e)
    sctd={}
    for endpoint in ['contents','releases']:
        r=requests.get('https://api.github.com/repos/automlresearch/SCTD-3.0/'+endpoint,timeout=60);r.raise_for_status();d=r.json()
        sctd[endpoint]=[x['name'] for x in d] if endpoint=='contents' else d
    sctd.update({'data_license':'Unverified: repository MIT is not proof of dataset licensing','paper_240k_classes':13,'paper_450k_classes':10})
    repo='idg101/Aperture_Lab_Synthetic_Aperture_Sonar_v1_1';base='https://huggingface.co/datasets/'+repo+'/resolve/main/'
    r=requests.get(base+'manifest.csv',timeout=60);r.raise_for_status();rows=list(csv.DictReader(io.StringIO(r.text)));entry=rows[0]
    r=requests.get(base+'LICENSE.md',timeout=60);r.raise_for_status();license_text=r.text
    with requests.get(base+entry['drc_path'],headers={'Range':'bytes=0-15'},stream=True,timeout=60) as r:
        r.raise_for_status();head=next(r.iter_content(16));hf={'repo':repo,'manifest_rows':len(rows),'image_http':r.status_code,'image_header_hex':head.hex(),'content_range':r.headers.get('Content-Range'),'license':'ALSD-1.0','named_exclusions_present':all(x in license_text for x in ['Kitware','PMS 408','PMS 495','NSWC PCD']),'status':'Access check only; not used for training'}
    output={'air_sas':air,'sctd3':sctd,'aperturelab_v1_1':hf}
    (root/'results/v2/external_data_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
