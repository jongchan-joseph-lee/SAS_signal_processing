"""Acquire remaining public PoSSM raw and SLC products, scene by scene."""
from pathlib import Path
import gzip,hashlib,json,shutil,tarfile,time,tempfile,xml.etree.ElementTree as ET
import requests

ROOT=Path(__file__).resolve().parents[1]
API='https://api.marine-geo.org/services/download/download_accept.php'


def main():
    (ROOT/'data').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'references/download_manifest.json').read_text())
    out=ROOT/'references/download_manifest_v2.json'
    if out.exists():manifest=json.loads(out.read_text())
    files={x['path']:x for x in manifest['files']}
    catalogs=[]
    for uid in [31898,31899]:
        r=requests.get('https://www.marine-geo.org/services/xml/datafileservice.php',params={'data_set_uid':uid},timeout=60);r.raise_for_status()
        (ROOT/'references'/f'mgds_{uid}_catalog.xml').write_bytes(r.content)
        for e in ET.fromstring(r.content).findall('.//files/file'):
            name=e.findtext('file_info/file_name');data_uid=int(e.attrib['data_uid'])
            scene=name[name.index('scene'):name.index('scene')+7]
            dest=ROOT/'data'/('possm_reference' if uid==31899 else 'possm_'+scene)/name
            catalogs.append({'data_uid':data_uid,'source_gz':name+'.gz','path':str(dest.relative_to(ROOT)),'scene':scene})
    for number in range(11):
        scene=f'scene{number:02d}'
        wanted={}
        for item in catalogs:
            if item['scene']!=scene:continue
            dest=ROOT/item['path'];pin=files.get(item['path'])
            valid=dest.is_file()
            if valid and pin:
                with dest.open('rb') as src:valid=dest.stat().st_size==pin['bytes'] and hashlib.file_digest(src,'sha256').hexdigest()==pin['sha256']
            if not valid:wanted[item['source_gz']]=item
        if not wanted:continue
        start=time.perf_counter();seen=set()
        form={'purpose':'Educational','client':'DataLink','force_download':'1','data_uids':','.join(str(i['data_uid']) for i in wanted.values()),'doc_uids':''}
        with tempfile.TemporaryDirectory(prefix='possm-all-',dir=ROOT/'data') as tmp:
            archive=Path(tmp)/'download.tar'
            with requests.post(API,data=form,stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                with archive.open('wb') as f:
                    for block in r.iter_content(1024*1024):f.write(block)
            with tarfile.open(archive,mode='r:*') as tar:
                for member in tar:
                    name=Path(member.name).name
                    if name not in wanted:continue
                    item=wanted[name];dest=ROOT/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix(dest.suffix+'.part')
                    with tar.extractfile(member) as packed:raw=gzip.decompress(packed.read())
                    temp.write_bytes(raw)
                    item.update({'bytes':temp.stat().st_size,'sha256':hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()})
                    pin=files.get(item['path'])
                    if pin and item['sha256']!=pin['sha256']:raise ValueError('Pinned input changed: '+item['path'])
                    temp.replace(dest);files[item['path']]=item;seen.add(name)
                    print(scene,dest.name,item['bytes'],flush=True)
        if seen!=set(wanted):raise ValueError('Missing products: '+str(set(wanted)-seen))
        manifest['files']=list(files.values());manifest['scope']='All eleven scenes; raw 44 GDF and 11 SLC files; hashes pinned at retrieval, not publisher signatures'
        out.write_text(json.dumps(manifest,indent=2)+'\n');print(scene,'complete',time.perf_counter()-start,flush=True)
    assert len(manifest['files'])==55
    from gdf import read_gdf
    for i in range(11):
        _,_,pings,_=read_gdf((ROOT/'data'/f'possm_scene{i:02d}').glob('*.gdf'))
        assert len(pings)==100
        del pings
    print('COMPLETE: 44 raw GDF + 11 SLC',flush=True)


if __name__=='__main__':main()
