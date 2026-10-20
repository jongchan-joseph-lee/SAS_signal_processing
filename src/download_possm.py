"""Public MGDS download form, extraction and SHA-256 verification.

This requests files from the repository, not from the authors. Review the
CC BY-NC-SA 3.0 source terms in THIRD_PARTY.md before reuse.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,tarfile,tempfile
import requests

BASE='https://api.marine-geo.org/services/download/'


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    root=a.root;manifest=json.loads((root/'references'/'download_manifest.json').read_text())
    (root/'data').mkdir(parents=True,exist_ok=True)
    pending=[]
    for item in manifest['files']:
        dest=root/item['path']
        if dest.exists() and sha256(dest)==item['sha256']:
            print('Verified existing',dest.name,flush=True)
        else:pending.append(item)
    if pending:
        with tempfile.TemporaryDirectory(prefix='possm-',dir=root/'data') as temp:
            temp=Path(temp);archive=temp/'download.tar'
            form={'purpose':'Educational','client':'DataLink','force_download':'1',
                  'data_uids':','.join(str(i['data_uid']) for i in pending),'doc_uids':''}
            with requests.post(BASE+'download_accept.php',data=form,stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                with archive.open('wb') as f:
                    for block in r.iter_content(1024*1024):f.write(block)
            wanted={i['source_gz']:i for i in pending};seen=set()
            with tarfile.open(archive,'r:*') as tar:
                for member in tar:
                    name=Path(member.name).name
                    if name not in wanted:continue
                    if not member.isfile():raise ValueError('Expected regular archive member')
                    item=wanted[name];dest=root/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
                    # No archive paths are used for extraction.
                    with tar.extractfile(member) as packed,gzip.GzipFile(fileobj=packed) as src,dest.open('wb') as output:
                        shutil.copyfileobj(src,output)
                    if dest.stat().st_size!=item['bytes'] or sha256(dest)!=item['sha256']:
                        raise ValueError('Checksum/size mismatch: '+str(dest))
                    seen.add(name);print('Downloaded and verified',dest.name,flush=True)
            if seen!=set(wanted):raise ValueError('Missing files: '+str(set(wanted)-seen))
    for uid,name in [(5059,'Description_of_Simulated_SAS_Data_Package.pdf'),(5060,'PSU_GDF_Format_20240612.pdf')]:
        dest=root/'references'/name
        if not dest.exists():
            r=requests.get(BASE+'Document_Accept.php',params={'client':'DataLink','doc_uid':uid},timeout=60)
            r.raise_for_status();dest.write_bytes(r.content)
    print('Complete: both raw scenes and both complex reference products verified.')


if __name__=='__main__':main()
