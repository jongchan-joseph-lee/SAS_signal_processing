"""Check self-contained HTML and embedded ZIP without browser availability."""
from pathlib import Path
import base64,hashlib,io,json,re,subprocess,zipfile
from html.parser import HTMLParser
from PIL import Image
from protocol_v3 import ROOT,OUT


class Parse(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.targets=[];self.images=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='a' and a.get('href','').startswith('#'):self.targets.append(a['href'][1:])
  if tag=='img':self.images.append(a.get('src',''))


def main():
 p=ROOT/'deliverables/SAS_Focus_Report_v3.html';s=p.read_text();h=Parse();h.feed(s)
 assert len(h.ids)==len(set(h.ids)) and all(v in h.ids for v in h.targets)
 assert len(h.images)==7
 for img in h.images:
  assert img.startswith('data:image/png;base64,')
  with Image.open(io.BytesIO(base64.b64decode(img.split(',',1)[1]))) as im:im.verify()
 b=base64.b64decode(re.search(r'<script id="zip-data"[^>]*>(.*?)</script>',s,re.S).group(1));digest=hashlib.sha256(b).hexdigest()
 assert digest in s and b==(ROOT/'deliverables/SAS_Focus_Reproducibility_v3.zip').read_bytes()
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  assert z.testzip() is None
  assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
  assert not any(n.endswith('.gdf') or 'new_dataset_audit' in n or 'ASASIN_2101' in n for n in z.namelist())
  assert any(n.endswith('point_detector.joblib') for n in z.namelist())
  entries=len(z.namelist())
  sel=json.loads(z.read('sas_focus_benchmark/results/v3/detection_selection.json'))
  assert hashlib.sha256(z.read('sas_focus_benchmark/results/v3/point_detector.joblib')).hexdigest()==sel['model_sha256']
 js='\n'.join(re.findall(r'<script>(.*?)</script>',s,re.S));path=ROOT/'data/v3_report_script.js';path.parent.mkdir(exist_ok=True);path.write_text(js)
 subprocess.run(['node','--check',str(path)],check=True,capture_output=True)
 result={'html_bytes':p.stat().st_size,'zip_bytes':len(b),'zip_sha256':digest,'zip_entries':entries,'embedded_pngs':len(h.images),
  'checks':['HTML anchor/ID integrity','7 embedded PNG images decode','standalone and embedded ZIP byte-identical','ZIP CRC and safe relative paths','model checksum matches selected model','download JavaScript syntax'],
  'browser_full_render':False,'scientific_pngs_visually_inspected':True}
 (OUT/'delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
