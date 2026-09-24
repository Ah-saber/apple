"""Materialize normalized samples and calibration inputs without GT or images."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
for folder in ['samples','calibration']:
 for meta in sorted((ROOT/folder).glob('*/metadata.json')):
  m=json.loads(meta.read_text());src=meta.parent/'raw.u16';assert hashlib.sha256(src.read_bytes()).hexdigest()==m['raw_sha256']
  raw=np.fromfile(src,dtype='<u2');assert raw.size==1024*1280;n=m['normalization'];assert np.isfinite([n['offset'],n['scale']]).all() and n['scale']>0
  x=((raw.astype(np.float32)-n['offset'])/n['scale']).astype('<f4');assert hashlib.sha256(x.tobytes()).hexdigest()==m['normalized_input_sha256']
  out=a.output/folder/(meta.parent.name+'.f32');out.parent.mkdir(exist_ok=True);x.tofile(out)
  rows.append({'kind':folder,'group':m['group'],'sample':m['record']['sample_id'],'path':str(out.resolve()),'sha256':m['normalized_input_sha256'],'range':m['input_range']})
(a.output/'inputs.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'status':'passed','inputs':len(rows),'manifest':str(a.output/'inputs.json')}))
