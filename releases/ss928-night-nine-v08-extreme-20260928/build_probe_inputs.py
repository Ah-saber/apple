"""Generate deterministic dynamic input bytes for exported full-shape probes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);rows=[]
for row in json.loads(Path(a.manifest).read_text())['probes']:
 dtype=np.float16 if row['input_dtype']=='torch.float16' else np.float32;rng=np.random.default_rng(row['runtime_input_seed']);x=rng.uniform(*row['input_range'],size=row['input_shape']).astype(dtype);path=out/(row['file'].replace('.onnx','.input.bin'));x.tofile(path);rows.append({'path':path.name,'shape':row['input_shape'],'dtype':str(x.dtype),'bytes':x.nbytes,'sha256':hashlib.sha256(x.tobytes()).hexdigest(),'synthetic_not_quality_input':True})
(out/'manifest.json').write_text(json.dumps({'inputs':rows},indent=2))
