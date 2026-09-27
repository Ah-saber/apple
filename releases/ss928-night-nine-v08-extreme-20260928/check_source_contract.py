"""Verify real input bytes and typed source outputs for every deployed graph."""
import json,hashlib
from pathlib import Path
import numpy as np
import onnx
r=Path(__file__).parent;rows=[]
for scene,frame in [('ordinary',20),('special',60)]:
 v=np.load(r/'test_vectors'/f'{scene}_frame_{frame}.npz');raw=v['nine_raw'];ctx=v['reference_thumb'];padded=np.pad(raw,((0,0),(0,7),(0,0),(0,0)));layouts={(1,9,1024,1280):raw,(1,16,1024,1280):padded,(1,1024,1280,16):padded.transpose(0,2,3,1),(1,1,1024,1280,16):padded.transpose(0,2,3,1)[:,None]};cache={}
 for manifest in [r/'reference_outputs'/f'{scene}_manifest.json',r/'reference_outputs'/f'{scene}_supplement_manifest.json']:
  for row in json.loads(manifest.read_text())['models']:
   g=onnx.load(r/'source_onnx'/row['graph_file']);declarations={i.name:i.type.tensor_type for i in g.graph.input}
   for item in row['inputs']:
    t=declarations[item['name']];shape=tuple(d.dim_value for d in t.shape.dim);dtype=np.float16 if t.elem_type==10 else np.float32;key=(scene,item['name'],shape,str(dtype))
    if key not in cache:
     a=np.ascontiguousarray(ctx if item['name']=='reference_thumb' else layouts[shape],dtype=dtype);cache[key]=(hashlib.sha256(a.tobytes()).hexdigest(),a.nbytes,str(a.dtype))
    digest,size,type_name=cache[key];assert digest==item['sha256'] and size==item['bytes'] and type_name==item['dtype'],(scene,row['case'],item['name'])
   path=r/'reference_outputs'/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['npz_sha256'];y=np.load(path)['output'];assert list(y.shape)==row['output_shape'];assert str(y.dtype)==row['output_dtype'];assert np.isfinite(y).all();assert hashlib.sha256(y.tobytes()).hexdigest()==row['output_sha256_compact'];rows.append({'scene':scene,'case':row['case'],'source_input_bytes_verified':True,'typed_output_verified':True,'SDK_verified':False})
(r/'source_contract_checks.json').write_text(json.dumps({'count':len(rows),'results':rows},indent=2));print('SOURCE_CONTRACT_VERIFIED',len(rows))
