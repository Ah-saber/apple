"""Expand complete real model inputs and current model reference outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=Path)
p.add_argument('--input-precision',choices=('float16','float32'),default='float16')
a=p.parse_args();package=Path(__file__).resolve().parents[1]
a.output.mkdir(parents=True,exist_ok=False);manifest={}
for scene,frame in [('ordinary',20),('special',60)]:
 with np.load(package/'input_vectors'/f'{scene}_frame_{frame}.npz',allow_pickle=False) as v:
  dtype=np.dtype('<f2' if a.input_precision=='float16' else '<f4')
  arrays={'nine_raw':np.ascontiguousarray(v['nine_raw'],dtype=dtype),'reference_thumb':np.ascontiguousarray(v['reference_thumb'],dtype=np.dtype('<f4'))}
 with np.load(package/'reference_outputs'/(scene+'_reference_outputs.npz'),allow_pickle=False) as ref:
  for name in ('source','compiled'):arrays['display_uint8_'+name]=np.ascontiguousarray(ref[a.input_precision+'_'+name])
 expected={'nine_raw':((1,9,1024,1280),dtype),'reference_thumb':((1,1,64,64),np.dtype('<f4'))}
 manifest[scene]={}
 for name,array in arrays.items():
  shape,kind=expected.get(name,((1,1,3072,3840),np.dtype('u1')))
  if array.shape!=shape or array.dtype!=kind:raise ValueError(f'{scene}/{name}: unexpected shape or dtype')
  if name in expected and not np.isfinite(array).all():raise ValueError('Non-finite input')
  suffix='u8' if name.startswith('display') else ('f16' if name=='nine_raw' and a.input_precision=='float16' else 'f32')
  path=a.output/(scene+'_'+name+'.'+suffix);path.write_bytes(array.tobytes(order='C'))
  manifest[scene][name]={'path':path.name,'shape':list(shape),'dtype':str(kind),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
