"""Read ONNX declarations and write exact nine-frame/16-slot input bytes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
p=argparse.ArgumentParser();p.add_argument('--npz',required=True);p.add_argument('--onnx',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);g=onnx.load(a.onnx);v=np.load(a.npz);raw=v['nine_raw'];context=v['reference_thumb'];assert raw.shape==(1,9,1024,1280);assert context.shape==(1,1,64,64);manifest=[]
for declared in g.graph.input:
 t=declared.type.tensor_type;shape=tuple(d.dim_value for d in t.shape.dim);dtype={onnx.TensorProto.FLOAT:np.float32,onnx.TensorProto.FLOAT16:np.float16}.get(t.elem_type)
 if dtype is None:raise ValueError('Unsupported declared input type')
 if declared.name=='reference_thumb':x=context
 elif shape==(1,9,1024,1280):x=raw
 else:
  padded=np.pad(raw,((0,0),(0,7),(0,0),(0,0)))
  if shape==(1,16,1024,1280):x=padded
  elif shape==(1,1024,1280,16):x=padded.transpose(0,2,3,1)
  elif shape==(1,1,1024,1280,16):x=padded.transpose(0,2,3,1)[:,None]
  else:raise ValueError('Graph input is not a supported complete-model interface')
 x=np.ascontiguousarray(x,dtype=dtype);assert x.shape==shape;assert np.isfinite(x).all();filename=declared.name+'.bin';x.tofile(out/filename);np.save(out/(declared.name+'.npy'),x);manifest.append({'name':declared.name,'file':filename,'shape':list(shape),'dtype':str(x.dtype),'bytes':x.nbytes,'sha256':hashlib.sha256(x.tobytes()).hexdigest()})
(out/'manifest.json').write_text(json.dumps({'graph':Path(a.onnx).name,'graph_sha256':hashlib.sha256(Path(a.onnx).read_bytes()).hexdigest(),'source':Path(a.npz).name,'inputs':manifest,'nine_actual_frames':True,'seven_zero_slots_when_declared':True,'representation_only_no_statistics_or_reference_model_execution':True},indent=2))
