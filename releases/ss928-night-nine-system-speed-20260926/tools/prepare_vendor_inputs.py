"""Convert existing normalized vectors to the declared model I/O, without features."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto
p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--vector',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();model=onnx.load(str(a.onnx));a.output_dir.mkdir(parents=True,exist_ok=True);record=[]
with np.load(a.vector,allow_pickle=False) as vectors:
 for v in model.graph.input:
  shape=[d.dim_value for d in v.type.tensor_type.shape.dim];dtype={TensorProto.FLOAT:np.float32,TensorProto.FLOAT16:np.float16}.get(v.type.tensor_type.elem_type)
  if dtype is None:raise ValueError('Unsupported input dtype')
  if shape in ([1,9,1024,1280],[1,1024,1280,9]):
   value=vectors['nine_raw']
   if tuple(value.shape)!=(1,9,1024,1280):raise ValueError('Invalid normalized nine-frame vector')
   if shape[3]==9:value=value.transpose(0,2,3,1)
  elif shape==[1,1,64,64]:value=vectors['reference_thumb']
  else:raise ValueError(f'Unexpected complete-model input {shape}')
  value=np.ascontiguousarray(value,dtype=dtype)
  if not np.isfinite(value).all():raise ValueError('Nonfinite input')
  # Graph input names are data; do not let them name arbitrary filesystem paths.
  label='nine_raw' if value.size==9*1024*1280 else 'reference_thumb';path=a.output_dir/f'{label}.{np.dtype(dtype).name}.bin'
  with path.open('xb') as f:f.write(value.tobytes())
  record.append({'model_input':v.name,'path':path.name,'shape':shape,'dtype':np.dtype(dtype).name,'bytes':int(value.nbytes)})
(a.output_dir/'inputs.json').write_text(json.dumps({'inputs':record,'vector':str(a.vector),'model':str(a.onnx),'network_features_outside_model':False},indent=2));print(json.dumps(record,indent=2))
