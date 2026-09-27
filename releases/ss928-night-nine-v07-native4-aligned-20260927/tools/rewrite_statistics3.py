"""Replace validated even-sensor statistics2 with statistics3, optional ReLU32.
Keep native reference/body/output paths. Does not infer SDK timing.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
from rewrite_hotspots import constants,rewrite_statistics
from materialize_aliases import materialize

def rewrite(model,rectified=False):
 model,_=materialize(model);rewrite_statistics(copy.deepcopy(model),'space_pack') # Guard actual frozen balanced coefficients.
 c=constants(model);found=[n for n in model.graph.node if n.op_type=='Conv' and n.input[1] in c and c[n.input[1]].shape==(12,9,2,2)]
 if len(found)!=1:raise ValueError('Require unique statistics2')
 old=found[0];inferred=onnx.shape_inference.infer_shapes(model);info={v.name:v for v in list(inferred.graph.value_info)+list(inferred.graph.input)};v=info.get(old.input[0]);shape=[] if v is None else [d.dim_value for d in v.type.tensor_type.shape.dim]
 if len(shape)!=4 or shape[1]!=9 or not all(shape[-2:]) or shape[-2]%2 or shape[-1]%2:raise ValueError('Require known even NCHW nine-frame sensor input')
 prefix='continued_stats3__'
 if any(v.startswith(prefix) for n in model.graph.node for v in list(n.input)+list(n.output)):raise ValueError('Already rewritten')
 w=c[old.input[1]];channels=32 if rectified else 12;kernel=np.zeros((channels,9,3,3),w.dtype);kernel[:12,:,1:,1:]=w
 if rectified:kernel[16:28,:,1:,1:]=-w
 def init(label,a):name=prefix+label;model.graph.initializer.append(numpy_helper.from_array(a,name));return name
 dest=prefix+'linear' if rectified else old.output[0];nodes=[helper.make_node('Conv',[old.input[0],init('weight',kernel),init('bias',np.zeros(channels,w.dtype))],[dest],kernel_shape=[3,3],strides=[2,2],pads=[1]*4)]
 if rectified:
  axes=init('axis',np.array([1],np.int64));nodes.append(helper.make_node('Relu',[dest],[prefix+'relu']))
  for name,start,end in [('positive',0,12),('negative',16,28)]:nodes.append(helper.make_node('Slice',[prefix+'relu',init(name+'_start',np.array([start],np.int64)),init(name+'_end',np.array([end],np.int64)),axes],[prefix+name]))
  nodes.append(helper.make_node('Sub',[prefix+'positive',prefix+'negative'],list(old.output)))
 result=[]
 for n in model.graph.node:result.extend(nodes if n==old else [n])
 del model.graph.node[:];model.graph.node.extend(result);del model.graph.value_info[:];onnx.checker.check_model(model);return model

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rectified32',action='store_true');a=p.parse_args()
 if a.output.exists() or a.output.with_suffix('.stats3.json').exists():raise FileExistsError(a.output)
 m=rewrite(onnx.load(a.native),a.rectified32);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,a.output);meta={'input_sha256':hashlib.sha256(a.native.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'rectified32':a.rectified32,'requires_even_sensor_input':True,'SDK_verified':False};a.output.with_suffix('.stats3.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
