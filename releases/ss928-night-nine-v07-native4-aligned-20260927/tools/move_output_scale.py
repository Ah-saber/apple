"""Move output clip/scale to small phase tensor; only supported single layout paths.
Exact movement preserves source arithmetic. Folded mode rounds scaled weights.
Actual SDK mapping and mixed-precision conversion remain unverified.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from materialize_aliases import materialize

def attrs(n):return {a.name:helper.get_attribute_value(a) for a in n.attribute}

def move_scale(model,mode='exact_phase'):
 if mode not in ['exact_phase','fold_phase']:raise ValueError('Unsupported mode')
 model,_=materialize(model);prefix='continued_small_phase_scale__';ci={v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
 for n in model.graph.node:
  if n.op_type=='Constant' and 'value' in attrs(n):ci[n.output[0]]=numpy_helper.to_array(attrs(n)['value'])
 if any(v.startswith(prefix) for n in model.graph.node for v in list(n.input)+list(n.output)):raise ValueError('Already rewritten')
 candidates=[n for n in model.graph.node if n.op_type=='Conv' and n.input[1] in ci and ci[n.input[1]].shape in [(36,16,3,3),(9,16,3,3)]]
 if len(candidates)!=1:raise ValueError('Need unique output projection Conv16-to-36/9')
 conv=candidates[0]
 if len(conv.input)!=3 or conv.input[2] not in ci:raise ValueError('Explicit bias required')
 w,b=ci[conv.input[1]],ci[conv.input[2]];a=attrs(conv)
 if w.dtype not in (np.float16,np.float32) or b.dtype!=w.dtype or b.shape!=(w.shape[0],) or a.get('group',1)!=1:raise ValueError('Invalid output projection')
 consumers={}
 for n in model.graph.node:
  for v in n.input:consumers.setdefault(v,[]).append(n)
 if len(model.graph.output)!=1:raise ValueError('Single display output required')
 end=model.graph.output[0].name;value=conv.output[0];layouts=[];point=[];started_point=False
 while value!=end:
  ns=consumers.get(value,[])
  if len(ns)!=1:raise ValueError('Require single output-layout chain; split native paths unsupported')
  n=ns[0];at=attrs(n)
  if n.op_type=='DepthToSpace' and not started_point:
   if at.get('mode',b'DCR')!=b'CRD' or at.get('blocksize') not in (3,6):raise ValueError('Unsupported phase order')
   layouts.append(n)
  elif n.op_type=='Resize' and not started_point:
   if at.get('mode',b'nearest')!=b'nearest' or at.get('coordinate_transformation_mode')!=b'asymmetric' or at.get('nearest_mode')!=b'floor':raise ValueError('Only exact nearest expansion supported')
   layouts.append(n)
  elif n.op_type in ['Cast','Clip','Mul']:
   started_point=True;dynamic=[i for i in n.input if i and i not in ci]
   if dynamic!=[value] or any(ci[i].size!=1 for i in n.input if i in ci):raise ValueError('Only scalar output arithmetic supported')
   point.append(n)
  else:raise ValueError('Unsupported output chain '+n.op_type)
  value=n.output[0]
 if [n.op_type for n in point]!=['Cast','Clip','Mul'] or not layouts:raise ValueError('Require layout followed by Cast, Clip, Mul')
 if attrs(point[0]).get('to')!=TensorProto.FLOAT:raise ValueError('Expected F32 display arithmetic')
 clip=point[1];mul=point[2]
 if len(clip.input)!=3 or float(ci[clip.input[1]])!=0 or float(ci[clip.input[2]])!=1:raise ValueError('Expected clip01')
 gain=[float(ci[i]) for i in mul.input if i in ci]
 if gain!=[255.]:raise ValueError('Expected gray scale255')
 def init(name,array):model.graph.initializer.append(numpy_helper.from_array(array,name));return name
 extra=[]
 if mode=='exact_phase':
  value=conv.output[0]
  for index,n in enumerate(point):
   nn=copy.deepcopy(n);old_dynamic=next(i for i in n.input if i and i not in ci)
   for j,i in enumerate(nn.input):
    if i==old_dynamic:nn.input[j]=value
   value=prefix+str(index);nn.output[0]=value;nn.name=value;extra.append(nn)
  layouts[0].input[0]=value;tail=helper.make_node('Identity',[layouts[-1].output[0]],[end])
 else:
  conv.input[1]=init(prefix+'weight',(w.astype(np.float64)*255).astype(w.dtype));conv.input[2]=init(prefix+'bias',(b.astype(np.float64)*255).astype(b.dtype))
  low=init(prefix+'low',np.array(0,w.dtype));high=init(prefix+'high',np.array(255,w.dtype));value=prefix+'clipped'
  extra=[helper.make_node('Clip',[conv.output[0],low,high],[value])];layouts[0].input[0]=value;tail=helper.make_node('Cast',[layouts[-1].output[0]],[end],to=TensorProto.FLOAT)
 nodes=[]
 for n in model.graph.node:
  if n in point:continue
  nodes.append(n)
  if n==conv:nodes.extend(extra)
  if n==layouts[-1]:nodes.append(tail)
 del model.graph.node[:];model.graph.node.extend(nodes);lookup={o:n for n in nodes for o in n.output};live=set()
 def visit(v):
  if v in live:return
  live.add(v)
  if v in lookup:
   for i in lookup[v].input:
    if i:visit(i)
 visit(end);pending=[n for n in model.graph.node if any(o in live for o in n.output)];inits=[v for v in model.graph.initializer if v.name in live]
 available={v.name for v in model.graph.input}|{v.name for v in inits};nodes=[]
 while pending:
  ready=[n for n in pending if all(not i or i in available for i in n.input)]
  if not ready:raise ValueError('Rewritten graph has missing inputs or a dependency cycle')
  for n in ready:nodes.append(n);available.update(n.output);pending.remove(n)
 del model.graph.node[:];model.graph.node.extend(nodes);del model.graph.initializer[:];model.graph.initializer.extend(inits);del model.graph.value_info[:];onnx.checker.check_model(model);return model

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['exact_phase','fold_phase'],required=True);a=p.parse_args()
 if a.output.exists() or a.output.with_suffix('.scale.json').exists():raise FileExistsError(a.output)
 m=move_scale(onnx.load(a.native),a.mode);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,a.output)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();meta={'input_sha256':sha(a.native),'output_sha256':sha(a.output),'mode':a.mode,'SDK_verified':False,'NPU_measured':False,'finite_precision_weights_changed':a.mode=='fold_phase'};a.output.with_suffix('.scale.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
if __name__=='__main__':main()
