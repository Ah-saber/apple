"""Conservative pure-model graph controls, retaining native_r2 compatibility graph."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from materialize_aliases import materialize


def phase_permutation(first,second):
 return [(second*i+p)*6+second*j+q for p in range(second) for q in range(second) for i in range(first) for j in range(first)]


def rewrite(model,order='six',fixed_pack=False):
 m,aliases=materialize(model);report={'initializer_aliases':aliases,'order':order,'fixed_pack':fixed_pack,'NPU_verified':False,'real_native_r2_verified':False}
 def producers():return {v:n for n in m.graph.node for v in n.output}
 def initializers():return {v.name:numpy_helper.to_array(v) for v in m.graph.initializer}
 if order!='six':
  prod=producers();init=initializers()
  def const(v):
   if v in init:return init[v]
   n=prod.get(v)
   if n is not None and n.op_type=='Constant':
    t=next((a.t for a in n.attribute if a.name=='value'),None)
    if t is not None:return numpy_helper.to_array(t)
   return None
  matches=[n for n in m.graph.node if n.op_type=='DepthToSpace' and next((a.i for a in n.attribute if a.name=='blocksize'),0)==6]
  if len(matches)!=1:raise ValueError('Expected one undecomposed final DepthToSpace6')
  shuffle=matches[0]
  if next((a.s for a in shuffle.attribute if a.name=='mode'),b'DCR')not in (b'CRD',b'DCR'):raise ValueError('Unsupported output phase order')
  v=shuffle.input[0];chain=[]
  while v in prod:
   n=prod[v]
   if n.op_type=='Conv':break
   if n.op_type in ('Identity','Cast'):
    previous=n.input[0]
   elif n.op_type=='Clip' and all(const(k) is not None and const(k).size==1 for k in n.input[1:] if k):
    previous=n.input[0]
   elif n.op_type=='Mul':
    scalar=[k for k in n.input if const(k) is not None and const(k).size==1]
    if len(scalar)!=1:raise ValueError('Cannot commute non-scalar output multiply')
    previous=next(k for k in n.input if k!=scalar[0])
   else:raise ValueError('Cannot safely trace phase channels through '+n.op_type)
   chain.append(n);v=previous
  conv=prod.get(v)
  if conv is None or conv.op_type!='Conv' or conv.input[1] not in init or tuple(init[conv.input[1]].shape)!=(36,16,3,3):raise ValueError('Expected final Conv36x16x3x3')
  if next((a.i for a in conv.attribute if a.name=='group'),1)!=1:raise ValueError('Grouped output convolution is unsupported')
  # Channel permutations must not change any other observable branch.
  ordered=[conv]+list(reversed(chain))+[shuffle]
  output_names={o.name for o in m.graph.output}
  for node,nxt in zip(ordered,ordered[1:]):
   for val in node.output:
    if val in output_names or any(val in p.input and p is not nxt and p!=nxt and p.op_type!='Shape' for p in m.graph.node):raise ValueError('Output phase tensor has other consumers')
  first,second=(2,3) if order=='two_three' else (3,2);permutation=phase_permutation(first,second)
  for j,key in enumerate(list(conv.input)[1:],1):
   if key not in init:raise ValueError('Output learned parameter is not a constant initializer')
   value=init[key]
   if value.shape[0]!=36:raise ValueError('Output parameter channel dimension differs')
   name=key+'__'+order
   if name in init:raise ValueError('Generated initializer collision')
   m.graph.initializer.append(numpy_helper.from_array(value[permutation].copy(),name));conv.input[j]=name
  middle='multistage_output__'+order
  n1=helper.make_node('DepthToSpace',[shuffle.input[0]],[middle],name=middle+'__first',blocksize=first,mode='CRD')
  n2=helper.make_node('DepthToSpace',[middle],list(shuffle.output),name=middle+'__second',blocksize=second,mode='CRD')
  nodes=[]
  for n in m.graph.node:
   if n is shuffle or n==shuffle:nodes.extend([n1,n2])
   else:nodes.append(n)
  del m.graph.node[:];m.graph.node.extend(nodes);report['phase_permutation']=permutation
 if fixed_pack:
  inferred=onnx.shape_inference.infer_shapes(m);types={v.name:v.type.tensor_type for v in list(inferred.graph.value_info)+list(inferred.graph.input)+list(inferred.graph.output)}
  prod=producers();raw=[v for v in m.graph.input if len(v.type.tensor_type.shape.dim)==4 and v.type.tensor_type.shape.dim[1].dim_value==9]
  if len(raw)!=1 or raw[0].type.tensor_type.elem_type!=TensorProto.FLOAT:raise ValueError('Fixed FP32 pack requires unique FP32 nine-frame input')
  candidates=[]
  for n in m.graph.node:
   if n.op_type!='Reshape':continue
   t=prod.get(n.input[0]);r=prod.get(t.input[0]) if t is not None else None
   if t is None or t.op_type!='Transpose' or r is None or r.op_type!='Reshape':continue
   perm=next((list(a.ints) for a in t.attribute if a.name=='perm'),[])
   if perm!=[0,1,3,5,2,4]:continue
   tp=types.get(n.output[0]);rp=types.get(r.input[0])
   if tp is None or rp is None:continue
   dims=[d.dim_value for d in tp.shape.dim];before=[d.dim_value for d in rp.shape.dim]
   rd=[d.dim_value for d in raw[0].type.tensor_type.shape.dim]
   if dims==[rd[0],4,rd[2]//2,rd[3]//2] and before==[rd[0],1,rd[2],rd[3]] and tp.elem_type==TensorProto.FLOAT:candidates.append((n,r))
  if len(candidates)!=1:raise ValueError('Expected unique FP32 current-frame pixel-unshuffle2 pattern')
  target,reshape=candidates[0]
  # Verify the one-channel source is the last frame, never merely infer from shape.
  v=reshape.input[0]
  while v in prod and prod[v].op_type in ('Cast','Identity'):v=prod[v].input[0]
  sl=prod.get(v)
  if sl is None or sl.op_type!='Slice':raise ValueError('Current pack input is not a verified final-frame Slice')
  ini=initializers()
  def literal(v):
   if v in ini:return ini[v]
   node=prod.get(v)
   if node is not None and node.op_type=='Constant':
    t=next((a.t for a in node.attribute if a.name=='value'),None)
    if t is not None:return numpy_helper.to_array(t)
   raise ValueError('Slice has dynamic constants')
  starts,ends,axes=[literal(v).reshape(-1).tolist() for v in sl.input[1:4]]
  step=literal(sl.input[4]).reshape(-1).tolist() if len(sl.input)>4 else [1]
  if starts not in ([-1],[8]) or axes!=[1] or len(ends)!=1 or ends[0]<9 or step!=[1]:raise ValueError('Slice is not channel eight only')
  v=sl.input[0]
  while v in prod and prod[v].op_type in ('Cast','Identity'):v=prod[v].input[0]
  if v!=raw[0].name:raise ValueError('Current pack Slice is not sourced directly from nine_raw')
  weight=np.zeros((4,9,2,2),dtype=np.float32)
  for dy in range(2):
   for dx in range(2):weight[dy*2+dx,8,dy,dx]=1
  wn='fixed_current_pack__weight';m.graph.initializer.append(numpy_helper.from_array(weight,wn));replacement=helper.make_node('Conv',[raw[0].name,wn],list(target.output),name='fixed_current_pack__Conv',kernel_shape=[2,2],strides=[2,2],pads=[0,0,0,0],dilations=[1,1],group=1)
  nodes=[replacement if n is target or n==target else n for n in m.graph.node];del m.graph.node[:];m.graph.node.extend(nodes)
 # Prune only unreachable tensors; downstream graph and all I/O contracts remain.
 prod=producers();live=set();nodes=set()
 def mark(value):
  if value in live:return
  live.add(value)
  if value in prod:
   n=prod[value];nodes.add(id(n))
   for v in n.input:
    if v:mark(v)
 for o in m.graph.output:mark(o.name)
 kept=[n for n in m.graph.node if id(n) in nodes];del m.graph.node[:];m.graph.node.extend(kept)
 kept=[v for v in m.graph.initializer if v.name in live];del m.graph.initializer[:];m.graph.initializer.extend(kept)
 infos=[v for v in m.graph.value_info if v.name in live];del m.graph.value_info[:];m.graph.value_info.extend(infos)
 onnx.checker.check_model(m);return m,report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--output-shuffle',choices=('six','two_three','three_two'),default='six');p.add_argument('--fixed-pack',action='store_true');a=p.parse_args()
 m,r=rewrite(onnx.load(str(a.input)),a.output_shuffle,a.fixed_pack);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output));r.update(input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),original_IO_preserved=True);a.output.with_suffix('.rewrite.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
