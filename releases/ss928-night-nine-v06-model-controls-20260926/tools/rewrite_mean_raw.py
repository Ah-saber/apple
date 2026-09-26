"""Derive a RAW-scale color head and fixed gray-phase compensation from native R2."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from materialize_aliases import materialize


def rewrite_mean(model,two_stage=False):
 m,aliases=materialize(model);initial={v.name:numpy_helper.to_array(v) for v in m.graph.initializer};prod={v:n for n in m.graph.node for v in n.output}
 outs=list(m.graph.output)
 if len(outs)!=1 or outs[0].type.tensor_type.elem_type!=TensorProto.FLOAT or len(outs[0].type.tensor_type.shape.dim)!=4 or outs[0].type.tensor_type.shape.dim[1].dim_value!=1:raise ValueError('Requires one FP32 full-gray graph output')
 shuffles=[n for n in m.graph.node if n.op_type=='DepthToSpace' and next((a.i for a in n.attribute if a.name=='blocksize'),0)==6]
 if len(shuffles)!=1:raise ValueError('Requires original undecomposed DepthToSpace6')
 shuffle=shuffles[0]
 if next((a.s for a in shuffle.attribute if a.name=='mode'),b'DCR')not in (b'CRD',b'DCR'):raise ValueError('Unsupported output phase order')
 def constant(v):
  if v in initial:return initial[v]
  n=prod.get(v)
  if n is not None and n.op_type=='Constant':
   t=next((a.t for a in n.attribute if a.name=='value'),None)
   if t is not None:return numpy_helper.to_array(t)
  return None
 def previous(n):
  if n.op_type in ('Cast','Identity'):return n.input[0]
  if n.op_type=='Clip' and all(constant(v) is not None and constant(v).size==1 for v in n.input[1:] if v):return n.input[0]
  if n.op_type=='Mul':
   scalar=[v for v in n.input if constant(v) is not None and constant(v).size==1]
   if len(scalar)==1:return next(v for v in n.input if v!=scalar[0])
  raise ValueError('Unsupported operation in final color path: '+n.op_type)
 # Ensure old output returns clipped gray scaled by exactly255.
 v=outs[0].name;scale_count=0;tail=[]
 while v in prod:
  n=prod[v]
  if n is shuffle or n==shuffle:break
  if n.op_type=='Mul':
   if not any(constant(x) is not None and constant(x).size==1 and float(constant(x).reshape(-1)[0])==255 for x in n.input):raise ValueError('Final scale is not255')
   scale_count+=1
  tail.append(n);v=previous(n)
 if v not in shuffle.output:raise ValueError('Shuffle6 is not the final full output path')
 v=shuffle.input[0];pre=[]
 while v in prod:
  n=prod[v]
  if n.op_type=='Conv':break
  if n.op_type=='Mul':
   if not any(constant(x) is not None and constant(x).size==1 and float(constant(x).reshape(-1)[0])==255 for x in n.input):raise ValueError('Pre-shuffle scale is not255')
   scale_count+=1
  pre.append(n);v=previous(n)
 conv=prod.get(v)
 if conv is None or conv.op_type!='Conv' or conv.input[1] not in initial or initial[conv.input[1]].shape!=(36,16,3,3) or scale_count!=1:raise ValueError('Requires original output Conv36 and exactly one gray scaling')
 if next((a.i for a in conv.attribute if a.name=='group'),1)!=1:raise ValueError('Grouped color head unsupported')
 clips=[n for n in pre+tail if n.op_type=='Clip']
 if len(clips)!=1:raise ValueError('Requires exactly one normalized output Clip')
 clip=clips[0]
 if len(clip.input)!=3 or any(not v or constant(v) is None for v in clip.input[1:]) or [float(constant(v).reshape(-1)[0]) for v in clip.input[1:]]!=[0.,1.]:raise ValueError('Requires normalized Clip bounds [0,1]')
 # The original scale must follow clipping in forward order.
 chain=[conv]+list(reversed(pre))+[shuffle]+list(reversed(tail));names={o.name for o in outs}
 if next(i for i,n in enumerate(chain) if n.op_type=='Mul')<chain.index(clip):raise ValueError('Gray scaling must follow normalized clipping')
 for n,nxt in zip(chain,chain[1:]):
  for val in n.output:
   if val in names or any(val in p.input and p is not nxt and p!=nxt for p in m.graph.node):raise ValueError('Old color path has extra consumers')
 # Learning parameters are derived by averaging each RAW pixel's nine phases.
 indices=[[(3*dy+p)*6+3*dx+q for p in range(3) for q in range(3)] for dy in range(2) for dx in range(2)]
 for j,key in enumerate(list(conv.input)[1:],1):
  if key not in initial:raise ValueError('Output parameter is not an initializer')
  old=initial[key];new=np.stack([old[idx].astype(np.float32).mean(0) for idx in indices]).astype(np.float16).astype(old.dtype);name=key+'__mean_raw'
  if name in initial:raise ValueError('Generated parameter name collision')
  m.graph.initializer.append(numpy_helper.from_array(new,name));conv.input[j]=name
 inferred=onnx.shape_inference.infer_shapes(model);info={v.name:v.type.tensor_type for v in list(inferred.graph.value_info)+list(inferred.graph.output)};tp=info.get(conv.output[0])
 if tp is None or len(tp.shape.dim)!=4:raise ValueError('Cannot determine final packed head shape')
 dims=[v.dim_value for v in tp.shape.dim]
 output_dims=[v.dim_value for v in outs[0].type.tensor_type.shape.dim]
 if dims[0] not in (0,1) or dims[1]!=36 or not dims[2] or not dims[3] or output_dims[0]!=1 or any(v.type.tensor_type.shape.dim[0].dim_value!=1 for v in m.graph.input):raise ValueError('Requires static batch1 I/O and packed36 spatial shape')
 b,h,w=1,dims[2],dims[3]
 if output_dims!=[b,1,h*6,w*6]:raise ValueError('Full output geometry does not match sixfold head')
 prefix='mean_raw__';extra=[]
 if any(v.startswith(prefix) for n in m.graph.node for v in list(n.input)+list(n.output)) or any(v.name.startswith(prefix) for v in m.graph.initializer):raise ValueError('Generated graph namespace collision')
 def initializer(name,array):
  name=prefix+name;m.graph.initializer.append(numpy_helper.from_array(np.asarray(array),name));return name
 zero=initializer('zero',np.array(0,dtype=np.float32));one_scalar=initializer('one',np.array(1,dtype=np.float32));scale=initializer('scale255',np.array(255,dtype=np.float32))
 half=prefix+'packed_half';fp32=prefix+'packed_float';norm=prefix+'normalized';color=prefix+'color'
 extra.extend([helper.make_node('Cast',[conv.output[0]],[half],to=TensorProto.FLOAT16),helper.make_node('Cast',[half],[fp32],to=TensorProto.FLOAT),helper.make_node('Clip',[fp32,zero,one_scalar],[norm]),helper.make_node('Mul',[norm,scale],[color])])
 offsets=np.arange(-4,5,dtype=np.float32).reshape(3,3)/9.
 if two_stage:
  raw=prefix+'raw';extra.append(helper.make_node('DepthToSpace',[color],[raw],blocksize=2,mode='CRD'));ones=np.ones((b,1,h*2,w*2),dtype=np.float32);weight=np.ones((2,1,3,3),dtype=np.float32);weight[1,0]=offsets;input_color=raw;stride=3
 else:
  ones=np.ones((b,1,h,w),dtype=np.float32);weight=np.zeros((5,1,6,6),dtype=np.float32)
  for dy in range(2):
   for dx in range(2):weight[dy*2+dx,0,dy*3:dy*3+3,dx*3:dx*3+3]=1;weight[4,0,dy*3:dy*3+3,dx*3:dx*3+3]=offsets
  input_color=color;stride=6
 const=initializer('ones',ones);kernel=initializer('repeat_weight',weight);joined=prefix+'joined';full=prefix+'full'
 extra.extend([helper.make_node('Concat',[input_color,const],[joined],axis=1),helper.make_node('ConvTranspose',[joined,kernel],[full],strides=[stride,stride],pads=[0,0,0,0],kernel_shape=[stride,stride],group=1),helper.make_node('Clip',[full,zero,scale],[outs[0].name])])
 # Insert after the learned head. Old entire color path becomes unreachable.
 nodes=[]
 for n in m.graph.node:
  nodes.append(n)
  if n is conv or n==conv:nodes.extend(extra)
 # Its old graph-output producer must be removed to avoid duplicate outputs.
 removed={id(n) for n in chain[1:]};nodes=[n for n in nodes if id(n) not in removed];del m.graph.node[:];m.graph.node.extend(nodes)
 prod={v:n for n in m.graph.node for v in n.output};live=set();live_nodes=set()
 def mark(value):
  if value in live:return
  live.add(value)
  if value in prod:
   node=prod[value];live_nodes.add(id(node))
   for v in node.input:
    if v:mark(v)
 for o in m.graph.output:mark(o.name)
 kept=[n for n in m.graph.node if id(n) in live_nodes];del m.graph.node[:];m.graph.node.extend(kept)
 kept=[v for v in m.graph.initializer if v.name in live];del m.graph.initializer[:];m.graph.initializer.extend(kept)
 # Packed output channel count changed: invalidate all internal annotations and infer anew.
 del m.graph.value_info[:];onnx.checker.check_model(m)
 report={'initializer_aliases':aliases,'output_conv_channels':4,'mean_phase_indices':indices,'output_full_shape':output_dims,'repeat_stage':'two' if two_stage else 'one','fixed_phase_offsets_gray':offsets.tolist(),'clipping_and_parameter_averaging_changes_model':True,'main_backbone_and_reference_branch_retained':True,'original_IO_preserved':True,'NPU_verified':False,'real_native_r2_verified':False}
 return m,report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--two-stage',action='store_true');a=p.parse_args();m,r=rewrite_mean(onnx.load(str(a.input)),a.two_stage);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output));r.update(input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest());a.output.with_suffix('.rewrite.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
