"""Replace frozen output36 and its full downstream shuffle path on a native graph.
Keeps existing front, Body and native reference compatibility nodes. First apply
previous release's validated-source reference/statistics rewrites as needed.
Reject unknown learned projection, multiple outputs or shared downstream paths.
SDK compilation remains required; this tool does not reproduce board quantization.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper,helper,TensorProto
from materialize_aliases import materialize

def constants(m):
 out={v.name:numpy_helper.to_array(v) for v in m.graph.initializer}
 for n in m.graph.node:
  if n.op_type=='Constant':
   for a in n.attribute:
    if a.name=='value':out[n.output[0]]=numpy_helper.to_array(a.t)
 return out

def replace_output(native,source,frozen):
 native,_=materialize(native);source,_=materialize(source);frozen,_=materialize(frozen)
 if len(native.graph.output)!=1 or len(source.graph.output)!=1:raise ValueError('Require one complete output')
 shape=lambda v:[d.dim_value for d in v.type.tensor_type.shape.dim]
 if shape(native.graph.output[0])!=shape(source.graph.output[0]) or not all(shape(native.graph.output[0])):raise ValueError('Source and native complete output shapes differ or are unknown')
 if source.graph.output[0].type.tensor_type.elem_type not in (TensorProto.FLOAT,TensorProto.FLOAT16):raise ValueError('Require floating source display output')
 nc,sc,fc=constants(native),constants(source),constants(frozen)
 def frozen_conv(m,c):
  found=[n for n in m.graph.node if n.op_type=='Conv' and n.input[1] in c and c[n.input[1]].shape==(36,16,3,3)]
  if len(found)!=1:raise ValueError('Require unique frozen projection36')
  return found[0]
 old=frozen_conv(native,nc);reference=frozen_conv(frozen,fc)
 for index in (1,2):
  if len(old.input)<=index or len(reference.input)<=index or old.input[index] not in nc or reference.input[index] not in fc:raise ValueError('Require explicit frozen weight and bias')
  a,b=nc[old.input[index]],fc[reference.input[index]]
  if a.dtype not in (np.float16,np.float32) or a.shape!=b.shape or not np.array_equal(a.astype(np.float16),b.astype(np.float16)):raise ValueError('Native output weights differ from frozen teacher')
 learned=[n for n in source.graph.node if n.op_type=='Conv' and '/output/' in n.name and n.input[1] in sc and sc[n.input[1]].shape in [(4,16,3,3),(8,16,3,3),(9,16,3,3),(16,16,3,3)]]
 if len(learned)!=1:raise ValueError('Require unique supported source output entry; Tail5 unsupported')
 entry=learned[0];boundary=entry.input[0];source_end=source.graph.output[0].name;native_end=native.graph.output[0].name
 producers={v:n for n in source.graph.node for v in n.output};live=set()
 def walk(v):
  if v in live:return
  live.add(v)
  if v!=boundary and v in producers:
   for i in producers[v].input:
    if i:walk(i)
 walk(source_end)
 if any(v.name in live and v.name!=boundary for v in source.graph.input):raise ValueError('Output source depends on another input')
 source_nodes=[n for n in source.graph.node if any(o in live for o in n.output) and boundary not in n.output];source_init=[v for v in source.graph.initializer if v.name in live]
 prefix='continued_output_transplant__'
 if any(v.startswith(prefix) for n in native.graph.node for v in list(n.input)+list(n.output)):raise ValueError('Already transplanted')
 required_type=TensorProto.FLOAT16 if sc[entry.input[1]].dtype==np.float16 else TensorProto.FLOAT
 boundary_value=prefix+'tail_input';mapping={boundary:boundary_value,source_end:native_end}
 for n in source_nodes:
  for v in list(n.input)+list(n.output):
   if v and v not in mapping:mapping[v]=prefix+v
 for v in source_init:
  if v.name not in mapping:mapping[v.name]=prefix+v.name
 graft=[helper.make_node('Cast',[old.input[0]],[boundary_value],name=prefix+'tail_cast',to=required_type)]
 for n in source_nodes:
  q=copy.deepcopy(n);q.name=prefix+n.name
  for i,v in enumerate(q.input):
   if v:q.input[i]=mapping[v]
  for i,v in enumerate(q.output):q.output[i]=mapping[v]
  graft.append(q)
 inits=[]
 for v in source_init:q=copy.deepcopy(v);q.name=mapping[v.name];inits.append(q)
 # Identify every native node downstream of old projection, including six deconvs.
 dead={old.output[0]};removed={old.name};kept=[]
 for n in native.graph.node:
  if n==old or any(i in dead for i in n.input):dead.update(n.output);removed.add(n.name)
  else:kept.append(n)
 if native_end not in dead:raise ValueError('Native output does not follow frozen projection')
 pending=kept+graft;all_init=list(native.graph.initializer)+inits;available={v.name for v in native.graph.input}|{v.name for v in all_init};nodes=[]
 while pending:
  ready=[n for n in pending if all(not i or i in available for i in n.input)]
  if not ready:raise ValueError('Shared downstream path, missing input or dependency cycle')
  for n in ready:nodes.append(n);available.update(n.output);pending.remove(n)
 lookup={o:n for n in nodes for o in n.output};wanted=set()
 def visit(v):
  if v in wanted:return
  wanted.add(v)
  if v in lookup:
   for i in lookup[v].input:
    if i:visit(i)
 visit(native_end);nodes=[n for n in nodes if any(o in wanted for o in n.output)];all_init=[v for v in all_init if v.name in wanted]
 del native.graph.node[:];native.graph.node.extend(nodes);del native.graph.initializer[:];native.graph.initializer.extend(all_init);del native.graph.value_info[:]
 native.graph.output[0].type.CopyFrom(source.graph.output[0].type);onnx.checker.check_model(native)
 return native,{'removed_native_nodes':len(removed),'source_output_nodes':len(graft),'source_entry_weight_shape':list(sc[entry.input[1]].shape),'tail_boundary_cast_to_source_type':required_type,'complete_output_shape_unchanged':True,'output_type_from_source':source.graph.output[0].type.tensor_type.elem_type,'native_front_body_reference_nodes_preserved':True,'native_input_type_unchanged':True,'SDK_verified':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--frozen-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists() or a.output.with_suffix('.transplant.json').exists():raise FileExistsError(a.output)
 model,meta=replace_output(onnx.load(a.native),onnx.load(a.source),onnx.load(a.frozen_source));a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,a.output)
 for key,path in [('native',a.native),('source',a.source),('frozen_source',a.frozen_source),('output',a.output)]:meta[key+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 a.output.with_suffix('.transplant.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
