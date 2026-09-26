"""Replace only denoised packed input ancestors in a supplied R4 ONNX; retain its core/output."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
p=argparse.ArgumentParser();p.add_argument('--r4',type=Path,required=True);p.add_argument('--student',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
old=onnx.load(str(a.r4));new=onnx.load(str(a.student));onnx.checker.check_model(old);onnx.checker.check_model(new)
def weights(model):return {v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
def find_conv(model,shape):
 w=weights(model);matches=[n for n in model.graph.node if n.op_type=='Conv' and len(n.input)>1 and n.input[1] in w and tuple(w[n.input[1]].shape)==shape]
 if len(matches)!=1:raise ValueError(f'Expected unique Conv with weight {shape}, got {len(matches)}')
 return matches[0]
old_gate=find_conv(old,(16,4,3,3));new_gate=find_conv(new,(16,4,3,3))
# Exact frozen learned-convolution agreement is required, including cast to the
# vendor graph dtype. Stop if R4 weights/layout have changed this contract.
for shape in ((16,4,3,3),(36,16,3,3)):
 oc,nc=find_conv(old,shape),find_conv(new,shape);ow,nw=weights(old),weights(new)
 if len(oc.input)!=len(nc.input):raise ValueError(f'Bias contract differs for {shape}')
 def attrs(node):
  given={v.name:helper.get_attribute_value(v) for v in node.attribute}
  return {k:given.get(k,default) for k,default in [('group',1),('strides',[1,1]),('pads',[0,0,0,0]),('dilations',[1,1]),('auto_pad',b'NOTSET')]}
 if attrs(oc)!=attrs(nc):raise ValueError(f'Convolution padding/stride/layout contract differs for {shape}')
 for oi,ni in zip(oc.input[1:],nc.input[1:]):
  if not np.array_equal(ow[oi],nw[ni].astype(ow[oi].dtype)):raise ValueError(f'Frozen weight/bias mismatch for {shape}; manual compatibility review required')
def raw_input(model):
 matches=[v for v in model.graph.input if len(v.type.tensor_type.shape.dim)==4 and v.type.tensor_type.shape.dim[1].dim_value==9]
 if len(matches)!=1:raise ValueError('Expected unique nine-channel RAW input')
 return matches[0]
old_raw,new_raw=raw_input(old),raw_input(new)
if old_raw.type.tensor_type.elem_type!=new_raw.type.tensor_type.elem_type:raise ValueError('RAW input dtypes differ; explicit adapter required')
for v,u in zip(old_raw.type.tensor_type.shape.dim,new_raw.type.tensor_type.shape.dim):
 if v.dim_value!=u.dim_value:raise ValueError('RAW dimensions differ')
producers={o:n for n in new.graph.node for o in n.output};needed=set();nodes=set()
def visit(value):
 if value in needed:return
 needed.add(value)
 if value in producers:
  n=producers[value];nodes.add(id(n))
  for inp in n.input:
   if inp:visit(inp)
visit(new_gate.input[0]);prefix='packed_front__'
rename=lambda name:old_raw.name if name==new_raw.name else prefix+name
front=[]
for n in new.graph.node:
 if id(n) not in nodes:continue
 copied=onnx.NodeProto();copied.CopyFrom(n);copied.name=prefix+(n.name or n.op_type)
 for j,v in enumerate(copied.input):
  if v:copied.input[j]=rename(v)
 for j,v in enumerate(copied.output):copied.output[j]=rename(v)
 front.append(copied)
for init in new.graph.initializer:
 if init.name in needed:
  copied=onnx.TensorProto();copied.CopyFrom(init);copied.name=rename(init.name);old.graph.initializer.append(copied)
# Source front supplies FP16 to the gate convolution. Match an FP32 R4 gate
# by casting at this one boundary; student computation itself remains unchanged.
weight_dtype=next(v.data_type for v in old.graph.initializer if v.name==old_gate.input[1])
final=rename(new_gate.input[0])
if weight_dtype!=TensorProto.FLOAT16:
 if weight_dtype!=TensorProto.FLOAT:raise ValueError('Unsupported R4 gate dtype')
 cast_output=prefix+'gate_input_vendor_dtype';front.append(helper.make_node('Cast',[final],[cast_output],name=prefix+'gate_dtype',to=weight_dtype));final=cast_output
old_gate.input[0]=final
ordered=[]
for n in old.graph.node:
 if n is old_gate or n==old_gate:ordered.extend(front)
 ordered.append(n)
del old.graph.node[:];old.graph.node.extend(ordered)
# Drop only unreachable old trajectory nodes/initializers. Keep graph I/O names,
# sizes, dtypes and every reachable downstream R4 node and learned initializer.
producers={o:n for n in old.graph.node for o in n.output};live=set();live_nodes=set()
def mark(value):
 if value in live:return
 live.add(value)
 if value in producers:
  n=producers[value];live_nodes.add(id(n))
  for inp in n.input:
   if inp:mark(inp)
for output in old.graph.output:mark(output.name)
kept=[n for n in old.graph.node if id(n) in live_nodes];del old.graph.node[:];old.graph.node.extend(kept)
inits=[v for v in old.graph.initializer if v.name in live];del old.graph.initializer[:];old.graph.initializer.extend(inits)
# Old internal annotations can refer to removed tensors or changed precision.
infos=[v for v in old.graph.value_info if v.name in live and v.name!=old_gate.input[0]];del old.graph.value_info[:];old.graph.value_info.extend(infos)
onnx.checker.check_model(old);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(old,str(a.output))
report={'r4':str(a.r4),'r4_sha256':hashlib.sha256(a.r4.read_bytes()).hexdigest(),'student_sha256':hashlib.sha256(a.student.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'inserted_front_nodes':len(front),'frozen_head_and_output_weights_checked':True,'original_graph_IO_preserved':True,'onnx_checker_passed':True,'real_r4_board_validation_required':True}
a.output.with_suffix('.splice.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
