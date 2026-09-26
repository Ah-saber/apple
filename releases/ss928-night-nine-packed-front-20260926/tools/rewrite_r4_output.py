"""Change only final sixfold output shuffle precision/order in a supplied R4 ONNX."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dtype',choices=('float32','float16'),default='float32');p.add_argument('--dcr',action='store_true');a=p.parse_args()
m=onnx.load(str(a.input));onnx.checker.check_model(m)
producers={o:n for n in m.graph.node for o in n.output};inits={v.name:numpy_helper.to_array(v) for v in m.graph.initializer}
def const(value):
 if value in inits:return inits[value]
 n=producers.get(value)
 if n is not None and n.op_type=='Constant':
  t=next((x.t for x in n.attribute if x.name=='value'),None)
  if t is not None:return numpy_helper.to_array(t)
 return None
shuffles=[n for n in m.graph.node if n.op_type=='DepthToSpace' and next((v.i for v in n.attribute if v.name=='blocksize'),0)==6]
if len(shuffles)!=1:raise ValueError('Expected unique final DepthToSpace6')
s=shuffles[0];mul=producers.get(s.input[0])
if mul is None or mul.op_type!='Mul':raise ValueError('Expected gray scale Mul immediately before DepthToSpace6')
scale_inputs=[v for v in mul.input if const(v) is not None and np.all(const(v)==255)]
if len(scale_inputs)!=1:raise ValueError('Expected scalar255 gray scale')
base_input=next(v for v in mul.input if v!=scale_inputs[0]);clip=producers.get(base_input)
if clip is None or clip.op_type!='Clip' or len(clip.input)<3 or not np.all(const(clip.input[1])==0) or not np.all(const(clip.input[2])==1):raise ValueError('Expected Clip[0,1] before gray scale')
# Ensure single-channel output before using DCR==CRD equivalence.
if a.dcr:
 outputs=[v for v in m.graph.output if v.name==s.output[0] and len(v.type.tensor_type.shape.dim)==4 and v.type.tensor_type.shape.dim[1].dim_value==1]
 if len(outputs)!=1 or any(s.output[0] in n.input for n in m.graph.node):
  raise ValueError('DCR proof requires shuffle itself directly return a single-channel output')
 attr=next((v for v in s.attribute if v.name=='mode'),None)
 if attr is not None:attr.s=b'DCR'
 else:s.attribute.append(helper.make_attribute('mode','DCR'))
old_out=s.output[0];half_input='output_half_shuffle__packed';half_result='output_half_shuffle__result';s.input[0]=half_input;s.output[0]=half_result
cast=helper.make_node('Cast',[base_input],[half_input],name='output_half_shuffle__cast',to=TensorProto.FLOAT16)
post=[]
if a.dtype=='float32':
 cast_out='output_half_shuffle__float';post.append(helper.make_node('Cast',[half_result],[cast_out],to=TensorProto.FLOAT,name='output_half_shuffle__to_float'))
 post.append(helper.make_node('Mul',[cast_out,scale_inputs[0]],[old_out],name='output_half_shuffle__scale'))
else:
 scale_name='output_half_shuffle__scale255';m.graph.initializer.append(numpy_helper.from_array(np.array(255,dtype=np.float16),scale_name));post.append(helper.make_node('Mul',[half_result,scale_name],[old_out],name='output_half_shuffle__scale'))
 # For float16 I/O, reject post-shuffle consumers: retain only directly returned output.
 if any(old_out in n.input for n in m.graph.node):raise ValueError('FP16 output requires shuffle output be returned directly')
 for v in m.graph.output:
  if v.name==old_out:v.type.tensor_type.elem_type=TensorProto.FLOAT16
ordered=[]
for n in m.graph.node:
 if n is s or n==s:ordered.append(cast);ordered.append(n);ordered.extend(post)
 else:ordered.append(n)
del m.graph.node[:];m.graph.node.extend(ordered)
# Remove only unreachable pre-shuffle scale; preserve preceding clipping/core.
producers={o:n for n in m.graph.node for o in n.output};live=set();nodes=set()
def mark(value):
 if value in live:return
 live.add(value)
 if value in producers:
  n=producers[value];nodes.add(id(n))
  for v in n.input:
   if v:mark(v)
for v in m.graph.output:mark(v.name)
kept=[n for n in m.graph.node if id(n) in nodes];del m.graph.node[:];m.graph.node.extend(kept)
kept=[v for v in m.graph.initializer if v.name in live];del m.graph.initializer[:];m.graph.initializer.extend(kept)
# Removed intermediates carry stale dtype/shape annotations.
kept=[v for v in m.graph.value_info if v.name in live and v.name not in (old_out,half_input,half_result)];del m.graph.value_info[:];m.graph.value_info.extend(kept)
onnx.checker.check_model(m);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output))
r={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'shuffle_dtype':'float16','output_dtype':a.dtype,'DCR':a.dcr,'full_output_retained':True,'onnx_checker_passed':True,'real_R4_or_NPU_verified':False,'source_fp16_packed_values':'float32 output exact; float16 output is rounding full gray','float32_R4_packed_values':'additional FP16 rounding; must check actual numeric conversion'}
a.output.with_suffix('.rewrite.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
