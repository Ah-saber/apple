"""Preserve complete output while grouping spatial rows into shuffle channels.
No compiler or board claim: compile/profile the rewritten native graph separately.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper

def rewrite(model,groups=16):
 model=copy.deepcopy(model);nodes=[n for n in model.graph.node if n.op_type=='DepthToSpace' and any(a.name=='blocksize' and a.i==6 for a in n.attribute)]
 if len(nodes)!=1:raise ValueError('Require exactly one native DepthToSpace6; decomposed vendor graphs require separate review')
 node=nodes[0]
 if len(model.graph.output)!=1:raise ValueError('Require one complete image output')
 shape=[d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
 if len(shape)!=4 or shape[:2]!=[1,1] or min(shape[2:])<=0 or any(v%6 for v in shape[2:]):raise ValueError('Require static [1,1,6H,6W] output')
 h,w=shape[2]//6,shape[3]//6
 if groups<=0 or h%groups:raise ValueError('Row groups must divide packed image height')
 # Verify a pointwise-only path between shuffle and complete output; never infer
 # shuffle geometry from a downstream resize or a different output branch.
 producers={o:n for n in model.graph.node for o in n.output};const={v.name for v in model.graph.initializer};value=model.graph.output[0].name
 while value!=node.output[0]:
  n=producers.get(value)
  if n is None or n.op_type not in {'Cast','Clip','Mul','Identity','Round'}:raise ValueError('Unsupported post-shuffle geometry')
  dynamic=[v for v in n.input if v and v not in const and (v not in producers or producers[v].op_type!='Constant')]
  if len(dynamic)!=1:raise ValueError('Require one image operand after shuffle')
  value=dynamic[0]
 prefix='system_rows__';names={v.name for v in model.graph.initializer}|{v.name for v in model.graph.input}|{o for n in model.graph.node for o in n.output}
 if any(n.startswith(prefix) for n in names):raise ValueError('Namespace already used')
 def shp(name,values):
  key=prefix+name;model.graph.initializer.append(numpy_helper.from_array(np.array(values,dtype=np.int64),key));return key
 a,b,c,d=[prefix+v for v in ('split','transpose','packed','shuffle')]
 replacement=[helper.make_node('Reshape',[node.input[0],shp('shape_split',[36,groups,h//groups,w])],[a]),helper.make_node('Transpose',[a],[b],perm=[1,0,2,3]),helper.make_node('Reshape',[b,shp('shape_packed',[1,36*groups,h//groups,w])],[c]),helper.make_node('DepthToSpace',[c],[d],blocksize=6,mode='CRD'),helper.make_node('Reshape',[d,shp('shape_full',shape)],list(node.output))]
 for i,n in enumerate(replacement):n.name=prefix+str(i)
 ordered=[]
 for n in model.graph.node:ordered.extend(replacement if n==node else [n])
 del model.graph.node[:];model.graph.node.extend(ordered);onnx.checker.check_model(model);return model

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--groups',type=int,default=16);p.add_argument('--packed-half-scale',action='store_true');p.add_argument('--output-byte',action='store_true');a=p.parse_args();
 if a.output_byte and not a.packed_half_scale:raise ValueError('Byte output requires verified packed half scaling')
 m=onnx.load(str(a.input))
 if a.packed_half_scale:
  from scale_packed_half import packed_half
  m=packed_half(m)
 m=rewrite(m,a.groups)
 if a.output_byte:
  name=m.graph.output[0].name;half=name+'__system_half';floating=name+'__system_float';rounded=name+'__system_rounded'
  for n in m.graph.node:
   for j,v in enumerate(n.output):
    if v==name:n.output[j]=half
  m.graph.node.extend([helper.make_node('Round',[half],[rounded]),helper.make_node('Cast',[rounded],[name],to=onnx.TensorProto.UINT8)]);m.graph.output[0].type.tensor_type.elem_type=onnx.TensorProto.UINT8;onnx.checker.check_model(m)
 a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output));report={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'groups':a.groups,'complete_image_shape_preserved':True,'output_changed_to_FP16':a.packed_half_scale and not a.output_byte,'output_changed_to_uint8':a.output_byte,'averaged_phases':False,'checker_passed':True,'NPU_verified':False};a.output.with_suffix('.rows.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
