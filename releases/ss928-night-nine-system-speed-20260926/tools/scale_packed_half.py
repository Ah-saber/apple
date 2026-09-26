"""Move verified normalized clip/255 scaling before shuffle, producing FP16 output."""
import copy
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from materialize_aliases import materialize

def packed_half(model):
 model,_=materialize(model);producers={o:n for n in model.graph.node for o in n.output};constants={v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
 for n in model.graph.node:
  if n.op_type=='Constant':
   attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
   if 'value' in attrs:constants[n.output[0]]=numpy_helper.to_array(attrs['value'])
 shuffles=[n for n in model.graph.node if n.op_type=='DepthToSpace' and any(a.name=='blocksize' and a.i==6 for a in n.attribute)]
 if len(shuffles)!=1 or len(model.graph.output)!=1:raise ValueError('Require unique native shuffle6 and one full output')
 shuffle=shuffles[0];value=model.graph.output[0].name;chain=[]
 while value!=shuffle.output[0]:
  n=producers.get(value)
  if n is None or n.op_type not in {'Cast','Clip','Mul','Identity'}:raise ValueError('Unsupported output chain')
  operands=[v for v in n.input if v and v not in constants]
  if len(operands)!=1:raise ValueError('Require pointwise scalar-only output chain')
  chain.append(n);value=operands[0]
 ordered=list(reversed(chain));ops=[n.op_type for n in ordered if n.op_type not in {'Cast','Identity'}]
 if ops!=['Clip','Mul']:raise ValueError('Require normalized clip followed by scale')
 for n in ordered:
  if n.op_type=='Clip':
   if len(n.input)!=3 or n.input[1] not in constants or n.input[2] not in constants:raise ValueError('Require explicit clip 0,1')
   if constants[n.input[1]].size!=1 or constants[n.input[2]].size!=1 or float(constants[n.input[1]].reshape(-1)[0])!=0 or float(constants[n.input[2]].reshape(-1)[0])!=1:raise ValueError('Unexpected clipping limits')
  if n.op_type=='Mul':
   scalar=[constants[v] for v in n.input if v in constants]
   if len(scalar)!=1 or scalar[0].size!=1 or float(scalar[0].reshape(-1)[0])!=255:raise ValueError('Require scalar gray scale 255')
 # Packed normalized values must already be half precision for this reordering.
 inferred=onnx.shape_inference.infer_shapes(model);types={v.name:v.type.tensor_type.elem_type for v in list(inferred.graph.value_info)+list(inferred.graph.input)}
 if types.get(shuffle.input[0])!=TensorProto.FLOAT16:raise ValueError('Require verified FP16 normalized packed input')
 prefix='system_scale__'
 if any(o.startswith(prefix) for n in model.graph.node for o in n.output):raise ValueError('Namespace exists')
 for name,value in [('zero',0),('one',1),('255',255)]:model.graph.initializer.append(numpy_helper.from_array(np.asarray(value,dtype=np.float16),prefix+name))
 lo,scaled=prefix+'clipped',prefix+'scaled';extra=[helper.make_node('Clip',[shuffle.input[0],prefix+'zero',prefix+'one'],[lo]),helper.make_node('Mul',[lo,prefix+'255'],[scaled])];shuffle.input[0]=scaled;shuffle.output[0]=model.graph.output[0].name
 replaced=[]
 for n in model.graph.node:
  if n in chain:continue
  if n==shuffle:replaced.extend(extra)
  replaced.append(n)
 del model.graph.node[:];model.graph.node.extend(replaced);model.graph.output[0].type.tensor_type.elem_type=TensorProto.FLOAT16
 # Drop constants and metadata belonging to the old pointwise chain.
 live=set();lookup={o:n for n in model.graph.node for o in n.output}
 def visit(v):
  if v in live:return
  live.add(v)
  if v in lookup:
   for i in lookup[v].input:
    if i:visit(i)
 visit(model.graph.output[0].name);nodes=[n for n in model.graph.node if any(o in live for o in n.output)];initializers=[v for v in model.graph.initializer if v.name in live];del model.graph.node[:];model.graph.node.extend(nodes);del model.graph.initializer[:];model.graph.initializer.extend(initializers);del model.graph.value_info[:];onnx.checker.check_model(model);return model
