"""Copy trained reference parameters and replace exact GELU with SiLU.
Preserve the supplied native graph's resize compatibility and frozen main body.
"""
import argparse,copy,hashlib,json,math
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
from materialize_aliases import materialize

def context_convs(model):
 inputs=[v.name for v in model.graph.input if [d.dim_value for d in v.type.tensor_type.shape.dim]==[1,1,64,64]]
 if len(inputs)!=1:raise ValueError('Require unique context input')
 thumb=inputs[0];producers={o:n for n in model.graph.node for o in n.output};graphinputs={v.name for v in model.graph.input};cache={}
 def deps(v):
  if v in cache:return cache[v]
  if v in graphinputs:return {v}
  result=set()
  if v in producers:
   for i in producers[v].input:
    if i:result|=deps(i)
  cache[v]=result;return result
 nodes=[n for n in model.graph.node if n.op_type=='Conv' and deps(n.input[0])=={thumb}]
 return nodes,deps,thumb

def rewrite(native,old_source,new_source):
 native,_=materialize(native);old_source,_=materialize(old_source);new_source,_=materialize(new_source);oldconvs,_,_=context_convs(old_source);newconvs,nd,nt=context_convs(new_source);nativeconvs,deps,thumb=context_convs(native)
 if len(oldconvs)!=len(newconvs) or len(nativeconvs)!=len(oldconvs):raise ValueError('Reference convolution count differs')
 arrays=lambda m:{v.name:numpy_helper.to_array(v) for v in m.graph.initializer}
 ow,nw,vw=arrays(old_source),arrays(new_source),arrays(native);available=list(nativeconvs);prefix='system_ref__'
 if any(v.name.startswith(prefix) for v in native.graph.initializer):raise ValueError('Reference rewrite already applied')
 def attrs(n):
  given={a.name:helper.get_attribute_value(a) for a in n.attribute}
  return {k:given.get(k,d) for k,d in [('group',1),('strides',[1,1]),('pads',[0,0,0,0]),('dilations',[1,1]),('auto_pad',b'NOTSET')]}
 for index,(old,new) in enumerate(zip(oldconvs,newconvs)):
  if attrs(old)!=attrs(new) or len(old.input)!=len(new.input) or any(ow[o].shape!=nw[n].shape for o,n in zip(old.input[1:],new.input[1:])):raise ValueError('Source reference roles differ')
  matches=[n for n in available if len(n.input)==len(old.input) and attrs(n)==attrs(old) and all(np.array_equal(vw[vi],ow[oi].astype(vw[vi].dtype)) for vi,oi in zip(n.input[1:],old.input[1:]))]
  if len(matches)!=1:raise ValueError('Native reference does not match frozen source parameters exactly')
  target=matches[0];available.remove(target)
  for j,source in enumerate(new.input[1:],1):
   name=prefix+f'conv{index}_{j}';native.graph.initializer.append(numpy_helper.from_array(nw[source].astype(vw[target.input[j]].dtype),name));target.input[j]=name
 producers={o:n for n in native.graph.node for o in n.output};constants=arrays(native)
 for n in native.graph.node:
  if n.op_type=='Constant':
   attrs0={a.name:helper.get_attribute_value(a) for a in n.attribute}
   if 'value' in attrs0:constants[n.output[0]]=numpy_helper.to_array(attrs0['value'])
 def scalar(v):
  a=constants.get(v);return None if a is None or a.size!=1 else float(a.reshape(-1)[0])
 consumers={}
 for n in native.graph.node:
  for v in n.input:consumers.setdefault(v,[]).append(n)
 replacements={};count=0
 for erf in [n for n in native.graph.node if n.op_type=='Erf' and deps(n.input[0])=={thumb}]:
  div=producers.get(erf.input[0])
  if div is None or div.op_type!='Div' or scalar(div.input[1]) is None or abs(scalar(div.input[1])-math.sqrt(2))>1e-3:raise ValueError('Unsupported GELU denominator')
  x=div.input[0];adds=[n for n in consumers.get(erf.output[0],[]) if n.op_type=='Add' and any(scalar(v)==1 for v in n.input)]
  if len(adds)!=1:raise ValueError('Unsupported GELU add')
  addout=adds[0].output[0]
  def factors(v):
   if v in (x,addout):return [v],1.
   c=scalar(v)
   if c is not None:return [],c
   n=producers.get(v)
   if n is None or n.op_type!='Mul':raise ValueError('Not a GELU product')
   aa,sa=factors(n.input[0]);bb,sb=factors(n.input[1]);return aa+bb,sa*sb
  matches=[]
  for n in native.graph.node:
   if n.op_type!='Mul':continue
   try:fs,scale=factors(n.output[0])
   except ValueError:continue
   if sorted(fs)==sorted([x,addout]) and scale==.5:matches.append(n)
  if len(matches)!=1:raise ValueError('Unsupported GELU product ordering')
  final=matches[0];sig=prefix+f'sigmoid{count}';replacements[final.output[0]]=[helper.make_node('Sigmoid',[x],[sig]),helper.make_node('Mul',[x,sig],list(final.output))];count+=1
 expected=sum(n.op_type=='Sigmoid' and nd(n.input[0])=={nt} for n in new_source.graph.node)
 if count!=expected or count not in (2,5):raise ValueError('Activation count differs from trained source')
 nodes=[]
 for n in native.graph.node:nodes.extend(replacements.get(n.output[0],[n]))
 del native.graph.node[:];native.graph.node.extend(nodes);lookup={o:n for n in native.graph.node for o in n.output};live=set()
 def visit(v):
  if v in live:return
  live.add(v)
  if v in lookup:
   for i in lookup[v].input:
    if i:visit(i)
 for o in native.graph.output:visit(o.name)
 nodes=[n for n in native.graph.node if any(o in live for o in n.output)];inits=[v for v in native.graph.initializer if v.name in live];del native.graph.node[:];native.graph.node.extend(nodes);del native.graph.initializer[:];native.graph.initializer.extend(inits);del native.graph.value_info[:];onnx.checker.check_model(native);return native,count

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--old-source',type=Path,required=True);p.add_argument('--new-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m,count=rewrite(onnx.load(str(a.native)),onnx.load(str(a.old_source)),onnx.load(str(a.new_source)));a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(a.output));report={'replaced_activations':count,'native_reference_frozen_weights_checked':True,'native_resize_compatibility_preserved':True,'NPU_verified':False,'source_sha256':hashlib.sha256(a.new_source.read_bytes()).hexdigest()};a.output.with_suffix('.reference.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
