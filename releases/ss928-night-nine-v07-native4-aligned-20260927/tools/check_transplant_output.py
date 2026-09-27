"""Synthetic six-deconvolution native output controls; no real R2 claim."""
import copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
from transplant_output import replace_output
rng=np.random.default_rng(2709);base=Path(__file__).resolve().parents[1];old=base/'frozen_source';rows=[]
def init(a,n):return numpy_helper.from_array(a,n)
def model(nodes,inits,output_type=TensorProto.FLOAT):
 m=helper.make_model(helper.make_graph(nodes,'module',[helper.make_tensor_value_info('features',TensorProto.FLOAT16,[1,16,6,8])],[helper.make_tensor_value_info('display',output_type,[1,1,36,48])],inits),opset_imports=[helper.make_opsetid('',17)]);m.ir_version=10;onnx.checker.check_model(m);return m
for scene in ('ordinary','special'):
 frozen=onnx.load(old/f'{scene}_baseline_small.onnx');c={t.name:numpy_helper.to_array(t) for t in frozen.graph.initializer};conv=next(n for n in frozen.graph.node if n.op_type=='Conv' and n.input[1] in c and c[n.input[1]].shape==(36,16,3,3));w,b=c[conv.input[1]],c[conv.input[2]]
 inits=[init(w,'w36'),init(b,'b36'),init(np.array(0,np.float32),'lo'),init(np.array(1,np.float32),'hi'),init(np.array(255,np.float32),'gain')];nodes=[helper.make_node('Conv',['features','w36','b36'],['phases'],name='frozen36',pads=[1]*4)]
 for dy in range(6):
  k=np.zeros((36,1,6,6),np.float16)
  for dx in range(6):k[dy*6+dx,0,dy,dx]=1
  inits.append(init(k,f'k{dy}'));nodes.append(helper.make_node('ConvTranspose',['phases',f'k{dy}'],[f'd{dy}'],name=f'deconv{dy}',strides=[6,6]))
 value='d0'
 for dy in range(1,6):nodes.append(helper.make_node('Add',[value,f'd{dy}'],[f'a{dy}'],name=f'add{dy}'));value=f'a{dy}'
 nodes.extend([helper.make_node('Cast',[value],['float'],to=TensorProto.FLOAT),helper.make_node('Clip',['float','lo','hi'],['clip']),helper.make_node('Mul',['clip','gain'],['display'])]);native=model(nodes,inits)
 for channels,half_output in ((9,False),(16,False),(16,True)):
  ww=np.zeros((channels,16,3,3),np.float16);bb=np.zeros(channels,np.float16);ww[:9]=w[:9];bb[:9]=b[:9];si=[init(ww,'wo'),init(bb,'bo')]+[copy.deepcopy(t) for t in inits[2:5]];sn=[helper.make_node('Conv',['features','wo','bo'],['rawphases'],name='/output/conv/Conv',pads=[1]*4)];value='rawphases'
  if channels==16:
   si.extend([init(np.array([0],np.int64),'start'),init(np.array([9],np.int64),'end'),init(np.array([1],np.int64),'axis')]);sn.append(helper.make_node('Slice',[value,'start','end','axis'],['nine']));value='nine'

  if half_output:
   for index in range(2,5):si[index]=init(numpy_helper.to_array(si[index]).astype(np.float16),si[index].name)
  sn.extend([helper.make_node('Cast',[value],['float'],to=TensorProto.FLOAT16 if half_output else TensorProto.FLOAT),helper.make_node('Clip',['float','lo','hi'],['clip']),helper.make_node('Mul',['clip','gain'],['scaled']),helper.make_node('DepthToSpace',['scaled'],['native3'],blocksize=3,mode='CRD')]);si.append(init(np.array([1,1,2,2],np.float32),'scales'));sn.append(helper.make_node('Resize',['native3','','scales'],['display'],mode='nearest',coordinate_transformation_mode='asymmetric',nearest_mode='floor'));source=model(sn,si,TensorProto.FLOAT16 if half_output else TensorProto.FLOAT);graft,meta=replace_output(native,source,frozen)
  for kind in ('constant','signed_random','edge'):
   x=np.full((1,16,6,8),.3,np.float16) if kind=='constant' else rng.normal(0,.3,(1,16,6,8)).astype(np.float16)
   if kind=='edge':x.fill(0);x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
   a=ReferenceEvaluator(source).run(None,{'features':x})[0];result=ReferenceEvaluator(graft).run(None,{'features':x})[0];assert np.array_equal(a,result);rows.append({'scene':scene,'channels':channels,'Half_output':half_output,'input':kind,'bit_equal':True,'removed_deconv6':True})
  bad=copy.deepcopy(native);bad.graph.initializer[0].float_data.extend([]);bad.graph.initializer[0].CopyFrom(init(ww[:9] if channels==9 else np.zeros_like(w),'w36'))
  try:replace_output(bad,source,frozen)
  except ValueError:pass
  else:raise AssertionError('Unknown frozen weights accepted')
(base/'evidence/output_transplant_controls.json').write_text(json.dumps({'controls':rows,'native_graph':'synthetic six-deconvolution path using actual frozen output weights','actual_native_R2_unavailable':True,'SDK_verified':False},indent=2));print('PASSED',len(rows))
