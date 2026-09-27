"""Full-shape nine-phase layout probes, with independent small mapping controls."""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator

def graph(mode,dtype,h=512,w=640):
 npdtype=np.float16 if dtype==TensorProto.FLOAT16 else np.float32;inits=[]
 if mode=='nearest':
  nodes=[helper.make_node('DepthToSpace',['phases'],['packed'],blocksize=3,mode='CRD'),helper.make_node('Resize',['packed','','scales'],['image'],mode='nearest',coordinate_transformation_mode='asymmetric',nearest_mode='floor')]
  inits=[numpy_helper.from_array(np.array([1,1,2,2],np.float32),'scales')]
 else:
  weight=np.zeros((9,1,6,6),dtype=npdtype)
  for y in range(3):
   for x in range(3):weight[y*3+x,0,y*2:y*2+2,x*2:x*2+2]=1
  inits=[numpy_helper.from_array(weight,'weight')];nodes=[helper.make_node('ConvTranspose',['phases','weight'],['image'],kernel_shape=[6,6],strides=[6,6])]
 g=helper.make_graph(nodes,'phase9_'+mode,[helper.make_tensor_value_info('phases',dtype,[1,9,h,w])],[helper.make_tensor_value_info('image',dtype,[1,1,h*6,w*6])],inits)
 m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)]);m.ir_version=8;onnx.checker.check_model(m);return m

def main():
 base=Path(__file__).parent.parent;rows=[];rng=np.random.default_rng(927)
 for name,dtype in [('fp16',TensorProto.FLOAT16),('fp32',TensorProto.FLOAT)]:
  for mode in ('nearest','fixed'):
   path=base/'onnx'/f'micro_phase9_{mode}_{name}.onnx'
   for kind in ('constant','random','edge'):
    x=np.full((1,9,3,5),.3,np.float16 if name=='fp16' else np.float32)
    if kind=='random':x=rng.standard_normal(x.shape).astype(x.dtype)
    if kind=='edge':x.fill(0);x[0,8,-1,-1]=1;x[0,0,0,0]=.8
    y=ReferenceEvaluator(graph(mode,dtype,3,5)).run(None,{'phases':x})[0]
    grid=x.reshape(1,1,3,3,3,5).transpose(0,1,4,2,5,3).reshape(1,1,9,15);expected=grid.repeat(2,2).repeat(2,3)
    assert np.array_equal(y,expected)
    rows.append({'mode':mode,'type':name,'input':kind,'bit_exact':True})
   onnx.save(graph(mode,dtype),path)
 (base/'evidence/phase9_micro_checks.json').write_text(json.dumps({'full_input':[1,9,512,640],'full_output':[1,1,3072,3840],'independent_engine':'ONNX ReferenceEvaluator','controls':rows,'SDK_verified':False},indent=2));print('PASSED',len(rows),'controls')
if __name__=='__main__':main()
