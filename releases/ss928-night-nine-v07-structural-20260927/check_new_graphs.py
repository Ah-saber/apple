"""Independent small controls plus complete-size micrographs for SDK mapping."""
import json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).parent/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages']
import onnx
from onnx import TensorProto,helper,numpy_helper
from onnx.reference import ReferenceEvaluator
from rewrite_hotspots import rewrite_statistics,rewrite_output
out=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-STRUCTURAL-20260927');(out/'onnx').mkdir(exist_ok=True)
def graph(kind,dtype,h,w):
 typ=TensorProto.FLOAT16 if dtype==np.float16 else TensorProto.FLOAT
 if kind=='statistics':
  kernel=np.zeros((12,9,2,2),dtype=dtype);temporal=np.zeros((3,9),dtype=dtype);temporal[0,8]=1;temporal[1,:5]=120/727;temporal[1,5]=127/727;temporal[2,6:8]=123/373;temporal[2,8]=127/373
  for g in range(3):
   for dy in range(2):
    for dx in range(2):kernel[g*4+dy*2+dx,:,dy,dx]=temporal[g]
  nodes=[helper.make_node('Conv',['input','weight'],['output'],kernel_shape=[2,2],strides=[2,2])];inputs=[1,9,h,w];outputs=[1,12,h//2,w//2];inits=[numpy_helper.from_array(kernel,'weight')]
 else:
  nodes=[helper.make_node('DepthToSpace',['input'],['output'],blocksize=6,mode='CRD')];inputs=[1,36,h,w];outputs=[1,1,h*6,w*6];inits=[]
 return helper.make_model(helper.make_graph(nodes,'mapping',[helper.make_tensor_value_info('input',typ,inputs)],[helper.make_tensor_value_info('output',typ,outputs)],inits),opset_imports=[helper.make_opsetid('',17)],ir_version=8)
rng=np.random.default_rng(2729);results=[];manifest=[]
for dtype in [np.float16,np.float32]:
 label='fp16' if dtype==np.float16 else 'fp32'
 for kind,modes in [('statistics',['space_pack','temporal_space_pack']),('output',['axis_horizontal','axis_vertical'])]:
  base=graph(kind,dtype,16,24)
  for mode in modes:
   rewrite=rewrite_statistics if kind=='statistics' else rewrite_output;candidate=rewrite(base,mode)
   for name in ['constant','random','edge_impulse']:
    shape=(1,9 if kind=='statistics' else 36,16,24)
    x=np.full(shape,.3,dtype=dtype) if name=='constant' else rng.random(shape).astype(dtype)
    if name=='edge_impulse':x=np.zeros(shape,dtype=dtype);x[:,:,0,0]=np.arange(shape[1])/max(1,shape[1]-1);x[:,:,-1,-1]=1
    y=ReferenceEvaluator(base).run(None,{'input':x})[0];z=ReferenceEvaluator(candidate).run(None,{'input':x})[0];delta=np.abs(y.astype(np.float64)-z.astype(np.float64));tol=.001 if dtype==np.float16 else 2e-7
    assert delta.max()<=tol,(kind,mode,name,float(delta.max()))
    if kind=='output':assert np.array_equal(y,z)
    results.append({'kind':kind,'mode':mode,'dtype':label,'input':name,'max_difference':float(delta.max()),'bit_exact':bool(np.array_equal(y,z))})
   h,w=(1024,1280) if kind=='statistics' else (512,640);full=rewrite(graph(kind,dtype,h,w),mode);onnx.checker.check_model(full);path=out/'onnx'/f'micro_{mode}_{label}.onnx';onnx.save(full,path);manifest.append({'file':str(path),'input_shape':[d.dim_value for d in full.graph.input[0].type.tensor_type.shape.dim],'output_shape':[d.dim_value for d in full.graph.output[0].type.tensor_type.shape.dim]})
(out/'new_graph_checks.json').write_text(json.dumps({'independent_engine':'ONNX ReferenceEvaluator','controls':results,'full_size_micrographs':manifest,'SDK_compiled':False,'NPU_measured':False},indent=2));print('PASSED',len(results),'controls',len(manifest),'micrographs',flush=True)
