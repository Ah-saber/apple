"""Independent rewritten-vs-source controls, including an extra native-like Conv."""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);a=p.parse_args()
base=Path(__file__).parent;sys.path[:0]=[str(base/'tools'),'/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages']
import onnx
from onnx import helper,numpy_helper
from onnx.reference import ReferenceEvaluator
from rewrite_reference_relu import rewrite,context_convs
out=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-STRUCTURAL-20260927');folder=out/'source_onnx';rng=np.random.default_rng(2769);records=[]
cases=['space_pack','temporal_space_pack','axis_horizontal','axis_vertical','phase3_preserve_nearest','phase3_preserve_fixed','reference_relu_output_stable','space_pack_phase3_preserve_nearest','space_pack_reference_relu_output_stable','space_pack_reference_relu_output_stable_phase3_preserve_nearest']
for kind in ['constant','random','edge_impulse']:
 x=np.full((1,9,64,96),.3,np.float32) if kind=='constant' else rng.random((1,9,64,96)).astype(np.float32);c=np.full((1,1,64,64),.3,np.float32) if kind=='constant' else rng.random((1,1,64,64)).astype(np.float32)
 if kind=='edge_impulse':x.fill(0);x[:,:,0,0]=1;x[:,:,-1,-1]=.8;c.fill(.3)
 inputs={'nine_raw':x,'reference_thumb':c}
 for case in cases:
  source=ReferenceEvaluator(str(folder/f'{a.scene}_{case}_small.onnx')).run(None,inputs)[0]
  changed=ReferenceEvaluator(str(folder/f'{a.scene}_{case}_rewritten_small.onnx')).run(None,inputs)[0]
  d=np.abs(source.astype(np.float64)-changed.astype(np.float64));assert d.max()<=.26,(case,kind,d.max())
  records.append({'case':case,'input':kind,'mean_gray':float(d.mean()),'max_gray':float(d.max()),'bit_exact':bool(np.array_equal(source,changed))})
# A native R2 graph may contain extra fixed interpolation Convs. Insert an
# identity reference Conv; the learned-role matcher must leave it intact.
old=onnx.load(folder/f'{a.scene}_baseline_small.onnx');new=onnx.load(folder/f'{a.scene}_reference_relu_output_stable_small.onnx');native=onnx.load(folder/f'{a.scene}_baseline_small.onnx')
weights={v.name:numpy_helper.to_array(v) for v in native.graph.initializer}
project=next(n for n in context_convs(native)[0] if n.input[1] in weights and weights[n.input[1]].shape==(16,12,1,1))
value=project.output[0];name='native_like_reference_identity';weight=np.ones((16,1,1,1),np.float16)
for node in native.graph.node:
 for index,item in enumerate(node.input):
  if item==value:node.input[index]=name
nodes=list(native.graph.node);index=nodes.index(project);nodes.insert(index+1,helper.make_node('Conv',[value,name+'_weight'],[name],kernel_shape=[1,1],group=16));del native.graph.node[:];native.graph.node.extend(nodes);native.graph.initializer.append(numpy_helper.from_array(weight,name+'_weight'))
converted,count=rewrite(native,old,new);assert any(n.output[0]==name for n in converted.graph.node)
inputs={'nine_raw':np.full((1,9,64,96),.3,np.float32),'reference_thumb':np.full((1,1,64,64),.3,np.float32)}
y=ReferenceEvaluator(converted).run(None,inputs)[0];z=ReferenceEvaluator(new).run(None,inputs)[0];assert np.max(np.abs(y-z))<=.26
(out/f'{a.scene}_export_checks.json').write_text(json.dumps({'independent_engine':'ONNX ReferenceEvaluator','controls':records,'extra_fixed_reference_conv_preserved':True,'activation_count':count,'SDK_verified':False},indent=2));print('PASSED',a.scene,len(records),'controls and fixed-Conv guard',flush=True)
