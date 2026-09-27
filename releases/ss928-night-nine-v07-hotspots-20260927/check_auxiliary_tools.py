import json,sys
from pathlib import Path
import numpy as np
sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
sys.path.insert(0,str(Path(__file__).parent/'tools'))
from rewrite_interfaces import change
from prepare_micro_vectors import main as generate
root=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-FOLLOWUP-20260927');case=root/'auxiliary_check';case.mkdir(exist_ok=True)
weight=np.zeros((36,1,2,2),np.float16)
for t in range(9):
 for dy in range(2):
  for dx in range(2):weight[t*4+dy*2+dx,0,dy,dx]=1
nodes=[helper.make_node('Cast',['nine_raw'],['half'],to=TensorProto.FLOAT16),helper.make_node('Conv',['half','pack'],['packed'],kernel_shape=[2,2],strides=[2,2],group=9),helper.make_node('DepthToSpace',['packed'],['output'],blocksize=6,mode='CRD')]
graph=helper.make_graph(nodes,'full_size_probe_control',[helper.make_tensor_value_info('nine_raw',TensorProto.FLOAT,[1,9,1024,1280]),helper.make_tensor_value_info('unused_thumb',TensorProto.FLOAT,[1,1,64,64])],[helper.make_tensor_value_info('output',TensorProto.FLOAT16,[1,1,3072,3840])],[numpy_helper.from_array(weight,'pack')]);model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);model.ir_version=8;onnx.checker.check_model(model);onnx.save(model,case/'control.onnx')
vector='/data/zhangbenzhuang/huawei_sr/runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS/ordinary_frame_20.npz';v=np.load(vector);raw=v['nine_raw'];thumb=v['reference_thumb'];expected=ReferenceEvaluator(model).run(None,{'nine_raw':raw,'unused_thumb':thumb})[0]
changed=change(model,'input_nchw_half');actual=ReferenceEvaluator(changed).run(None,{'nine_raw':raw.astype(np.float16),'unused_thumb':thumb})[0];assert np.array_equal(actual,expected)
sys.argv=['prepare_micro_vectors','--model',str(case/'control.onnx'),'--vector',vector,'--output-dir',str(case/'generated'),'--engine','reference'];generate()
packed=np.load(case/'generated/canonical_vectors.npz')['packed'];want=raw.astype(np.float16).reshape(1,9,512,2,640,2).transpose(0,1,3,5,2,4).reshape(1,36,512,640);assert np.array_equal(packed,want)
result=np.load(case/'generated/expected_output_permutation.npz')['normalized_output'];assert np.array_equal(result,expected)
(root/'auxiliary_tools_checks.json').write_text(json.dumps({'full_size_nchw_input_half_control_bitexact':True,'micro_generator_full_size_reference_engine_verified':True,'unused_input_pruned':True,'ORT_engine_run':False,'real_vector':'ordinary_frame_20','input_shape':[1,9,1024,1280],'full_output_shape':[1,1,3072,3840],'NPU_measured':False},indent=2));print('COMPLETE auxiliary full-size checks')
