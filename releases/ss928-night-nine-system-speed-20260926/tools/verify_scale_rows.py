import json
from pathlib import Path
import numpy as np
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
from scale_packed_half import packed_half
from rewrite_output_rows import rewrite
rng=np.random.default_rng(239);x=rng.uniform(-.2,1.2,(1,36,32,48)).astype(np.float16);inits=[numpy_helper.from_array(np.asarray(v,np.float32),k) for k,v in [('zero',0),('one',1),('scale',255)]];nodes=[helper.make_node('DepthToSpace',['x'],['shuffled'],blocksize=6,mode='CRD'),helper.make_node('Cast',['shuffled'],['f32'],to=TensorProto.FLOAT),helper.make_node('Clip',['f32','zero','one'],['clip']),helper.make_node('Mul',['clip','scale'],['out'])];model=helper.make_model(helper.make_graph(nodes,'scale',[helper.make_tensor_value_info('x',TensorProto.FLOAT16,list(x.shape))],[helper.make_tensor_value_info('out',TensorProto.FLOAT,[1,1,192,288])],inits),opset_imports=[helper.make_opsetid('',17)]);baseline=ReferenceEvaluator(model).run(None,{'x':x})[0];half=packed_half(model);report=[]
for g in (2,4,8,16):
 m=rewrite(half,g);y=ReferenceEvaluator(m).run(None,{'x':x})[0];delta=np.abs(y.astype(float)-baseline.astype(float));assert np.array_equal(y,baseline.astype(np.float16));report.append({'groups':g,'bitexact_to_original_rounded_FP16':True,'max_difference_from_FP32_gray':float(delta.max()),'mean_difference_from_FP32_gray':float(delta.mean())})
Path(__file__).with_name('half_scale_row_verification.json').write_text(json.dumps({'controls':report,'NPU_verified':False},indent=2));print('PASS',report)
