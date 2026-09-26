"""Independent ONNX interpreter verifies complete output phase and row ordering."""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto
from onnx.reference import ReferenceEvaluator
from rewrite_output_rows import rewrite
rng=np.random.default_rng(43);report=[]
for dtype,tt in [(np.float32,TensorProto.FLOAT),(np.float16,TensorProto.FLOAT16)]:
 for h,w in [(32,48),(48,64)]:
  x=rng.standard_normal((1,36,h,w)).astype(dtype)
  for mode in ('DCR','CRD'):
   original=helper.make_model(helper.make_graph([helper.make_node('DepthToSpace',['x'],['y'],blocksize=6,mode=mode)],'shuffle',[helper.make_tensor_value_info('x',tt,list(x.shape))],[helper.make_tensor_value_info('y',tt,[1,1,h*6,w*6])]),opset_imports=[helper.make_opsetid('',17)]);base=ReferenceEvaluator(original).run(None,{'x':x})[0]
   for groups in (2,4,8,16):
    changed=rewrite(original,groups);value=ReferenceEvaluator(changed).run(None,{'x':x})[0];assert np.array_equal(base,value);report.append({'dtype':str(dtype),'H':h,'W':w,'old_mode':mode,'groups':groups,'max_error':float(np.abs(base.astype(float)-value.astype(float)).max())})
p=Path(__file__).parent/'row_order_verification.json';p.write_text(json.dumps({'controls':report,'independent_interpreter':True,'NPU_verified':False},indent=2));print('PASS',len(report),'independent complete-image controls')
