"""Bounded isolated statistics controls with real frozen coefficient guards."""
import copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
from rewrite_statistics3 import rewrite
base=Path(__file__).resolve().parents[1];rng=np.random.default_rng(2709);rows=[]
for scene in ('ordinary','special'):
 old=onnx.load(base/f'frozen_source/{scene}_baseline_small.onnx');constants={t.name:numpy_helper.to_array(t) for t in old.graph.initializer};conv=next(n for n in old.graph.node if n.op_type=='Conv' and n.input[1] in constants and constants[n.input[1]].shape==(12,9,2,2));w=constants[conv.input[1]]
 for precision in ('float64','float16'):
  dtype=np.float64 if precision=='float64' else np.float16;typ=TensorProto.DOUBLE if precision=='float64' else TensorProto.FLOAT16;kernel=w.astype(dtype);node=helper.make_node('Conv',['nine','weight'],['stats'],kernel_shape=[2,2],strides=[2,2]);g=helper.make_graph([node],'module',[helper.make_tensor_value_info('nine',typ,[1,9,12,16])],[helper.make_tensor_value_info('stats',typ,[1,12,6,8])],[numpy_helper.from_array(kernel,'weight')]);m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)]);m.ir_version=10
  for rectified in (False,True):
   new=rewrite(m,rectified);oldexe,newexe=ReferenceEvaluator(m),ReferenceEvaluator(new)
   for kind in ('constant','signed_random','edge'):
    x=np.full((1,9,12,16),.3,dtype) if kind=='constant' else rng.normal(0,.3,(1,9,12,16)).astype(dtype)
    if kind=='edge':x.fill(0);x[:,:,0,0]=1;x[:,:,-1,-1]=-.5
    a=oldexe.run(None,{'nine':x})[0];b=newexe.run(None,{'nine':x})[0];err=float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max())
    if precision=='float64':assert err<1e-12
    else:assert err<.005
    rows.append({'scene':scene,'precision':precision,'rectified32':rectified,'input':kind,'max_normalized_error':err})
(base/'evidence/statistics3_rewrite_controls.json').write_text(json.dumps({'controls':rows,'scope':'real frozen statistics module; ONNX CPU evaluator','SDK_verified':False},indent=2));print('PASSED',len(rows))
