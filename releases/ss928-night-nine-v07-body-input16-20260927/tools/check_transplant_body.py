"""Bounded actual-coefficient body-only rewrite check; no whole NPU claim."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
from onnx.reference import ReferenceEvaluator
from transplant_body import transplant,source_chain,arrays
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);a=p.parse_args()
reference=onnx.load(a.reference);candidate=onnx.load(a.candidate)
def isolated(model,dtype,h,w):
    chain=copy.deepcopy(source_chain(model,(1,2,3,4)));weights=arrays(model);first=chain[0].input[0];last=chain[-1].output[0];required={name for n in chain for name in n.input if name in weights};initializers=[numpy_helper.from_array(weights[name].astype(dtype),name) for name in sorted(required)];elem=onnx.TensorProto.FLOAT16 if dtype==np.float16 else onnx.TensorProto.FLOAT
    graph=helper.make_graph(chain,'isolated_body',[helper.make_tensor_value_info(first,elem,[1,16,h,w])],[helper.make_tensor_value_info(last,elem,[1,16,h,w])],initializers)
    result=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);result.ir_version=10;onnx.checker.check_model(result);return result
rng=np.random.default_rng(2784);rows=[]
for dtype in (np.float16,np.float32):
    for h,w in ((3,4),(8,11),(12,14)):
        native=isolated(reference,dtype,h,w);expected=isolated(candidate,dtype,h,w);rewritten,report=transplant(native,reference,candidate);x=rng.normal(.1,.5,(1,16,h,w)).astype(dtype)
        y=ReferenceEvaluator(rewritten).run(None,{native.graph.input[0].name:x})[0];z=ReferenceEvaluator(expected).run(None,{expected.graph.input[0].name:x})[0];error=float(np.max(np.abs(y.astype(np.float64)-z.astype(np.float64))));assert error==0
        rows.append({'dtype':str(dtype),'shape':[h,w],'all_pixel_max_abs':error})
native=isolated(reference,np.float16,8,11);broken=copy.deepcopy(native);first=source_chain(broken,(4,))[0];index=next(i for i,v in enumerate(broken.graph.initializer) if v.name==first.input[1]);values=numpy_helper.to_array(broken.graph.initializer[index]).copy();values.flat[0]+=1;broken.graph.initializer[index].CopyFrom(numpy_helper.from_array(values,first.input[1]));rejected=False
try:transplant(broken,reference,candidate)
except ValueError:rejected=True
assert rejected
print(json.dumps({'reference':Path(a.reference).name,'candidate':Path(a.candidate).name,'body_only':True,'SDK_verified':False,'changed_frozen_weight_rejected':rejected,'controls':rows},indent=2))
