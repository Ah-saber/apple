"""Compare graph rewrite to independently exported source, not to itself."""
import argparse,json,subprocess,sys,copy
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto
from onnx.reference import ReferenceEvaluator
from rewrite_output_rows import rewrite
from scale_packed_half import packed_half
p=argparse.ArgumentParser();p.add_argument('--graphs',type=Path,required=True);a=p.parse_args();rng=np.random.default_rng(392);reports=[]
for scene in ('ordinary','special'):
 old=a.graphs/f'{scene}_base_small.onnx'
 for front in ('balanced','contrast'):
  source=a.graphs/f'{scene}_{front}_small.onnx'
  for nhwc in (False,True):
   g=onnx.load(str(source));shape=[d.dim_value for d in g.graph.input[0].type.tensor_type.shape.dim];x=rng.uniform(.1,.9,shape).astype(np.float16).astype(np.float32);ctx=rng.uniform(.1,.9,(1,1,64,64)).astype(np.float32)
   if nhwc:
    raw=g.graph.input[0];name=raw.name;internal='control_nchw';node=helper.make_node('Transpose',[name],[internal],perm=[0,3,1,2])
    for n in g.graph.node:
     for i,v in enumerate(n.input):
      if v==name:n.input[i]=internal
    ns=[node]+list(g.graph.node);del g.graph.node[:];g.graph.node.extend(ns);raw.CopyFrom(helper.make_tensor_value_info(name,TensorProto.FLOAT16,[shape[0],shape[2],shape[3],shape[1]]));x=x.transpose(0,2,3,1).astype(np.float16);source2=a.graphs/f'{scene}_{front}_small_nhwc.onnx';onnx.save(g,str(source2))
   else:source2=source
   output=a.graphs/f'{scene}_{front}_small_splice_{nhwc}.onnx';subprocess.run([sys.executable,str(Path(__file__).parent/'splice_native_front.py'),'--native',str(old),'--student',str(source2),'--output',str(output)],check=True,stdout=subprocess.DEVNULL)
   expected=ReferenceEvaluator(g).run(None,{'nine_raw':x,'reference_thumb':ctx})[0];actual=ReferenceEvaluator(onnx.load(str(output))).run(None,{'nine_raw':x,'reference_thumb':ctx})[0];assert np.array_equal(actual,expected)
   changed=rewrite(packed_half(onnx.load(str(output))),16);scaled=ReferenceEvaluator(changed).run(None,{'nine_raw':x,'reference_thumb':ctx})[0];assert np.array_equal(scaled,expected.astype(np.float16));reports.append({'scene':scene,'front':front,'NHWC_FP16':nhwc,'splice_bitexact':True,'row_scale_bitexact_to_rounded_half':True});print('PASS',reports[-1],flush=True)
(Path(__file__).parent/'front_splice_verification.json').write_text(json.dumps({'controls':reports,'actual_native_r2_available':False,'NPU_verified':False},indent=2))
