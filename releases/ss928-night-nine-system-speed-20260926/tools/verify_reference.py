import argparse,json
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from rewrite_reference import rewrite
p=argparse.ArgumentParser();p.add_argument('--graphs',type=Path,required=True);a=p.parse_args();rng=np.random.default_rng(961);reports=[]
for scene in ('ordinary','special'):
 old=onnx.load(str(a.graphs/f'{scene}_base_small.onnx'));new=onnx.load(str(a.graphs/f'{scene}_sigref_small.onnx'));patched,count=rewrite(old,old,new);onnx.save(patched,str(a.graphs/f'{scene}_sigref_small_patched.onnx'))
 for i in range(2):
  x=rng.uniform(.1,.9,(1,9,64,96)).astype(np.float32);ctx=rng.uniform(.1,.9,(1,1,64,64)).astype(np.float32);feed={'nine_raw':x,'reference_thumb':ctx};expected=ReferenceEvaluator(new).run(None,feed)[0];actual=ReferenceEvaluator(patched).run(None,feed)[0];delta=np.abs(actual-expected);assert np.array_equal(actual,expected);reports.append({'scene':scene,'case':i,'activations':count,'max_difference':float(delta.max())});print('PASS',reports[-1],flush=True)
(Path(__file__).parent/'reference_rewrite_verification.json').write_text(json.dumps({'controls':reports,'actual_native_r2_available':False,'NPU_verified':False},indent=2))
