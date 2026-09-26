"""Check CLI byte output against independent half-scaled complete image."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from rewrite_output_rows import rewrite
from scale_packed_half import packed_half
p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);a=p.parse_args();m=onnx.load(str(a.graph));output=a.graph.with_name('byte_control_verified.onnx');subprocess.run([sys.executable,str(Path(__file__).parent/'rewrite_output_rows.py'),'--input',str(a.graph),'--output',str(output),'--packed-half-scale','--output-byte'],check=True,stdout=subprocess.DEVNULL);rng=np.random.default_rng(104);feed={'nine_raw':rng.uniform(.1,.9,(1,9,64,96)).astype(np.float32),'reference_thumb':rng.uniform(.1,.9,(1,1,64,64)).astype(np.float32)};half=ReferenceEvaluator(rewrite(packed_half(m),16)).run(None,feed)[0];actual=ReferenceEvaluator(onnx.load(str(output))).run(None,feed)[0];expected=np.rint(half.astype(np.float32)).astype(np.uint8);assert np.array_equal(actual,expected);record={'source_graph':a.graph.name,'shape':list(actual.shape),'dtype':str(actual.dtype),'byte_matches_rounded_half':True,'NPU_verified':False};(Path(__file__).parent/'byte_output_verification.json').write_text(json.dumps(record,indent=2));print('PASS',record)
