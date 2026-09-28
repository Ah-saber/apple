"""CPU ONNX Runtime check of exported graph and numerical drift from C11."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
root=Path(__file__).resolve().parent
cases=['v08_c11','rows16static_refafter','rows32_refafter','rows16static_meanfirst','raw6_rows16static','raw6_rows32']
rng=np.random.default_rng(20260928)
rows=[]
for scene in ('ordinary','special'):
 x=rng.uniform(0.1,0.9,(1,9,64,96)).astype(np.float16)
 c=rng.uniform(0.1,0.9,(1,1,64,64)).astype(np.float32)
 ref=None
 for case in cases:
  path=root/'source_onnx'/f'{scene}_{case}_small.onnx'
  sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
  y=sess.run(None,{'nine_raw':x,'reference_thumb':c})[0]
  assert y.shape==(1,1,192,288) and np.isfinite(y).all()
  if ref is None:ref=y
  d=np.abs(y.astype(np.float32)-ref.astype(np.float32))
  item={'scene':scene,'case':case,'output_shape':list(y.shape),'mean_diff_gray':float(d.mean()),'max_diff_gray':float(d.max()),'NPU_verified':False}
  rows.append(item)
  print(item)
(root/'small_ort_extra.json').write_text(json.dumps({'results':rows},indent=2))
