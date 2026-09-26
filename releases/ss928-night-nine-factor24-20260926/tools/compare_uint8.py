"""Compare effective NCHW board output with the matching precision/source reference."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--scene',required=True,choices=('ordinary','special'))
p.add_argument('--actual',required=True,type=Path)
p.add_argument('--input-precision',choices=('float16','float32'),default='float16')
p.add_argument('--reference-runtime',choices=('source','compiled'),default='source')
p.add_argument('--output',type=Path)
a=p.parse_args();package=Path(__file__).resolve().parents[1]
with np.load(package/'reference_outputs'/(a.scene+'_reference_outputs.npz'),allow_pickle=False) as v:expected=v[a.input_precision+'_'+a.reference_runtime]
actual=np.fromfile(a.actual,dtype=np.uint8)
if actual.size!=expected.size:raise ValueError(f'Expected {expected.size} effective uint8 pixels, got {actual.size}; remove device padding first')
diff=np.abs(actual.astype(np.int16)-expected.reshape(-1).astype(np.int16));index=int(diff.argmax());y,x=divmod(index,3840)
report={'scene':a.scene,'input_precision':a.input_precision,'reference_runtime':a.reference_runtime,'pixels':int(diff.size),'mean_abs_gray':float(diff.mean()),'p99_abs_gray':float(np.percentile(diff,99)),'max_abs_gray':int(diff.max()),'fraction_over_1_gray':float((diff>1).mean()),'max_difference_output_yx':[y,x],'max_difference_raw_yx':[y//3,x//3]}
text=json.dumps(report,indent=2)
if a.output:a.output.write_text(text)
print(text)
