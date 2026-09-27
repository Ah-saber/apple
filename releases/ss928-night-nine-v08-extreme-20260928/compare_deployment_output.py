"""Compare already compact board outputs to source, never guess SDK strides."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--actual',required=True);p.add_argument('--dtype',choices=['float16','float32'],required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--restore-block16',action='store_true');a=p.parse_args();ref=np.load(a.reference)['output'];dtype=np.dtype(a.dtype);raw=Path(a.actual).read_bytes();expected=ref.size*dtype.itemsize
if len(raw)!=expected:raise ValueError(f'Compact byte count mismatch: got {len(raw)}, expected {expected}; extract SDK valid elements first')
x=np.frombuffer(raw,dtype=dtype).reshape(ref.shape)
if not np.isfinite(x).all():raise ValueError('Non-finite output')
if a.restore_block16:
 if ref.ndim!=4 or ref.shape[1]!=16:raise ValueError('Expected blocked [1,16,H/16,W]')
 def restore(v):return v.transpose(0,2,1,3).reshape(v.shape[0],1,v.shape[2]*16,v.shape[3])
 x=restore(x);ref=restore(ref)
d=np.abs(x.astype(np.float32)-ref.astype(np.float32));rd=np.abs(np.rint(x.astype(np.float32).clip(0,255))-np.rint(ref.astype(np.float32).clip(0,255)));r={'shape':list(x.shape),'actual_dtype':a.dtype,'reference_dtype':str(ref.dtype),'source_reference_sha256':hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(),'actual_sha256':hashlib.sha256(raw).hexdigest(),'mean_gray':float(d.mean()),'p95_gray':float(np.percentile(d,95)),'max_gray':float(d.max()),'rounded_mean_gray':float(rd.mean()),'byte_identity':bool(np.array_equal(x,ref)),'quality_passed':None,'timing_target_met':None,'SDK_stride_handling':'actual must already be compact'};Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
