"""Compare effective full-gray output against a matching source reference."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dtype',choices=('float32','float16'),default='float32');p.add_argument('--row-stride-bytes',type=int);a=p.parse_args()
with np.load(a.reference,allow_pickle=False) as v:ref=v['display_gray'].astype(np.float32).reshape(3072,3840)
dtype=np.dtype('<f4' if a.dtype=='float32' else '<f2');row_bytes=3840*dtype.itemsize;stride=a.row_stride_bytes or row_bytes
if stride<row_bytes or stride%dtype.itemsize:raise ValueError('Invalid row stride')
actual=np.fromfile(a.output,dtype=dtype)
if actual.size!=3072*stride//dtype.itemsize:raise ValueError('Output byte count does not match full frame and stride')
actual=actual.reshape(3072,stride//dtype.itemsize)[:,:3840].astype(np.float32)
if not np.isfinite(actual).all():raise ValueError('Non-finite model output')
delta=actual-ref;err=np.abs(delta);mse=float(np.mean(delta.astype(np.float64)**2))
byte_delta=np.abs(np.rint(np.clip(actual,0,255)).astype(np.int16)-np.rint(np.clip(ref,0,255)).astype(np.int16))
print(json.dumps({'mae_gray':float(err.mean()),'max_gray':float(err.max()),'p95_gray':float(np.percentile(err,95)),'psnr_vs_source_db':None if mse==0 else float(10*np.log10(255**2/mse)),'rounded_byte_max_gray':int(byte_delta.max()),'rounded_changed_pixels':int(np.count_nonzero(byte_delta)),'phase6_mean_error_gray':[[float(delta[y::6,x::6].mean()) for x in range(6)] for y in range(6)],'comparison_target':'matching source output, not GT'},indent=2))
