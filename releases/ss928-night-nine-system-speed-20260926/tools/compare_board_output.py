import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--dtype',choices=['float32','float16','uint8'],default='float32');p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();shape=(1,1,3072,3840);actual=np.fromfile(a.board,dtype=a.dtype)
if actual.size!=np.prod(shape):raise ValueError(f'Expect full {shape}, got {actual.size} samples')
actual=actual.reshape(shape).astype(np.float32)
with np.load(a.reference,allow_pickle=False) as v:expected=v['display_gray'].astype(np.float32)
if a.dtype=='uint8':expected=np.rint(expected).clip(0,255)
if expected.shape!=shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Invalid reference or board output')
error=actual-expected;phases=error[0,0].reshape(512,6,640,6).mean((0,2));result={'shape':list(shape),'MAE_gray':float(np.abs(error).mean()),'max_error_gray':float(np.abs(error).max()),'bias_gray':float(error.mean()),'phase_6x6_bias_gray':phases.tolist(),'phase_bias_spread_gray':float(phases.max()-phases.min()),'reference_rounded_for_uint8':a.dtype=='uint8','board_vs_matching_source_only':True,'GT_or_temporal_verified':False};a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
