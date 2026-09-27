"""Build declared input bytes from existing normalized nine-frame NPZ vectors."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--layout',choices=['nhwc16','native5'],required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True);v=np.load(a.source);x=v['nine_raw'];assert x.ndim==4 and x.shape[1]==9
half=x.astype(np.float16);padded=np.pad(half,((0,0),(0,7),(0,0),(0,0)));data=np.ascontiguousarray(padded.transpose(0,2,3,1))
if a.layout=='native5':data=data[:,None]
assert np.all(data.reshape(data.shape[0],x.shape[2],x.shape[3],16)[...,9:]==0)
context=np.ascontiguousarray(v['reference_thumb'],dtype=np.float32);np.save(out/'input.npy',data);np.save(out/'reference_thumb.npy',context);data.tofile(out/'input.f16');context.tofile(out/'reference_thumb.f32')
(out/'manifest.json').write_text(json.dumps({'layout':a.layout,'input_shape':list(data.shape),'input_dtype':str(data.dtype),'input_bytes':data.nbytes,'input_sha256':hashlib.sha256(data.tobytes()).hexdigest(),'real_temporal_channels':9,'zero_slots':7,'context_shape':list(context.shape),'context_dtype':str(context.dtype),'representation_only_no_model_statistics':True,'SDK_rank_layout_support_verified':False},indent=2))
