"""Replay inherited preprocessing; this includes substantial denoising outside NN."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import cv2

def dynamic_correct(stack):
 reference=np.median(stack[:6],axis=0);result=[]
 for frame in stack:
  residual=frame-reference;common=np.median(residual);row=np.median(residual-common,axis=1)[:,None];col=np.median(residual-common-row,axis=0)[None,:];broad=cv2.GaussianBlur(residual-common-row-col,(0,0),4);result.append(frame-common-row-col-.9*broad)
 return np.ascontiguousarray(np.stack(result),dtype=np.float32)

def reconstruct(directory,frame):
 directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text());rows=meta['frames'];r=rows[frame];raw={}
 for entry in meta['files']:
  if not entry['file'].startswith('raw_'):continue
  start,end=map(int,Path(entry['file']).stem.split('_')[1:]);needed=[i for i in r['frame_ids'] if start<=i<=end]
  if needed:
   path=directory/entry['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];arr=np.load(path)['raw'];raw.update({i:arr[i-start] for i in needed})
 x=np.stack([(raw[i].astype(np.float32)-rows[i]['offset'])/rows[i]['scale'] for i in r['frame_ids']]);x=dynamic_correct(x) if meta['dynamic_correction'] else x
 if hashlib.sha256(x.tobytes()).hexdigest()!=r['corrected_f32_sha256']:raise ValueError('Corrected input differs: verify NumPy/OpenCV versions and CPU arithmetic; do not change normalization silently')
 return x[None],np.load(directory/'contexts.npz')['reference_thumb'][frame][None]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--half',action='store_true',help='Optional lossy Half storage; default keeps full normalized Float32 for all declared interfaces');a=p.parse_args();path=Path(a.output)
 if path.exists():raise FileExistsError(path)
 x,c=reconstruct(a.sequence,a.frame);np.savez_compressed(path,nine_raw=x.astype(np.float16) if a.half else x,reference_thumb=c);print(path)
