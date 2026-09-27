"""Exact inherited RAW normalization and dynamic correction replay kit."""
import sys,json,hashlib,argparse,time
from pathlib import Path
import numpy as np
import torch
import cv2
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2)
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927/sequence_inputs'/a.scene;out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError(out/'manifest.json')
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene);count=120 if special else 60
sys.path.insert(0,str(Path(__file__).parent));from reconstruct_sequence import dynamic_correct
rows=[];raws=[];contexts=[];times=[]
for frame in range(count):
 row=dict(rec,frame_id=frame);raw_full=ds.base._raw(row);raw=raw_full[:1024,:1280];assert raw.dtype in (np.uint16,np.float32) and raw.shape==(1024,1280),(raw.dtype,raw.shape);raws.append(raw.copy());contexts.append(ds.context_for(row,crop_tlhw=(0,0,1024,1280))[0].numpy().astype(np.float32));n=ds.base.normalization_for(row)
 rows.append({'frame':frame,'source_raw_dtype':str(raw.dtype),'source_raw_shape':list(raw_full.shape),'model_input_crop_tlhw':[0,0,1024,1280],'frame_ids':list(map(int,ds.frame_ids(row))),'offset':float(n['offset']),'scale':float(n['scale'])})
 if len(raws)==16 or frame==count-1:
  start=frame-len(raws)+1;name=f'raw_{start:03d}_{frame:03d}.npz';np.savez_compressed(out/name,raw=np.stack(raws));raws=[]
np.savez_compressed(out/'contexts.npz',reference_thumb=np.stack(contexts));del contexts
cache={};chunks=sorted(out.glob('raw_*.npz'))
for ch in chunks:
 start=int(ch.stem.split('_')[1]);arr=np.load(ch)['raw'];cache.update({start+i:v for i,v in enumerate(arr)})
for row in rows:
 stack=np.stack([(cache[i].astype(np.float32)-rows[i]['offset'])/rows[i]['scale'] for i in row['frame_ids']]);t=time.perf_counter();x=dynamic_correct(stack) if ds.dynamic_correction else stack;times.append((time.perf_counter()-t)*1000);expected=ds.normalized_stack(dict(rec,frame_id=row['frame']),(0,0,1024,1280)).numpy();assert np.array_equal(x,expected),row['frame'];row['corrected_f32_sha256']=hashlib.sha256(x.tobytes()).hexdigest();row['corrected_f16_sha256']=hashlib.sha256(x.astype(np.float16).tobytes()).hexdigest()
 if row['frame']%10==0:print('SEQUENCE',a.scene,row['frame'],flush=True)
files=[{'file':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in out.glob('*.npz')]
(out/'manifest.json').write_text(json.dumps({'scene':a.scene,'frames':rows,'files':files,'dynamic_correction':bool(ds.dynamic_correction),'opencv_version':cv2.__version__,'algorithm_outside_measured_GPU_model':True,'correction_cpu_ms_mean':float(np.mean(times)),'correction_cpu_ms_p95':float(np.percentile(times,95)),'cpu_threads':2,'all_frame_exact_reconstruction':True,'nine_frames_are_real_causal_with_initial_duplicates':True},indent=2));print('SEQUENCE_COMPLETE',a.scene,flush=True)
