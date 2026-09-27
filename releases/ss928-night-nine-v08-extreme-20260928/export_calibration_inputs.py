"""Training-only full-frame Half inputs for device calibration; no GT outputs."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-EXTREME-20260927/calibration_inputs';out.mkdir(exist_ok=True)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');records=[r for r in ds.records if r['scene_id']=='night_'+a.scene][:-4];rows=[records[int(i)] for i in np.linspace(0,len(records)-1,4,dtype=int)];manifest=[]
for i,row in enumerate(rows):
 raw=ds.normalized_stack(row,(0,0,1024,1280))[None].numpy();ctx,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].numpy().astype(np.float32);half=raw.astype(np.float16);path=out/f'{a.scene}_train_{i:02d}_frame_{row["frame_id"]:03d}.npz'
 if path.exists():raise FileExistsError(path)
 np.savez_compressed(path,nine_raw=half,reference_thumb=ctx);manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'scene':a.scene,'sample_id':row['sample_id'],'frame_id':row['frame_id'],'split':'train; last four records excluded','raw_shape':list(half.shape),'raw_dtype':'float16','context_shape':list(ctx.shape),'context_dtype':'float32','raw_half_max_abs_error':float(np.abs(raw-half.astype(np.float32)).max()),'GT_included':False,'model_statistics_precomputed':False});print('INPUT',path.name,flush=True)
(out/f'{a.scene}_manifest.json').write_text(json.dumps({'inputs':manifest,'normalization':'unchanged frozen dataset configuration used by source evaluations'},indent=2))
