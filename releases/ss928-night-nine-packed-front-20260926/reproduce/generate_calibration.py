"""Generate quantization calibration inputs solely from train ROIs; no GT."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-PACKED-FRONT-20260926/calibration';out.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=('ordinary','special'));a=p.parse_args()
records=[]
for scene in [a.scene]:
 special=scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
 sys.path.insert(0,str(root/'code/worktrees'/work/'src'))
 # Both worktrees expose the same dataset API; config selects scene correction.
 from ir_sr.training import dataset_for_config
 state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'train');rows=[r for r in ds.records if r['scene_id']=='night_'+scene]
 for j,idx in enumerate(np.linspace(0,len(rows)-1,4,dtype=int)):
  row=rows[idx];cy,cx,ch,cw=row['train_roi_tlhw'];raw=ds.normalized_stack(row,(cy,cx,ch,cw)).cpu().numpy();pad=((0,0),(cy,1024-cy-ch),(cx,1280-cx-cw));stack=np.pad(raw,pad,mode='reflect') if any(sum(v)>0 for v in pad) else raw
  ctx,_=ds.context_for(row,crop_tlhw=(cy,cx,ch,cw));ctx=ctx.cpu().numpy();assert stack.shape==(9,1024,1280) and ctx.shape==(1,64,64)
  path=out/f'{scene}_train_{j:02d}.npz';np.savez_compressed(path,nine_raw=stack[None].astype(np.float32),reference_thumb=ctx[None].astype(np.float32))
  item={'scene':scene,'split':'train','sample_id':row['sample_id'],'frame_id':row['frame_id'],'train_roi_tlhw':[cy,cx,ch,cw],'padding':pad,'outside_training_roi':'reflection of training ROI only; held-out side RAW excluded','GT_read':False,'path':path.name,'bytes':path.stat().st_size};records.append(item);print(item,flush=True)
(out/f'{a.scene}_calibration.json').write_text(json.dumps({'samples':records,'GT_used':False,'test_used':False,'count_per_scene':4,'dataset_module':str(Path(sys.modules['ir_sr.training'].__file__)),'purpose':'small independent initial quantization calibration set; cannot guarantee all deployment noise/brightness coverage'},indent=2))
