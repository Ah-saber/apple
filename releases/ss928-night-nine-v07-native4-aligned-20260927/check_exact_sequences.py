"""Verify source full-resolution equality of phase scaling on every frame."""
import argparse,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--tag',default='exact_full_sequences');p.add_argument('--cases',nargs='+',default=['combo_scaled9_exact','combo_aligned16_nearest','combo_input9_half_scaled9_exact','combo_input16_half_native_stats_scaled9_exact']);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-NATIVE4-20260927';sys.path.insert(0,str(Path(__file__).parent/'runtime'));from continuation_candidates import load_continuation,prepare_inputs
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene);lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
old=load_continuation(a.scene,'previous_combo',out);models={case:load_continuation(a.scene,case,out) for case in a.cases};rows={case:[] for case in a.cases}
with torch.inference_mode():
 for frame in range(120 if special else 60):
  rr=dict(rec,frame_id=frame);x=ds.normalized_stack(rr,(0,0,1024,1280))[None].cuda();c,_=ds.context_for(rr,crop_tlhw=(0,0,1024,1280));c=c[None].cuda();y=old(x,c)
  for case,model in models.items():
   xx,cc=prepare_inputs(model,x,c);z=model(xx,cc);assert y.shape==z.shape==(1,1,3072,3840);d=(y-z).abs();rows[case].append({'frame':frame,'max_gray':float(d.max()),'mean_gray':float(d.mean()),'bit_exact':bool(torch.equal(y,z))})
  if frame%20==0:print('EXACT',a.scene,frame,{case:rows[case][-1] for case in models},flush=True)
(out/f'{a.scene}_{a.tag}.json').write_text(json.dumps({'scene':a.scene,'execution':'uncompiled source','baseline':'cada778 previous_combo','candidates':a.cases,'full_output':[1,1,3072,3840],'all_frames_bit_exact':{case:all(r['bit_exact'] for r in values) for case,values in rows.items()},'frames':rows,'NPU_verified':False},indent=2));print('COMPLETE',a.scene,flush=True)
