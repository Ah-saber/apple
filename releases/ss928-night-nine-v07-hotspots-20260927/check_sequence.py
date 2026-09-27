"""Full-canvas source regression over existing 60/120-frame development clips."""
import argparse,fcntl,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr')
sys.path.insert(0,str(Path(__file__).parent/'runtime'));from operator_candidates import CandidateSystem
sys.path.insert(0,'/tmp/ss928_release_verification_20260926/ss928-night-nine-system-speed-20260926/runtime');from load_system_release import load_release
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene)
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
base,_=load_release(a.scene,'front_f32');specs=[('statistics_'+mode,{'statistics':mode}) for mode in ('temporal_first','temporal_f32','pack_first')]+[('output_'+mode,{'output':mode}) for mode in ('reshape','deconv6','deconv2_shuffle3','deconv3_shuffle2','shuffle2_3','shuffle3_2')]+[('input_nchw_half',{'input_half':True}),('output_half',{'output_half':True}),('scale_packed_f32',{'scale':'packed_f32'}),('scale_packed_half',{'scale':'packed_half'})]
models={name:CandidateSystem(base,**kw).eval() for name,kw in specs};metrics={name:[] for name in models};count=120 if special else 60
with torch.inference_mode():
 for frame in range(count):
  row=dict(rec,frame_id=frame);x=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();expected=base(x,ctx);expected_byte=expected.round()
  for name,model in models.items():
   actual=model(x.half() if model.input_half else x,ctx).float();assert tuple(actual.shape)==(1,1,3072,3840);d=(actual-expected).abs();byte=(actual.round()-expected_byte).abs();metrics[name].append({'frame':frame,'max_gray':float(d.max()),'mean_gray':float(d.mean()),'rounded_byte_max_error':float(byte.max()),'rounded_byte_changed_fraction':float((byte!=0).float().mean())})
   if name.startswith('output_') and name!='output_half':assert d.max()==0,(name,frame)
   if name in ('input_nchw_half','scale_packed_f32'):assert d.max()==0,(name,frame)
  if frame%10==0:print('FRAME',a.scene,frame,flush=True)
summary={name:{'full_frames':count,'max_gray':max(r['max_gray'] for r in rows),'mean_gray':sum(r['mean_gray'] for r in rows)/count,'rounded_byte_max_error':max(r['rounded_byte_max_error'] for r in rows),'mean_byte_changed_fraction':sum(r['rounded_byte_changed_fraction'] for r in rows)/count} for name,rows in metrics.items()}
path=root/'runs/SS928-V07-FOLLOWUP-20260927'/f'{a.scene}_sequence_checks.json';path.write_text(json.dumps({'scene':a.scene,'shape':[1,1,3072,3840],'count':count,'baseline':'v0.7 front_f32 source','execution':'source eager','GT_metrics_recomputed':False,'NPU_measured':False,'summary':summary,'per_frame':metrics},indent=2));print('COMPLETE',a.scene,summary,flush=True)
