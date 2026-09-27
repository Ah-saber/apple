"""Check two previously recorded ordinary weak-motion windows without training on them."""
import fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-NATIVE4-20260927'
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from continuation_candidates import load_continuation as load_joint,prepare_inputs
def load_case(root,scene,case):
 m=load_joint(scene,case,out);return m,{'model':m}
def case_inputs(x,c,info):return prepare_inputs(info['model'],x,c)
sys.path.insert(0,str(root/'code/worktrees/ss928-quality-night-nine-frame-20260925/src'))
from ir_sr.training import dataset_for_config
state=torch.load(root/'runs/SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1/checkpoints/step_000002000.pt',map_location='cpu',weights_only=False)
ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_ordinary')
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
variants=['combo_stats3_relu32_aligned16_half_output_fixed','combo_stats3_scaled9_exact','combo_stats3_relu32_scaled9_exact','combo_input16_half_stats3_scaled9_exact','combo_stats3_aligned16_half_output_fixed','combo_aligned16_nearest_half_output','combo_aligned16_half_output_fixed','combo_input9_half_aligned16_nearest','combo_input16_half_native_stats_aligned16_fixed','combo_project_after_resize_aligned16_nearest','native4_linear_ref_dither_float_trained','combo_project_after_resize_scaled9_exact','project_after_resize_scaled36_exact','combo_input9_half_scaled9_exact','combo_input16_half_native_stats_scaled9_exact','relu_stats32_scaled36_exact','combo_relu_stats32_scaled9_exact','combo_aligned16_nearest','combo_aligned16_fixed','combo_aligned16_folded_fixed','combo_tail5_exact_edges','combo_tail5_interior','relu_stats24_scaled36_exact','combo_relu_stats24_scaled9_exact','native4_linear_dither_trained','native4_linear_dither_float_trained','native4_relu8_dither_trained','native4_relu8_ref_dither_trained','baseline','previous_combo','native4_linear_trained','native4_relu8_trained','native4_relu8_ref_trained','fused9_3_channel','fused9_3_image','fused9_6_channel','scaled36_exact','scaled36_folded','scaled9_exact','scaled9_folded','combo_scaled9_exact','combo_scaled9_folded']
loaded={k:load_case(root,'ordinary',k) for k in variants};models={k:v[0] for k,v in loaded.items()}
reports=[]
with torch.inference_mode():
 for frame,cy,cx in ((46,764,258),(58,736,522)):
  series={k:[] for k in ['gt']+variants}
  for f in (frame-2,frame-1,frame):
   row=dict(rec,frame_id=f);x=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda();c,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));c=c[None].cuda()
   path=(Path(state['config']['data_root'])/rec['target']['path']).with_name(f'{f:06d}.png');gt=np.asarray(Image.open(path),dtype=np.float32)
   sl=(slice(cy-16,cy+16),slice(cx-16,cx+16));series['gt'].append(gt[sl])
   for name,m in models.items():
    xx,cc=case_inputs(x,c,loaded[name][1]);value=m(xx,cc).float().round().clamp(0,255);raw=F.avg_pool2d(value,3,3)[0,0].cpu().numpy();series[name].append(raw[sl])
  residual={k:v[-1]-(v[0]+v[1])*.5 for k,v in series.items()};q=residual['gt']-np.median(residual['gt']);mask=(np.abs(q)>=.5)&(np.abs(q)<=8)
  item={'frame':frame,'center_yx':[cy,cx],'weak_temporal_pixels':int(mask.sum()),'models':{}}
  for k in models:
   val=series[k][-1];tar=series['gt'][-1];r=residual[k]-np.median(residual[k]);den=float(np.sum(q[mask]**2))
   item['models'][k]={'window_mae_gray':float(np.abs(val-tar).mean()),'weak_temporal_response_projection':None if den==0 else float(np.sum(r[mask]*q[mask])/den),'weak_temporal_residual_mae_gray':None if not mask.any() else float(np.abs(r[mask]-q[mask]).mean())}
  reports.append(item);print(item,flush=True)
(out/'system_weak_target_audit.json').write_text(json.dumps({'windows':reports,'GT_used_for_training':False,'test_windows_used_for_checkpoint_selection':False,'metric':'GT-defined low-amplitude temporal residual projection in previously recorded 32x32 windows; not detection labels'},indent=2))
