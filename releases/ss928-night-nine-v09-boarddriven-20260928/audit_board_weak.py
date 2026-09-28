"""Check two previously recorded ordinary weak-motion windows without training on them."""
import argparse,fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--tag',default='published_weak_target_audit');p.add_argument('--variants',nargs='+');a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-BOARD-V09-20260928'
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from board_candidates import load_board as load_joint,prepare_inputs
def load_case(root,scene,case):
 m=load_joint(scene,case,out);return m,{'model':m}
def case_inputs(x,c,info):return prepare_inputs(info['model'],x,c)
sys.path.insert(0,str(root/'code/worktrees/ss928-quality-night-nine-frame-20260925/src'))
from ir_sr.training import dataset_for_config
state=torch.load(root/'runs/SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1/checkpoints/step_000002000.pt',map_location='cpu',weights_only=False)
ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_ordinary')
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
variants=a.variants or ['body3_quantsearch','combo_input9_half_aligned16_nearest','cal_front3x1_trained_native4_alignpixel','cal_front3x1_trained_body3_native4_alignpixel','cal_front3x1_trained_body2_native4_alignpixel','cal_front3x1_trained_body3_native4_alignpixel_w8allpo','cal_front3x1_trained_body3_native4_alignpixel_w8allpt','quarter_w32_3x3_body1_stage1_fusedtail','quarter_w32_3x3_body1_qat_w8allpo_fusedtail','cal_graph_quarter_w32_3x3_body1_qat_w8allpo_fusedtail_sdk_w8po','cal_graph_quarter_w32_3x3_body1_qat_w8allpo_fusedtail_sdk_w8pt']
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
(out/(a.tag+'.json')).write_text(json.dumps({'windows':reports,'GT_used_for_training':False,'test_windows_used_for_checkpoint_selection':False,'metric':'GT-defined low-amplitude temporal residual projection in previously recorded 32x32 windows; not detection labels'},indent=2))
