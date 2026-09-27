"""Full-sequence source quality matrix, retaining per-frame GT and full output."""
import argparse,fcntl,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['ordinary','special']);p.add_argument('--tag',default='quality_matrix');p.add_argument('--variants',nargs='+',default=['baseline','front_init','front_refined','up2','up3','up6','joint_refined_up2','up2_image','up3_image']);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-V07-NATIVE4-20260927';sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from continuation_candidates import load_continuation as load_joint,prepare_inputs
def load_case(root,scene,case):
 m=load_joint(scene,case,out);return m,{'model':m}
def case_inputs(x,ctx,info):return prepare_inputs(info['model'],x,ctx)
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config;from ir_sr.metrics import image_metrics
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene);ry,rx,rh,rw=rec['eval_crop_tlhw'];roi=(slice(ry,ry+rh),slice(rx,rx+rw));lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory);loaded={k:load_case(root,a.scene,k) for k in a.variants};models={k:v[0] for k,v in loaded.items()};scores={k:[] for k in models};ssim={k:[] for k in models};series={k:[] for k in list(models)+['gt']};full_differences={k:[] for k in models};count=120 if special else 60
with torch.inference_mode():
 for frame in range(count):
  row=dict(rec,frame_id=frame);x=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();path=(Path(state['config']['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png');gt=np.asarray(Image.open(path),dtype=np.float32);series['gt'].append(gt[roi][::2,::2].astype(np.float16));base=None
  for name,m in models.items():
   xx,cc=case_inputs(x,ctx,loaded[name][1]);value=m(xx,cc).float().clamp(0,255).round();assert tuple(value.shape)==(1,1,3072,3840);pred=F.avg_pool2d(value,3,3)[0,0].cpu().numpy();pr=pred[roi];tr=gt[roi];mse=float(np.mean(((pr[3:-3,3:-3].astype(np.float64)-tr[3:-3,3:-3])/255)**2));scores[name].append(-10*math.log10(max(mse,1e-12)));series[name].append(pr[::2,::2].astype(np.float16))
   if frame%10==0:ssim[name].append(image_metrics(torch.from_numpy(np.ascontiguousarray(pr/255))[None,None],torch.from_numpy(np.ascontiguousarray(tr/255))[None,None],border=3)['ssim'])
   if name=='baseline':base=pred
   if base is not None:full_differences[name].append(float(np.abs(pred-base).mean()))
  if frame%10==0:print('FRAME',a.scene,frame,{n:round(scores[n][-1]-scores['baseline'][-1],5) for n in models},flush=True)
arr={k:np.stack(v).astype(np.float32) for k,v in series.items()};gt=arr['gt'];mean=gt.mean(0);temporal=gt.std(0);import cv2
grad=np.hypot(cv2.Sobel(mean,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(mean,cv2.CV_32F,0,1,ksize=3)/8);static=(temporal<=1)&(grad<=5)&(mean>5)&(mean<250);static[:4]=static[-4:]=False;static[:,:4]=static[:,-4:]=False;moving=temporal>2;weak=(temporal>2)&(temporal<=8)&(mean>5)&(mean<250);assert static.any() and moving.any() and weak.any()
def stats(seq):
 mids=[];spread=[]
 for pred in seq:
  h,w=pred.shape;pm=[];ps=[]
  for y in range(8):
   for x in range(8):
    sl=(slice(y*h//8,(y+1)*h//8),slice(x*w//8,(x+1)*w//8));lo,mid,hi=np.percentile(pred[sl],(10,50,90));pm.append(mid);ps.append(hi-lo)
  mids.append(pm);spread.append(ps)
 return np.asarray(mids),np.asarray(spread)
gm,gs=stats(gt);quality={}
for n in models:
 v=arr[n];err=v-gt;pm,ps=stats(v);quality[n]={'psnr_db':float(np.mean(scores[n])),'ssim_every10':float(np.mean(ssim[n])),'static_output_temporal_std_gray':float(v.std(0)[static].mean()),'static_error_temporal_std_gray':float(err.std(0)[static].mean()),'moving_mae_gray':float(np.abs(err[:,moving]).mean()),'low_amplitude_moving_mae_gray':float(np.abs(err[:,weak]).mean()),'brightness_error_temporal_std_gray':float(np.median((pm-gm).std(0))),'contrast_error_temporal_std_gray':float(np.median((ps-gs).std(0))),'mean_full_rawgrid_difference_gray':float(np.mean(full_differences[n]))};print('QUALITY',a.scene,n,quality[n],flush=True)
report={'scene':a.scene,'execution':'source','count':count,'roi':rec['eval_crop_tlhw'],'quality':quality,'per_frame_psnr':scores,'baseline':'v0.7 front_f32 source','NPU_verified':False,'test_used_for_checkpoint_selection':False,'test_used_for_architecture_assessment':True,'GT_native_scale_only':True};tag=a.tag;(out/f'{a.scene}_{tag}.json').write_text(json.dumps(report,indent=2));print('FINISHED',flush=True)
