"""Select with validation GT, then full-frame test quality, timing and videos."""
import argparse,fcntl,json,math,sys,time,subprocess,gc
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=('ordinary','special'));a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-PACKED-FRONT-20260926'
sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/npu_graph_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp']
from load_packed_model import load_packed_model
sys.path.insert(0,'/tmp/trajectory_student_20260926/runtime')
from load_student_model import load_student_model
from load_candidate import load_candidate
from graph_variants import FullGraphNine
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'))
from ir_sr.training import dataset_for_config
from ir_sr.metrics import image_metrics
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False)
lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
cp=root/'runs/SS928-NIGHT-NINE-REFINE-20260926/selected_models'/f'{a.scene}_factor24.pt';bp=root/'runs/SS928-NIGHT-NINE-SPEED-20260926/models'/f'{a.scene}_fused.pt'
def model(step=None):
 if step is None:
  previous=root/'runs/SS928-TRAJECTORY-STUDENT-COMPACT-20260926'
  chosen=json.loads((previous/f'{a.scene}_selection.json').read_text())['selected_step']
  return load_student_model(previous/f'{a.scene}_student_{chosen:06d}.pt',cp,bp)
 return load_packed_model(out/f'{a.scene}_packed_front_{step:06d}.pt',cp,bp)
def tensors(ds,row):
 x=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));return x,ctx[None].cuda()
def prediction(m,x,c):
 y=m(x,c).clamp(0,255).round();return F.avg_pool2d(y,3,3)[0,0].cpu().numpy()
def truth(row):return np.asarray(Image.open((Path(state['config']['data_root'])/row['target']['path']).with_name(f'{row["frame_id"]:06d}.png')),dtype=np.float32)
def psnr(pred,gt):return -10*math.log10(max(float(np.mean(((pred[3:-3,3:-3].astype(np.float64)-gt[3:-3,3:-3])/255)**2)),1e-12))
base=model();ds=dataset_for_config(state['config'],'val');rows=[r for r in ds.records if r['scene_id']=='night_'+a.scene]
val={};ry,rx,rh,rw=rows[0]['eval_crop_tlhw'];roi=(slice(ry,ry+rh),slice(rx,rx+rw))
with torch.inference_mode():
 for step in (None,500,1000,2000,5000,10000):
  m=base if step is None else model(step);values=[]
  for row in rows:
   x,c=tensors(ds,row);pred=prediction(m,x,c);gt=truth(row);values.append(psnr(pred[roi],gt[roi]));del x,c
  val[str(step)]=float(np.mean(values));print('VAL',a.scene,step,val[str(step)],flush=True)
  if step is not None:del m;torch.cuda.empty_cache()
selected=max((500,1000,2000,5000,10000),key=lambda v:val[str(v)])
student=model(selected)
# Save selection before slower full test; test split never selects checkpoint.
(out/f'{a.scene}_selection.json').write_text(json.dumps({'validation_psnr':val,'selected_step':selected,'selection_split':'val','test_used_for_selection':False},indent=2))
compiled={name:torch.compile(m,fullgraph=True,options={'triton.cudagraphs':False}) for name,m in [('base',base),('student',student)]}
ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene);ry,rx,rh,rw=rec['eval_crop_tlhw'];roi=(slice(ry,ry+rh),slice(rx,rx+rw))
probe_x,probe_c=tensors(ds,dict(rec,frame_id=60 if special else 20))
with torch.inference_mode():
 source_student=student(probe_x,probe_c);compiled_student=compiled['student'](probe_x,probe_c);compiled_base=compiled['base'](probe_x,probe_c)
 execution_check={'compiled_vs_source_float_mean_gray':float((compiled_student-source_student).abs().mean()),'compiled_candidate_vs_base_float_mean_gray':float((compiled_student-compiled_base).abs().mean())}
 assert execution_check['compiled_candidate_vs_base_float_mean_gray']>0,execution_check
 print('EXECUTION_CHECK',execution_check,flush=True)
del probe_x,probe_c,source_student,compiled_student,compiled_base
count=120 if special else 60;scores={n:[] for n in compiled};ssim={n:[] for n in compiled};series={n:[] for n in ('base','student','gt')};diffs=[]
video=out/f'{a.scene}_student_full.mp4'
writer=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','5120x1080','-r','12','-i','-','-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(video)],stdin=subprocess.PIPE)
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',27)
with torch.inference_mode():
 for frame in range(count):
  row=dict(rec,frame_id=frame);x,c=tensors(ds,row);full={n:prediction(m,x,c) for n,m in compiled.items()};gt=truth(row)
  for n in compiled:
   pr=full[n][roi];tr=gt[roi];scores[n].append(psnr(pr,tr));series[n].append(pr[::2,::2].astype(np.float16))
   if frame%10==0:ssim[n].append(image_metrics(torch.from_numpy(np.ascontiguousarray(pr/255))[None,None],torch.from_numpy(np.ascontiguousarray(tr/255))[None,None],border=3)['ssim'])
  series['gt'].append(gt[roi][::2,::2].astype(np.float16));diffs.append(float(np.abs(full['student']-full['base']).mean()))
  norm=ds.normalization_for(row);raw=(ds.base._raw(row).astype(np.float32)-norm['offset'])/norm['scale']*255
  canvas=Image.new('RGB',(5120,1080),'#17191c');draw=ImageDraw.Draw(canvas)
  for j,(label,array) in enumerate(zip(('输入 RAW','GT','当前八通道版本','轨迹门控联合候选'),(raw,gt,full['base'],full['student']))):
   draw.text((j*1280+18,12),f'{label}  第{frame:03d}帧',font=font,fill='white');canvas.paste(Image.fromarray(np.rint(np.clip(array,0,255)).astype(np.uint8)).convert('RGB'),(j*1280,56))
  writer.stdin.write(np.asarray(canvas).tobytes())
  if frame in (0,count//2,count-1):canvas.save(out/f'{a.scene}_student_frame_{frame:03d}.jpg',quality=95)
  if frame%10==0:print('TEST',a.scene,frame,scores['student'][-1]-scores['base'][-1],flush=True)
  del x,c
writer.stdin.close();assert writer.wait()==0
arr={k:np.stack(v).astype(np.float32) for k,v in series.items()};gt=arr['gt'];mean=gt.mean(0);temporal=gt.std(0)
import cv2
grad=np.hypot(cv2.Sobel(mean,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(mean,cv2.CV_32F,0,1,ksize=3)/8)
static=(temporal<=1)&(grad<=5)&(mean>5)&(mean<250);static[:4]=static[-4:]=False;static[:,:4]=static[:,-4:]=False;moving=temporal>2;weak=(temporal>2)&(temporal<=8)&(mean>5)&(mean<250)
assert static.any() and moving.any() and weak.any(),{'static':int(static.sum()),'moving':int(moving.sum()),'weak':int(weak.sum())}
assert np.isfinite(gt).all() and float(temporal.max())>0
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
for n in compiled:
 v=arr[n];err=v-gt;pm,ps=stats(v)
 quality[n]={'psnr_db':float(np.mean(scores[n])),'ssim_every10':float(np.mean(ssim[n])),'static_output_temporal_std_gray':float(v.std(0)[static].mean()),'static_error_temporal_std_gray':float(err.std(0)[static].mean()),'moving_mae_gray':float(np.abs(err[:,moving]).mean()),'low_amplitude_moving_mae_gray':float(np.abs(err[:,weak]).mean()),'brightness_error_temporal_std_gray':float(np.median((pm-gm).std(0))),'contrast_error_temporal_std_gray':float(np.median((ps-gs).std(0)))}
# Paired complete-model benchmark, same input/output dtype for both models.
x,c=tensors(ds,dict(rec,frame_id=60 if special else 20));timing={}
with torch.inference_mode():
 for n,m in compiled.items():
  for _ in range(30):m(x,c)
  torch.cuda.synchronize();values=[]
  for _ in range(600):
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();m(x,c);end.record();end.synchronize();values.append(start.elapsed_time(end))
  timing[n]={'mean_ms':float(np.mean(values)),'p95_ms':float(np.percentile(values,95)),'samples_ms':values}
report={'execution_check':execution_check,'scene':a.scene,'selected_step':selected,'validation_psnr':val,'quality':quality,'timing':timing,'count':count,'roi':rec['eval_crop_tlhw'],'video':str(video),'full_sensor_video':True,'output_shape':[1,1,3072,3840],'full_model_input_dtype':'float32','full_model_output_dtype':'float32','GPU':'RTX5090','warmups':30,'samples':600,'TF32':False,'NPU_tested':False,'baseline':'published compact8 70a68eb','candidate':'joint learned packed denoising front; half shuffle output','mask_pixels':{'static':int(static.sum()),'moving':int(moving.sum()),'weak':int(weak.sum())},'weak_motion_is_proxy':True,'test_used_for_checkpoint_selection':False,'mean_full_rawgrid_difference_gray':float(np.mean(diffs))}
(out/f'{a.scene}_evaluation.json').write_text(json.dumps(report,indent=2));print('FINISHED',a.scene,json.dumps({'quality':quality,'timing':{n:{k:v for k,v in r.items() if k!='samples_ms'} for n,r in timing.items()}},indent=2),flush=True)
