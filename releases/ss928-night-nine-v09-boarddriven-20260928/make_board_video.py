"""Render every complete sensor frame for source comparisons, plus native 3x PNGs."""
import argparse,fcntl,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser()
p.add_argument('--scene',choices=['ordinary','special'],required=True)
p.add_argument('--variant',required=True)
a=p.parse_args()
root=Path('/data/zhangbenzhuang/huawei_sr')
run=root/'runs/SS928-BOARD-V09-20260928'
out=run/'videos';out.mkdir(exist_ok=True)
sys.path.insert(0,str(Path(__file__).parent))
from board_candidates import load_board,prepare_inputs
special=a.scene=='special'
work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925'
training='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1'
sys.path.insert(0,str(root/'code/worktrees'/work/'src'))
from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/training/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False)
ds=dataset_for_config(state['config'],'test')
rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene)
count=120 if special else 60
lease=(run.parent/'TASK-019-gpu1.lock').open('a')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
models={key:load_board(a.scene,case,run) for key,case in [('baseline','v08_c11'),('candidate',a.variant)]}
video=out/f'{a.scene}_{a.variant}_full_comparison.mp4'
writer=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','5120x1080','-r','12','-i','-','-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(video)],stdin=subprocess.PIPE)
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',27)
differences=[]
try:
 with torch.inference_mode():
  for frame in range(count):
   row=dict(rec,frame_id=frame)
   x=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda()
   ctx,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda()
   images={}
   for key,model in models.items():
    xx,cc=prepare_inputs(model,x,ctx)
    value=model(xx,cc).float().clamp(0,255).round()
    assert tuple(value.shape)==(1,1,3072,3840)
    images[key]=F.avg_pool2d(value,3,3)[0,0].cpu().numpy()
    if frame==count//2:
     Image.fromarray(value[0,0].byte().cpu().numpy()).save(out/f'{a.scene}_{a.variant}_{key}_native3x_frame_{frame:03d}.png')
   target=(Path(state['config']['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')
   gt=np.asarray(Image.open(target),dtype=np.float32)
   norm=ds.normalization_for(row)
   raw=(ds.base._raw(row).astype(np.float32)-norm['offset'])/norm['scale']*255
   canvas=Image.new('RGB',(5120,1080),'#17191c')
   draw=ImageDraw.Draw(canvas)
   for j,(label,arr) in enumerate(zip(('输入 RAW','GT','v0.8 C11 源图',a.variant),(raw,gt,images['baseline'],images['candidate']))):
    assert arr.shape==(1024,1280)
    draw.text((j*1280+18,12),f'{label} 第{frame:03d}帧',font=font,fill='white')
    canvas.paste(Image.fromarray(np.rint(np.clip(arr,0,255)).astype(np.uint8)).convert('RGB'),(j*1280,56))
   writer.stdin.write(np.asarray(canvas).tobytes())
   differences.append(float(np.abs(images['candidate']-images['baseline']).mean()))
   if frame in (0,count//2,count-1):canvas.save(out/f'{a.scene}_{a.variant}_frame_{frame:03d}.jpg',quality=95)
   if frame%10==0:print('VIDEO',a.scene,a.variant,frame,flush=True)
finally:
 writer.stdin.close()
 assert writer.wait()==0
meta={'scene':a.scene,'variant':a.variant,'baseline':'v08_c11','count':count,'fps':12,'video_shape':[1080,5120],'each_panel_sensor_shape':[1024,1280],'model_output_shape':[1,1,3072,3840],'display':'full field at sensor grid; 3x output averaged per 3x3 cell','source_execution':True,'NPU_verified':False,'mean_full_rawgrid_difference_gray':float(np.mean(differences))}
(out/f'{a.scene}_{a.variant}_video.json').write_text(json.dumps(meta,indent=2))
print('FINISHED',video,flush=True)
