"""Complete-sensor RAW/GT/v0.6/new comparison, no cropped-only output."""
import argparse,fcntl,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();root=Path('/data/zhangbenzhuang/huawei_sr');out=root/'runs/SS928-SYSTEM-SPEED-20260926/videos';out.mkdir(exist_ok=True);sys.path[:0]=[str(Path(__file__).parent/'runtime'),'/tmp/packed_front_20260926/runtime',str(root/'runs/SS928-NIGHT-NINE-SPEED-20260926/runtime'),'/tmp'];from build_system import load_case,case_inputs
special=a.scene=='special';work='ss928-quality-special-night-nine-frame-20260925' if special else 'ss928-quality-night-nine-frame-20260925';run='SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1' if special else 'SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1';sys.path.insert(0,str(root/'code/worktrees'/work/'src'));from ir_sr.training import dataset_for_config
state=torch.load(root/'runs'/run/'checkpoints/step_000002000.pt',map_location='cpu',weights_only=False);ds=dataset_for_config(state['config'],'test');rec=next(r for r in ds.records if r['scene_id']=='night_'+a.scene);lease=(root/'runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
variant='balanced_low_rows16_nhwc16_half';loaded={k:load_case(root,a.scene,k) for k in ('base',variant,'balanced_rows16_nhwc16_half')};count=120 if special else 60;video=out/f'{a.scene}_full_comparison.mp4';writer=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','5120x1080','-r','12','-i','-','-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(video)],stdin=subprocess.PIPE);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',27);diff=[]
try:
 with torch.inference_mode():
  for frame in range(count):
   row=dict(rec,frame_id=frame);x=ds.normalized_stack(row,(0,0,1024,1280))[None].cuda();ctx,_=ds.context_for(row,crop_tlhw=(0,0,1024,1280));ctx=ctx[None].cuda();full={};preds={}
   for name,(model,info) in loaded.items():
    xx,cc=case_inputs(x,ctx,info);value=model(xx,cc);assert tuple(value.shape)==(1,1,3072,3840);full[name]=value;preds[name]=F.avg_pool2d(value.float().clamp(0,255).round(),3,3)[0,0].cpu().numpy()
   assert torch.equal(full[variant],full['balanced_rows16_nhwc16_half']),'Packed scaling altered FP16 output'
   gt=np.asarray(Image.open((Path(state['config']['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')),dtype=np.float32);norm=ds.normalization_for(row);raw=(ds.base._raw(row).astype(np.float32)-norm['offset'])/norm['scale']*255;canvas=Image.new('RGB',(5120,1080),'#17191c');draw=ImageDraw.Draw(canvas)
   for j,(label,array) in enumerate(zip(('输入 RAW','GT','v0.6 部署对应源码','九帧统计前端＋完整行分组输出'),(raw,gt,preds['base'],preds[variant]))):
    draw.text((j*1280+18,12),f'{label}  第{frame:03d}帧',font=font,fill='white');canvas.paste(Image.fromarray(np.rint(np.clip(array,0,255)).astype(np.uint8)).convert('RGB'),(j*1280,56))
   writer.stdin.write(np.asarray(canvas).tobytes());diff.append(float(np.abs(preds[variant]-preds['base']).mean()))
   if frame in (0,count//2,count-1):canvas.save(out/f'{a.scene}_frame_{frame:03d}.jpg',quality=95)
   if frame==count//2:
    for name in ('base',variant):Image.fromarray(full[name][0,0].float().clamp(0,255).round().byte().cpu().numpy()).save(out/f'{a.scene}_{name}_native3x_frame_{frame:03d}.png')
   if frame%10==0:print('VIDEO',a.scene,frame,flush=True)
   del full,x,ctx
 finally_status=True
finally:
 writer.stdin.close();code=writer.wait();assert code==0
(out/f'{a.scene}_video.json').write_text(json.dumps({'scene':a.scene,'variant':variant,'count':count,'fps':12,'video_shape':[1080,5120],'each_panel_sensor_shape':[1024,1280],'model_output_shape':[1,1,3072,3840],'video_downsample':'rounded full 3x output averaged in 3x3 cells; no crop','packed_half_scale_bitexact_to_output_half_all_frames':True,'source_execution':True,'NPU_verified':False,'mean_full_rawgrid_difference_gray':float(np.mean(diff))},indent=2));print('FINISHED',flush=True)
