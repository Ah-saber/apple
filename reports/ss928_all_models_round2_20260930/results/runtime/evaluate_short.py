"""Full paired sequence evaluation; GT masks stay solely in offline analysis."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930';MOTION=ROOT/'runs/SS928-MOTION-SPEED-20260930'
sys.path[:0]=[str(V13/'runtime'),str(MOTION/'runtime')]
from run_compact import setup,GROUPS,sha,temporal_metrics
from compact_model import BoardCompact
from equivalent_model import BoardHalfOptimized
from short_history import ShortHistory
from speed_variants import RepeatByteReorder
from integer_output import IntegerOutput
from finalize_candidates import benchmark
from prepare_deployment import export_model,ONNX_SITE
from history_probe import foreground
from baseline_contract import load_group,independent_control


class BoardShort(nn.Module):
    def __init__(self,model,box):
        super().__init__();self.count=model.count
        board=BoardCompact(model.source,box) if model.kind=='quarter' else BoardHalfOptimized(model.source,box)
        factor=4 if model.kind=='quarter' else 2
        self.model=IntegerOutput(board,factor);self.model.output=RepeatByteReorder(factor)

    def forward(self,x,c):return self.model(x[:,-self.count:],c)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--group',choices=GROUPS,required=True)
    p.add_argument('--prefix',default='short');p.add_argument('--counts',nargs='+',type=int,default=[3,1]);p.add_argument('--kind',choices=['half','quarter']);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ONNX_SITE));torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    compact,base,config,_,basepath,df,box_for=load_group(a.group);kind=a.kind or ('half' if a.group=='day' else 'quarter')
    baseline_control=independent_control(a.group,compact)
    if kind=='quarter':compact.load_state_dict(torch.load(V13/(a.group+'_temporal')/'best.pt',map_location='cpu',weights_only=False)['model'])
    source=base if kind=='half' else compact;shorts=[]
    for count in a.counts:
        path=a.root/f'{a.prefix}_{a.group}_{count}/best.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        model=ShortHistory(source,count,kind);model.load_state_dict(ck['model']);shorts.append(model.cuda().float().eval())
    base=base.cuda().float().eval();compact=compact.cuda().float().eval()
    labels={1:'当前帧训练',3:'最近三帧训练',9:'九帧目标约束训练'}
    names=['GT','原半网格九帧']+([] if kind=='half' else ['原四分之一九帧'])+[labels[count] for count in a.counts]
    models=[base]+([] if kind=='half' else [compact])+shorts
    data=df(config,'test');report={'NPU_measured':False,'packaged':False,'pushed':False,'scenes':{}}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
    for scene in GROUPS[a.group]['scenes']:
        directory=a.out/scene;directory.mkdir();row=next(r for r in data.records if r['scene_id']==scene);box=box_for(row).cuda()
        video=directory/'full120_history_intervention.mp4'
        writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{1280*len(names)}x1080','-r','12','-i','-',
            '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        roads={n:[] for n in names};per_frame=[];previous=None
        with torch.inference_mode():
            try:
                for frame in range(120):
                    rr=dict(row,frame_id=frame);x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda();c,_=data.context_for(rr);c=c[None].cuda()
                    gt=np.array(Image.open((Path(config['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')),dtype=np.float32)
                    images=[gt]+[(m.native(x,c,box).clamp(0,1)[0,0]*255).cpu().numpy() for m in models]
                    canvas=Image.new('RGB',(1280*len(names),1080),'#161b22');draw=ImageDraw.Draw(canvas)
                    y,x0,h,w=rr['eval_crop_tlhw'];y+=3;x0+=3;h-=6;w-=6;g=gt[y:y+h,x0:x0+w];r={'frame':frame,'methods':{}}
                    for col,(name,image) in enumerate(zip(names,images)):
                        canvas.paste(Image.fromarray(np.rint(image.clip(0,255)).astype(np.uint8)).convert('RGB'),(1280*col,56))
                        draw.text((1280*col+12,10),f'{scene} | {name} | {frame:03d}',font=font,fill='white');roads[name].append(image[:650].copy() if scene=='day_normal' else image[240:600].copy())
                        if col:
                            pred=image[y:y+h,x0:x0+w];err=pred-g;item={'psnr_db':float(-10*np.log10(max(float((err**2).mean())/255**2,1e-12))),
                                'mae_gray':float(np.abs(err).mean()),'bias_gray':float(err.mean())}
                            if previous:item.update(temporal_metrics(pred,previous[col][y:y+h,x0:x0+w],g,previous[0][y:y+h,x0:x0+w]))
                            r['methods'][name]=item
                    writer.stdin.write(np.asarray(canvas).tobytes());per_frame.append(r);previous=images
                    if frame%30==0:print('SHORT_EVAL',scene,frame,flush=True)
            finally:writer.stdin.close();assert writer.wait()==0
        summary={n:{k:float(np.mean([r['methods'][n][k] for r in per_frame if k in r['methods'][n]])) for k in per_frame[-1]['methods'][n]} for n in names[1:]}
        # C32 does not include this road in its valid test region; retain no target metric for it.
        if scene!='weather_heavy_c32':
            polygon=([[0,18],[110,134],[220,280],[350,345],[600,393],[930,464],[1279,575],[1279,615],[920,500],[590,429],[360,381],[195,339],[95,199],[0,75]]
                     if scene=='day_normal' else [[0,15],[1279,245],[1279,330],[0,100]])
            road=np.zeros(next(iter(roads.values()))[0].shape,np.uint8);cv2.fillPoly(road,[np.array(polygon)],1)
            signals={n:foreground(np.stack(v),road.astype(bool))[0] for n,v in roads.items()}
            np.savez_compressed(directory/'road_signals_and_masks.npz',**signals)
            (directory/'target_region.json').write_text(json.dumps({'roi_y':0 if scene=='day_normal' else 240,'roi_x':0,'polygon':polygon,
                'valid_native_tlhw':rr['eval_crop_tlhw'],'metric_border_pixels':3,
                'selection':'bridge polygon inspected on same-scene GT; exclude C32 invalid road'}))
        entry={'summary':summary,'per_frame':per_frame,'video':str(video),'video_sha256':sha(video),'models':{}}
        for m in shorts:
            frozen=BoardShort(m,box).cuda().half().eval()
            with torch.inference_mode():
                float_native=m.native(x,c,box).clamp(0,1)*255;fp16native=frozen(x.half(),c.half())[:,:,::3,::3].float()
                control={'FP16_vs_FP32_byte_mean_gray':float((fp16native-float_native.byte().float()).abs().mean()),'max_gray':float((fp16native-float_native.byte().float()).abs().max())}
            entry['models'][str(m.count)]={'GPU':benchmark(frozen,(x.half(),c.half())),'precision_control':control,
                'graph':export_model(frozen,(x.half(),c.half()),directory/f'{scene}_short{m.count}_integer.onnx',['nine_raw','reference_thumb'],'display_gray')}
            torch._dynamo.reset();torch.cuda.empty_cache()
        report['scenes'][scene]=entry;(a.out/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print('SHORT_FULL_RESULT',scene,json.dumps(summary,ensure_ascii=False),flush=True)
    print('SHORT_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
