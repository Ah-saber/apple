"""Compare the candidate on complete sequences; preserve masks and exact timing controls."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

PREVIOUS=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V13-20260930')
sys.path.insert(0,str(PREVIOUS/'runtime'))
from run_compact import setup,GROUPS,sha,temporal_metrics
from prepare_deployment import ONNX_SITE,export_model
from finalize_candidates import benchmark
from reliable_model import ReliableModel
from native_reliable_model import NativeReliableModel
from board_reliable import BoardReliable
from integer_output import IntegerOutput
from probe_integer_output import ByteAfterRows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--native-gate',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ONNX_SITE))
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
    report={'NPU_measured':False,'packaged':False,'pushed':False,'scenes':{}}
    for group in ['light','heavy']:
        prior=torch.load(PREVIOUS/(group+'_temporal')/'best.pt',map_location='cpu',weights_only=False)
        GROUPS[group].update(prior['spec'])
        compact,base,config,teacher,base_path,df,box_for=setup(group)
        compact.load_state_dict(prior['model']);compact=compact.cuda().float().eval();base=base.cuda().float().eval()
        run=a.root/(('native_reliable_' if a.native_gate else 'reliable16_')+group)
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        cls=NativeReliableModel if a.native_gate else ReliableModel
        model=cls(base.quarter.reference,width=ck['width'],threshold_gray=ck['threshold_gray']).cuda().float().eval()
        model.load_state_dict(ck['model']);data=df(config,'test')
        scenes=['weather_light','weather_medium'] if group=='light' else ['weather_heavy']
        for scene in scenes:
            out=a.out/scene;out.mkdir();row=next(r for r in data.records if r['scene_id']==scene)
            box=box_for(row).cuda();board=BoardReliable(model,box).cuda().float().eval()
            names=['GT','原半网格九帧','四分之一九帧','原网格历史约束' if a.native_gate else '当前细节及历史约束']
            video=out/'full120_history_intervention.mp4'
            writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','5120x1080','-r','12','-i','-',
                '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
            roads=[];metrics=[];previous=None;control_max=0.;history_feature_max=0.
            with torch.inference_mode():
                try:
                    for frame in range(120):
                        rr=dict(row,frame_id=frame)
                        x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda();c,_=data.context_for(rr);c=c[None].cuda()
                        gt=np.array(Image.open((Path(config['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')),dtype=np.float32)
                        source=model.native(x,c,box).clamp(0,1)*255
                        if frame in [8,60,119]:
                            frozen=board(x,c)[:,:,::3,::3];control_max=max(control_max,float((frozen-source).abs().max()))
                        _,_,history,reliability=model.components(x)
                        history_feature_max=max(history_feature_max,float(history.abs().max())*255)
                        images=[gt]+[(m.native(x,c,box).clamp(0,1)[0,0]*255).cpu().numpy() for m in [base,compact]]+[source[0,0].cpu().numpy()]
                        canvas=Image.new('RGB',(5120,1080),'#161b22');draw=ImageDraw.Draw(canvas)
                        for i,(name,image) in enumerate(zip(names,images)):
                            canvas.paste(Image.fromarray(np.rint(image.clip(0,255)).astype(np.uint8)).convert('RGB'),(1280*i,56))
                            draw.text((1280*i+12,10),f'{scene} | {name} | {frame:03d}',font=font,fill='white')
                        writer.stdin.write(np.asarray(canvas).tobytes());roads.append(images[-1][240:600].copy())
                        y,x0,h,w=rr['eval_crop_tlhw'];y+=3;x0+=3;h-=6;w-=6
                        g=gt[y:y+h,x0:x0+w];p=images[-1][y:y+h,x0:x0+w]
                        err=p-g;item={'frame':frame,'new_mae_gray':float(np.abs(err).mean()),'new_bias_gray':float(err.mean()),
                            'new_contrast_error_gray':float(p.std()-g.std()),'new_psnr_db':float(-10*np.log10(max(float((err**2).mean())/255**2,1e-12)))}
                        if previous:item.update({'new_'+k:v for k,v in temporal_metrics(p,previous[0],g,previous[1]).items()})
                        metrics.append(item);previous=(p.copy(),g.copy())
                        if frame%30==0:print('RELIABLE_VIDEO',scene,frame,flush=True)
                finally:writer.stdin.close();assert writer.wait()==0
            assert control_max<.01,control_max
            if not a.native_gate:assert history_feature_max<=.5001,history_feature_max
            cached_root=a.root/('history_probe' if group=='light' else 'history_heavy')/scene
            cached=np.load(cached_root/'road_signals_and_masks.npz',allow_pickle=False)
            road=np.zeros((360,1280),np.uint8);cv2.fillPoly(road,[np.array([[0,15],[1279,245],[1279,330],[0,100]])],1);road=road.astype(bool)
            new=np.stack(roads);new=new-np.median(new,axis=0)
            drift=np.median(new[:,road],axis=1);new=new-drift[:,None,None]
            np.savez_compressed(out/'road_signals_and_masks.npz',**{k:cached[k] for k in names[:3]},**{names[-1]:new})
            summary={k:float(np.mean([v[k] for v in metrics if k in v])) for k in metrics[-1] if k!='frame'}
            record={'checkpoint':str(run/'best.pt'),'checkpoint_sha256':sha(run/'best.pt'),
                'source_to_frozen_max_gray':control_max,'history_feature_max_gray':history_feature_max,
                'history_feature_bound_gray':None if a.native_gate else .5,'bound_does_not_apply_to_final_output':True,
                'full_quality_summary':summary,'per_frame':metrics,'video':str(video),'video_sha256':sha(video),'graphs':[]}
            if scene in ['weather_light','weather_heavy']:
                xx=x.half();cc=c.half();float_board=copy.deepcopy(board).half().eval();integer=IntegerOutput(float_board,4).cuda().eval()
                with torch.inference_mode():assert torch.equal(integer(xx,cc),float_board(xx,cc).byte())
                for name,m in [('float_rows32',float_board),('byte_after_rows32',ByteAfterRows(copy.deepcopy(float_board))),('integer_byte',integer)]:
                    record[name+'_GPU']=benchmark(m,(xx,cc));torch._dynamo.reset();torch.cuda.empty_cache()
                    print('RELIABLE_GPU',scene,name,record[name+'_GPU']['mean_ms'],flush=True)
                record['graphs'].append(export_model(integer,(xx,cc),out/f'{scene}_reliable16_integer_byte_fp16.onnx',['nine_raw','reference_thumb'],'display_gray'))
            report['scenes'][scene]=record
            (a.out/'evaluation.json').write_text(json.dumps(report,indent=2))
            print('RELIABLE_FULL_RESULT',scene,json.dumps(summary),flush=True)
    print('RELIABLE_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
