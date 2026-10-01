"""Night causal front blending, calibrated only on real training RAW inputs."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
os.environ['RAWIR_V09_CODE']=str(ROOT/'runs/SS928-BOARD-V09-20260928/code_snapshot')
sys.path[:0]=[str(ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'),str(ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot'),
             str(ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925/src'),str(V13/'runtime')]
from reference_layout import load_candidate as refload
from lowres_reference import load_candidate as lowload
from output_candidates import prepare_inputs
from ir_sr.training import dataset_for_config
from run_compact import sha,temporal_metrics
from finalize_candidates import benchmark
from adaptive_history import HistorySignal,GatedNightFront
from speed_variants import NightByte
from prepare_deployment import ONNX_SITE
sys.path.insert(0,str(ONNX_SITE))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--strength',type=float,default=1.);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(930);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'trained_new_weights':False,'calibration_uses_GT':False,'packaged':False,'pushed':False,'scenes':{}}
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',22)
    for scene in ['ordinary','special']:
        name='night_'+scene;out=a.out/name;out.mkdir()
        source=refload(scene,'after12','rows32',ROOT/'runs/SS928-BOARD-V09-20260928') if scene=='ordinary' else lowload(scene,5,'rows32',ROOT/'runs/SS928-BOARD-V09-20260928',trained=True)
        source=source.eval();directory=V13/'night_calibration_final'/name;manifest=json.loads((directory/'manifest.json').read_text())
        signals=[]
        for path in sorted(directory.glob('train_*.npz')):
            x=torch.from_numpy(np.load(path)['nine_raw']).cuda();s=HistorySignal(1,1).cuda().signal(x).float().cpu().numpy()
            signals.append(s[:,:,3:-3,3:-3].reshape(-1))
        raw_signal=np.concatenate(signals);thresholds={str(p):float(np.percentile(raw_signal,p)) for p in [80,90,98]}
        candidates={p:copy.deepcopy(source) for p in thresholds}
        for p,m in candidates.items():m.front=GatedNightFront(source.front,thresholds[p],strength=a.strength).cuda().eval()
        zero=copy.deepcopy(source);zero.front=GatedNightFront(source.front,1,force=0).cuda().eval()
        sample=np.load(directory/'test_00.npz');args=tuple(torch.from_numpy(sample[k]).cuda() for k in ['nine_raw','reference_thumb'])
        with torch.inference_mode():assert torch.equal(zero(*args),source(*args)),name
        val=dataset_for_config(manifest['config'],'val');rows=[r for r in val.records if r['scene_id']==name][:12];assert rows
        validation={p:[] for p in ['source']+list(candidates)}
        with torch.inference_mode():
            for row in rows:
                stack=val.normalized_stack(row,(0,0,1024,1280))[None].cuda();c,_=val.context_for(row,crop_tlhw=(0,0,1024,1280));args=prepare_inputs(source,stack,c[None].cuda())
                gt=np.array(Image.open(Path(manifest['config']['data_root'])/row['target']['path']),dtype=np.float32)
                y,x,h,w=row['eval_crop_tlhw'];y+=3;x+=3;h-=6;w-=6
                for p,m in {'source':source,**candidates}.items():
                    pred=m(*args)[0,0,::3,::3].float().cpu().numpy();err=pred[y:y+h,x:x+w]-gt[y:y+h,x:x+w]
                    validation[p].append(float(-10*np.log10(max(float((err**2).mean())/255**2,1e-12))))
        means={p:float(np.mean(v)) for p,v in validation.items()};eligible=[p for p in thresholds if means[p]>=means['source']-.15]
        chosen=min(eligible,key=int) if eligible else '98';candidate=candidates[chosen]
        data=dataset_for_config(manifest['config'],'test');row=next(r for r in data.records if r['scene_id']==name)
        from ir_sr.sequence_normalization import regional_key
        frame_count=int(data.base.sequence_normalization.by_key[regional_key(row)]['frames'])
        labels=['GT','原夜间九帧','局部当前夜间'];video=out/f'full{frame_count}_history_intervention.mp4'
        writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','3840x1080','-r','12','-i','-',
            '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        metrics=[];roads={label:[] for label in labels};previous=None
        roi_y=0 if scene=='ordinary' else 680;roi_h=720 if scene=='ordinary' else 344
        polygon=([[0,80],[80,208],[175,336],[270,387],[600,452],[900,526],[1279,657],[1279,710],[880,570],[550,480],[260,416],[155,363],[70,250],[0,120]]
                 if scene=='ordinary' else [[0,70],[1279,282],[1279,344],[0,132]])
        with torch.inference_mode():
            try:
                for frame in range(frame_count):
                    rr=dict(row,frame_id=frame);stack=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda();c,_=data.context_for(rr,crop_tlhw=(0,0,1024,1280));args=prepare_inputs(source,stack,c[None].cuda())
                    gt=np.array(Image.open((Path(manifest['config']['data_root'])/row['target']['path']).with_name(f'{frame:06d}.png')),dtype=np.float32)
                    images=[gt]+[m(*args)[0,0,::3,::3].float().cpu().numpy() for m in [source,candidate]]
                    canvas=Image.new('RGB',(3840,1080),'#161b22');draw=ImageDraw.Draw(canvas);item={'frame':frame,'methods':{}}
                    y,x,h,w=row['eval_crop_tlhw'];y+=3;x+=3;h-=6;w-=6;g=gt[y:y+h,x:x+w]
                    for col,(label,image) in enumerate(zip(labels,images)):
                        canvas.paste(Image.fromarray(np.rint(image.clip(0,255)).astype(np.uint8)).convert('RGB'),(1280*col,56));draw.text((1280*col+12,10),f'{name} | {label} | {frame:03d}',font=font,fill='white')
                        roads[label].append(image[roi_y:roi_y+roi_h].copy())
                        if col:
                            pred=image[y:y+h,x:x+w];err=pred-g;r={'psnr_db':float(-10*np.log10(max(float((err**2).mean())/255**2,1e-12))),'mae_gray':float(np.abs(err).mean())}
                            if previous:r.update(temporal_metrics(pred,previous[col][y:y+h,x:x+w],g,previous[0][y:y+h,x:x+w]))
                            item['methods'][label]=r
                    writer.stdin.write(np.asarray(canvas).tobytes());metrics.append(item);previous=images
                    if frame%30==0:print('NIGHT_MOTION',name,frame,flush=True)
            finally:writer.stdin.close();assert writer.wait()==0
        road=np.zeros((roi_h,1280),np.uint8);cv2.fillPoly(road,[np.array(polygon)],1);road=road.astype(bool)
        yy,xx,hh,ww=row['eval_crop_tlhw'];coord_y,coord_x=np.mgrid[:roi_h,:1280]
        road&=(coord_y+roi_y>=yy+3)&(coord_y+roi_y<yy+hh-3)&(coord_x>=xx+3)&(coord_x<xx+ww-3)
        values={}
        for label,v in roads.items():
            v=np.stack(v);v=v-np.median(v,axis=0);v=v-np.median(v[:,road],axis=1)[:,None,None];values[label]=v
        np.savez_compressed(out/'road_signals_and_masks.npz',**values)
        (out/'target_region.json').write_text(json.dumps({'roi_y':roi_y,'roi_x':0,'polygon':polygon,'valid_native_tlhw':row['eval_crop_tlhw'],'metric_border_pixels':3,
            'selection':'bridge polygon inspected on same-scene GT; intersect actual label ROI; special only right192 pixels valid'}))
        summary={label:{k:float(np.mean([v['methods'][label][k] for v in metrics if k in v['methods'][label]])) for k in metrics[-1]['methods'][label]} for label in labels[1:]}
        board=NightByte(candidate).cuda().eval();timing=benchmark(board,args);timing['precision']='original mixed dtype preserved'
        import onnx
        graph=out/(name+'_adaptive_front_integer.onnx');torch.onnx.export(board,args,str(graph),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(str(graph)))
        report['scenes'][name]={'strength':a.strength,'actual_frames':frame_count,'thresholds_input_units':thresholds,'chosen_percentile':chosen,'validation_psnr':means,'passed_val_guard':bool(eligible),
            'zero_gate_exact_original':True,'summary':summary,'per_frame':metrics,'GPU':timing,'video':str(video),'video_sha256':sha(video),
            'graph':{'path':str(graph),'sha256':sha(graph),'retains_original_night_Resize':True},'calibration':[{ 'path':str(p),'sha256':sha(p)} for p in sorted(directory.glob('train_*.npz'))]}
        (a.out/'night_motion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print('NIGHT_MOTION_RESULT',name,summary,flush=True)
    print('NIGHT_MOTION_COMPLETE',flush=True)


if __name__=='__main__':main()
