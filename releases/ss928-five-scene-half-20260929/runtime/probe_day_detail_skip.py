"""Probe cheap current-frame detail paths on the held-out day validation clip."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--night-runtime',type=Path,required=True)
    p.add_argument('--config-checkpoint',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path,required=True)
    p.add_argument('--student-checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=int,default=30)
    a=p.parse_args()
    sys.path[:0]=[str(a.code/'src'),str(a.night_runtime),str(Path(__file__).parent)]
    from ir_sr.model import inference_model
    from ir_sr.training import dataset_for_config
    from quarter_candidates import QuarterSystem
    from train_quarter_student import BoxQuarter,box_for
    config=torch.load(a.config_checkpoint,map_location='cpu',weights_only=False)['config']
    rs=torch.load(a.reference_checkpoint,map_location='cpu',weights_only=False)
    reference=copy.deepcopy(inference_model(rs['config'],rs['model']).global_reference)
    ck=torch.load(a.student_checkpoint,map_location='cpu',weights_only=False)
    model=BoxQuarter(QuarterSystem(reference,depth=ck.get('depth',1),
                     mode='pixel',front_kind=ck.get('front_kind','k4'),width=32))
    model.load_state_dict(ck['model'],strict=True)
    model=model.cuda().eval()
    data=dataset_for_config(config,'val')
    record=next(r for r in data.records if r['scene_id']=='day_normal')
    alphas=(0.,.25,.5,1.)
    keys=[f'{kind}_{alpha:g}' for kind in ('spatial5','temporal3','temporal5')
          for alpha in alphas]
    scores={key:[] for key in keys}
    previous={}
    previous_gt=None
    for frame in range(a.frames):
        row=dict(record,frame_id=frame)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            stack=data.normalized_stack(row,(0,0,1024,1280))[None].cuda()
            context,_=data.context_for(row)
            base=model.native(stack,context[None].cuda(),box_for(row).cuda()).float()*255
            current=stack[:,-1:].float()*255
            reference=stack[:,:6].float().mean(1,keepdim=True)*255
            spatial=current-F.avg_pool2d(current,5,stride=1,padding=2)
            temporal3=F.avg_pool2d(current-reference,3,stride=1,padding=1)
            temporal5=F.avg_pool2d(current-reference,5,stride=1,padding=2)
        components={'spatial5':spatial,'temporal3':temporal3,
                    'temporal5':temporal5}
        target=np.asarray(Image.open((Path(config['data_root'])/
            row['target']['path']).with_name(f'{frame:06d}.png')),
            dtype=np.float32)
        y,x,h,w=row['eval_crop_tlhw']
        y,x,h,w=y+3,x+3,h-6,w-6
        gt=target[y:y+h,x:x+w]
        gradient=np.maximum.reduce([np.abs(gt-np.roll(gt,k,axis=axis))
            for axis in (0,1) for k in (-1,1)])
        for key in keys:
            kind,alpha=key.rsplit('_',1)
            pred=(base+float(alpha)*components[kind]).clamp(0,255)
            roi=pred[0,0,y:y+h,x:x+w].cpu().numpy()
            error=roi-gt
            item={'mae':float(np.abs(error).mean()),
                  'psnr':float(10*np.log10(255**2/max(np.mean(error**2),1e-12)))}
            if previous_gt is not None:
                gt_change=gt-previous_gt
                pred_change=roi-previous[key]
                static=(np.abs(gt_change)<=1)&(gradient<=5)
                moving=np.abs(gt_change)>=3
                weak=(np.abs(gt_change)>=1)&(np.abs(gt_change)<3)&(gradient>=5)
                item['static_residual_change']=float(np.abs(pred_change-gt_change)[static].mean())
                item['motion_response']=float((np.sign(gt_change[moving])*pred_change[moving]).mean()/
                                              np.abs(gt_change[moving]).mean())
                item['weak_error']=float(np.abs(pred_change-gt_change)[weak].mean())
            scores[key].append(item)
            previous[key]=roi
        previous_gt=gt
        if frame%10==0:print('FRAME',frame,flush=True)
    summary={key:{metric:float(np.mean([v[metric] for v in values if metric in v]))
                  for metric in ('mae','psnr','static_residual_change',
                                 'motion_response','weak_error')}
             for key,values in scores.items()}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'report.json').write_text(json.dumps({'checkpoint':str(a.student_checkpoint),
        'split':'val','frames':a.frames,'summary':summary},indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
