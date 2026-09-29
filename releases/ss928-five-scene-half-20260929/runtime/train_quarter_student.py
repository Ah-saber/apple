"""Train a quarter-grid nine-frame student for the five earlier scene classes.

The compact structure follows the independently measured night path. Training
uses train-split GT and a frozen current-model teacher. Checkpoint selection
uses val only. Test is deliberately absent from this script.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


class BoxQuarter(torch.nn.Module):
    def __init__(self, quarter, current_only=False):
        super().__init__()
        self.quarter = quarter
        self.current_only = current_only

    def native(self, stack, context, box):
        if self.current_only:
            stack = stack[:, -1:].expand_as(stack)
        q = self.quarter
        features = q.body(q.front(stack))
        reference = q.reference(context.to(dtype=q.front.first.weight.dtype,
                                            memory_format=torch.channels_last),
                                box, features.shape[-2:]).to(features.dtype)
        phases = q.output.conv(q.tail(features + reference))
        if getattr(q,'raw_skip',False):
            phases=phases+F.pixel_unshuffle(stack[:,-1:],2)
        return F.pixel_shuffle(phases, getattr(q,'factor',4))

    def forward(self, stack, context, box):
        return F.interpolate(self.native(stack, context, box).clamp(0, 1)*255,
                             scale_factor=3, mode='nearest')


def box_for(row, y=0, x=0, h=1024, w=1280):
    from ir_sr.sequence_normalization import allowed_region
    top, left, height, width = allowed_region(row)
    return torch.tensor([[-(top-y)/height, -(left-x)/width,
                          (y+h-top)/height, (x+w-left)/width]],
                        dtype=torch.float32)


def target_path(config, row, frame):
    return (Path(config['data_root']) / row['target']['path']).with_name(
        f'{frame:06d}.png')


def cache_data(data, teacher, config, scenes, out, seed, rows_per_scene, crops):
    rng = np.random.default_rng(seed)
    entries = []
    metadata = []
    for scene in scenes:
        rows = [r for r in data.records if r['scene_id'] == scene]
        selected = [rows[int(i)] for i in np.linspace(
            0, len(rows)-1, min(rows_per_scene, len(rows)), dtype=int)]
        for row in selected:
            top,left,height,width=row['train_roi_tlhw']
            for _ in range(crops):
                cy=int(rng.integers((top+3)//4, (top+height-128)//4+1))*4
                cx=int(rng.integers((left+3)//4, (left+width-128)//4+1))*4
                pair=[]
                for frame in (max(0,row['frame_id']-1),row['frame_id']):
                    rr=dict(row,frame_id=frame)
                    stack=data.normalized_stack(rr,(cy,cx,128,128))[None]
                    context,_=data.context_for(rr)
                    gt=np.asarray(Image.open(target_path(config,rr,frame)),
                                  dtype=np.float32)[cy:cy+128,cx:cx+128]/255.
                    if teacher is not None:
                        with torch.inference_mode():
                            x=stack.cuda()
                            c=context[None].cuda()
                            b=box_for(rr,cy,cx,128,128).cuda()
                            prediction=teacher(x,c,b).float().clamp(0,1)
                            sensor=F.avg_pool2d(prediction,3,3).cpu().half()[0]
                    else:
                        sensor=torch.from_numpy(gt[None]).half()
                    pair.append((stack[0].half(),context.half(),
                                 box_for(rr,cy,cx,128,128)[0],
                                 torch.from_numpy(gt[None]).half(),sensor))
                entries.append(pair)
                metadata.append({'scene':scene,'sample_id':row['sample_id'],
                                 'frames':[max(0,row['frame_id']-1),row['frame_id']],
                                 'crop':[cy,cx,128,128]})
            print('CACHE',scene,row['frame_id'],len(entries),flush=True)
    values={key:torch.stack([torch.stack([pair[j][index]
                        for j in range(2)]) for pair in entries])
            for key,index in [('stack',0),('context',1),('box',2),
                              ('gt',3),('teacher',4)]}
    torch.save({'values':values,'metadata':metadata},out/'train_cache.pt')
    return values,metadata


def validation(model, data, config, scenes):
    scores={scene:[] for scene in scenes}
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for row in data.records:
            scene=row['scene_id']
            if scene not in scores:
                continue
            stack=data.normalized_stack(row,(0,0,1024,1280))[None].cuda()
            context,_=data.context_for(row)
            output=model.native(stack,context[None].cuda(),box_for(row).cuda())
            gt=np.asarray(Image.open(Path(config['data_root'])/
                                     row['target']['path']),dtype=np.float32)/255.
            y,x,h,w=row['eval_crop_tlhw']
            y,x,h,w=y+3,x+3,h-6,w-6
            delta=output.float()[0,0,y:y+h,x:x+w].clamp(0,1)-torch.from_numpy(
                gt[y:y+h,x:x+w]).cuda()
            mse=float(delta.square().mean())
            scores[scene].append(float(10*np.log10(1/max(mse,1e-12))))
    return {key:float(np.mean(value)) for key,value in scores.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--night-runtime',type=Path,required=True)
    p.add_argument('--fast-runtime',type=Path,required=True)
    p.add_argument('--teacher-checkpoint',type=Path)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--config-checkpoint',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--rows-per-scene',type=int,default=32)
    p.add_argument('--crops',type=int,default=4)
    p.add_argument('--seed',type=int,default=929)
    p.add_argument('--depth',type=int,choices=(1,2,3),default=1)
    p.add_argument('--front-kind',choices=('k4','3x3'),default='k4')
    p.add_argument('--grid',choices=('quarter','half'),default='quarter')
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--resume-student',type=Path)
    p.add_argument('--train-cache',type=Path)
    p.add_argument('--current-only',action='store_true')
    p.add_argument('--raw-skip',action='store_true')
    p.add_argument('--teacher-weight',type=float,default=.35)
    p.add_argument('--motion-weight',type=float,default=0.)
    p.add_argument('--weak-weight',type=float,default=.3)
    p.add_argument('--pair-weight',type=float,default=.35)
    p.add_argument('--static-weight',type=float,default=.3)
    p.add_argument('--save-every-validation',action='store_true')
    p.add_argument('--lr',type=float,default=2e-4)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(a.code/'src'),str(a.night_runtime),
                  str(a.fast_runtime),str(Path(__file__).parent)]
    from ir_sr.model import inference_model
    from ir_sr.training import dataset_for_config
    from quarter_candidates import QuarterSystem
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    state=torch.load(a.config_checkpoint,map_location='cpu',weights_only=False)
    config=state['config']
    train=dataset_for_config(config,'train')
    val=dataset_for_config(config,'val')
    teacher=None
    if a.teacher_checkpoint is not None:
        from fast_nine_inference import build_fast_model
        ts=torch.load(a.teacher_checkpoint,map_location='cpu',weights_only=False)
        teacher=build_fast_model(ts['config'],ts['model'],inference_model).cuda().eval()
        teacher.requires_grad_(False)
        reference=copy.deepcopy(teacher.model.global_reference).cpu().float()
    elif a.reference_checkpoint is not None:
        rs=torch.load(a.reference_checkpoint,map_location='cpu',weights_only=False)
        reference=copy.deepcopy(inference_model(
            rs['config'],rs['model']).global_reference).cpu().float()
    else:
        from ir_sr.model import RT4KSRB0
        source=RT4KSRB0(channels=16,blocks=1,global_reference=True,
                        reference_pyramid=True)
        reference=copy.deepcopy(source.global_reference)
    cache=a.train_cache if a.train_cache else a.out/'train_cache.pt'
    if cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=False)
        data,metadata=saved['values'],saved['metadata']
    else:
        data,metadata=cache_data(train,teacher,config,a.scenes,a.out,
                                 a.seed,a.rows_per_scene,a.crops)
    del teacher
    torch.cuda.empty_cache()
    if a.grid=='half':
        from half_student import HalfSystem
        quarter=HalfSystem(reference,depth=a.depth,width=a.width,
                           raw_skip=a.raw_skip)
    else:
        quarter=QuarterSystem(reference,depth=a.depth,mode='pixel',
                              front_kind=a.front_kind,width=a.width)
    model=BoxQuarter(quarter,current_only=a.current_only).cuda().to(memory_format=torch.channels_last)
    if a.resume_student is not None:
        resumed=torch.load(a.resume_student,map_location='cpu',weights_only=False)
        missing,unexpected=model.load_state_dict(resumed['model'],strict=False)
        if unexpected or any(not key.startswith('quarter.body.') for key in missing):
            raise ValueError(f'Unexpected resume keys: {missing}, {unexpected}')
        print('RESUME',str(a.resume_student),'new_layers',missing,flush=True)
    # The trained reference encoder remains a fixed input transform. Its new
    # 12-to-32 projection and every student image layer learn from train data.
    model.quarter.reference.encoder.requires_grad_(False)
    if hasattr(model.quarter.reference,'pyramid'):
        model.quarter.reference.pyramid.requires_grad_(False)
    params=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,a.steps,eta_min=2e-6)
    rng=np.random.default_rng(a.seed)
    best=-float('inf')
    reports=[]
    started=time.perf_counter()
    for step in range(1,a.steps+1):
        ids=rng.integers(0,len(metadata),4)
        x=data['stack'][ids].flatten(0,1).cuda().float()
        c=data['context'][ids].flatten(0,1).cuda().float()
        b=data['box'][ids].flatten(0,1).cuda().float()
        gt=data['gt'][ids].flatten(0,1).cuda().float()
        target=data['teacher'][ids].flatten(0,1).cuda().float()
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.native(x,c,b).float().clamp(0,1)
            err_gt=(pred-gt)*255
            err_teacher=(pred-target)*255
            edge=slice(12,-12)
            pixel=err_gt[:,:,edge,edge].abs().mean()
            imitation=err_teacher[:,:,edge,edge].abs().mean()
            pair_error=err_gt.reshape(4,2,1,128,128)
            temporal=(pair_error[:,1]-pair_error[:,0])[:,:,edge,edge].abs().mean()
            delta=(gt[1::2]-gt[::2]).abs()[:,:,edge,edge]*255
            weak=(delta>=1)&(delta<=3)
            weak_error=((pair_error[:,1]-pair_error[:,0])[:,:,edge,edge].abs()*weak).sum()/weak.sum().clamp_min(1)
            strong=(delta>=3)
            motion_error=((pair_error[:,1]-pair_error[:,0])[:,:,edge,edge].abs()*strong).sum()/strong.sum().clamp_min(1)
            local=F.avg_pool2d(err_gt[:,:,edge,edge],8,8).abs().mean()
            loss=(pixel+a.teacher_weight*imitation+
                  a.pair_weight*temporal+a.weak_weight*weak_error+
                  a.motion_weight*motion_error+a.static_weight*local)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,5)
        optimizer.step()
        scheduler.step()
        if step==1 or step%1000==0:
            row={'step':step,'loss':float(loss.detach()),
                 'train_pixel_gray':float(pixel.detach()),
                 'train_weak_temporal_gray':float(weak_error.detach()),
                 'train_motion_temporal_gray':float(motion_error.detach()),
                 'elapsed_seconds':time.perf_counter()-started}
            if step%2000==0 or step==a.steps:
                row['val_psnr_db']=validation(model,val,config,a.scenes)
                row['val_macro_psnr_db']=float(np.mean(list(row['val_psnr_db'].values())))
                if a.save_every_validation:
                    torch.save({'format':'box_quarter_student_v1',
                        'step':step,'scenes':a.scenes,'seed':a.seed,
                        'depth':a.depth,'front_kind':a.front_kind,
                        'grid':a.grid,'width':a.width,
                        'raw_skip':a.raw_skip,
                        'current_only':a.current_only,
                        'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'val_psnr_db':row['val_psnr_db'],'GT_used':True,
                        'test_used_for_selection':False},
                        a.out/f'step_{step:06d}.pt')
                if row['val_macro_psnr_db']>best:
                    best=row['val_macro_psnr_db']
                    torch.save({'format':'box_quarter_student_v1',
                        'step':step,'scenes':a.scenes,'seed':a.seed,
                        'depth':a.depth,'front_kind':a.front_kind,
                        'grid':a.grid,'width':a.width,
                        'raw_skip':a.raw_skip,
                        'current_only':a.current_only,
                        'resume_student':str(a.resume_student),
                        'teacher_weight':a.teacher_weight,
                        'motion_weight':a.motion_weight,
                        'weak_weight':a.weak_weight,
                        'pair_weight':a.pair_weight,
                        'static_weight':a.static_weight,
                        'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'val_psnr_db':row['val_psnr_db'],'GT_used':True,
                        'test_used_for_selection':False,
                        'reference_source':str(a.teacher_checkpoint)},
                        a.out/'best.pt')
            reports.append(row)
            print('TRAIN',json.dumps(row),flush=True)
            (a.out/'training.json').write_text(json.dumps({
                'scenes':a.scenes,'seed':a.seed,'steps':a.steps,
                'depth':a.depth,'front_kind':a.front_kind,
                'grid':a.grid,'width':a.width,
                'raw_skip':a.raw_skip,
                'current_only':a.current_only,
                'resume_student':str(a.resume_student),
                'teacher_weight':a.teacher_weight,'lr':a.lr,
                'motion_weight':a.motion_weight,
                'weak_weight':a.weak_weight,
                'pair_weight':a.pair_weight,
                'static_weight':a.static_weight,
                'train_samples':metadata,'logs':reports,
                'best_macro_psnr_db':best},indent=2))
    torch.save({'format':'box_quarter_student_v1','step':a.steps,
                'model':{k:v.detach().cpu() for k,v in model.state_dict().items()}},
               a.out/'last.pt')


if __name__=='__main__':main()
