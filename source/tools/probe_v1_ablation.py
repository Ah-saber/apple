"""Short controlled engineering probe on two actual training crops; no formal weights reused."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
import torch
from ir_sr.data import paired_patch, area_downsample3
from ir_sr.model import RT4KSRB0, inference_model
from ir_sr.auxiliary import loss_terms, ModuleTimer
from ir_sr.training import seed_all, atomic_json, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1'
    args.output.mkdir(parents=True,exist_ok=False)
    lease=(args.output.parent/'TASK-019-gpu0.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=Path('/data/zhangbenzhuang/huawei_sr/data');cache=args.output.parent/'MIDGT-TRAIN-V1-20260922'
    index=json.loads((cache/'index.json').read_text());entry=index['sequences'][0]
    assert entry['roi_tlhw']==[0,0,1024,1280]
    assert sha(cache/entry['path'])==entry['sha256']
    records=[json.loads(s) for s in (root/'manifests/dataset_d1/pairs.jsonl').read_text().splitlines()]
    norm=json.loads((root/'recipes/preprocess_v2.json').read_text())['raw_normalization']
    raw=np.load(root/entry['source_raw_path'],mmap_mode='r');middle=np.load(cache/entry['path'],mmap_mode='r')
    inputs=[];targets=[];middles=[];selected=[]
    for k in (0,len(raw)//2):
        row=next(r for r in records if r['sample_id'] in entry['sample_ids'] and r['frame_id']==k)
        with Image.open(root/row['target']['path']) as im:gt=np.asarray(im)
        x,y=paired_patch(raw[k],gt,120,300,768,768,norm['offset'],norm['scale'])
        z=((area_downsample3(middle[k,120:888,300:1068])-norm['offset'])/norm['scale'])[None]
        inputs.append(torch.from_numpy(x));targets.append(torch.from_numpy(y));middles.append(torch.from_numpy(z))
        selected.append(row['sample_id'])
    x,y,z=[torch.stack(v).cuda() for v in (inputs,targets,middles)]
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    all_results=[];begin=time.monotonic();initial=None
    for augmentation,auxiliary in ((False,False),(True,False),(False,True),(True,True)):
        seed_all(928);model=RT4KSRB0(auxiliary_raw=auxiliary).cuda();opt=torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.99))
        timer=ModuleTimer(model);times=[];losses=[];module_times=[]
        with torch.no_grad():prediction=model(x)
        if initial is None:initial=prediction.detach().clone()
        else:torch.testing.assert_close(prediction,initial,rtol=0,atol=0)
        for step in range(100):
            op=step%8 if augmentation else 0
            def transform(a):
                a=torch.rot90(a,op%4,(-2,-1))
                return a.flip(-1).contiguous() if op//4 else a.contiguous()
            xx,yy,zz=map(transform,(x,y,z))
            t=time.monotonic();opt.zero_grad(set_to_none=True);timer.start_step(step in (0,50,99))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,display,aux=loss_terms(model,xx,yy,zz if auxiliary else None,.1 if auxiliary else 0)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            measured=timer.finish()
            if measured is not None:module_times.append({'step':step+1,'module_ms':measured})
            losses.append({'step':step+1,'total':float(loss),'display':float(display),'raw':float(aux),'operation':op})
            times.append(time.monotonic()-t)
        assert np.mean([r['total'] for r in losses[-10:]]) < np.mean([r['total'] for r in losses[:10]])
        gradient=None
        if auxiliary:
            opt.zero_grad(set_to_none=True)
            loss,display,aux=loss_terms(model,x,y,z,.1)
            main=torch.autograd.grad(display,model.head[0].weight,retain_graph=True)[0]
            rawgrad=torch.autograd.grad(.1*aux,model.head[0].weight)[0]
            assert torch.isfinite(rawgrad).all() and rawgrad.abs().sum()>0
            gradient={'display_head_gradient_norm':float(main.norm()),'weighted_raw_head_gradient_norm':float(rawgrad.norm())}
        model.eval();plain=inference_model({'channels':24,'blocks':4,'auxiliary_raw_weight':.1 if auxiliary else 0},model.state_dict()).cuda().eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x),plain(x),rtol=0,atol=0)
            fixed_l1=float((plain(x)-y).abs().mean())
        result={'augmentation':augmentation,'auxiliary':auxiliary,'steps':100,'losses':losses,
                'fixed_probe_display_l1':fixed_l1,'shared_gradient_check':gradient,'step_seconds':times,'module_times':module_times}
        all_results.append(result)
        atomic_json(args.output/'probe.json',{'status':'running','results':all_results})
        print(json.dumps({k:v for k,v in result.items() if k not in ('losses','step_seconds','module_times')}),flush=True)
        del model,opt,plain;torch.cuda.empty_cache()
    atomic_json(args.output/'probe.json',{'status':'passed','results':all_results,'sample_ids':selected,
                'crop_tlhw':[120,300,768,768],'source_middle_sha256':entry['sha256'],'script_sha256':sha(__file__),
                'wall_seconds':time.monotonic()-begin,'formal_initialization_reused':False,
                'scope':'100-step engineering ablations on two fixed training crops; not full-data quality evidence'})
    fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
