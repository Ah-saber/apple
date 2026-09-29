"""Replace cached day teacher predictions with the frozen single-frame model."""
import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--source-cache',type=Path,required=True)
    p.add_argument('--baseline-checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.code/'src'))
    from ir_sr.model import inference_model
    torch.set_num_threads(2)
    state=torch.load(a.baseline_checkpoint,map_location='cpu',weights_only=False)
    model=inference_model(state['config'],state['model']).cuda().eval()
    saved=torch.load(a.source_cache,map_location='cpu',weights_only=False)
    new=copy.deepcopy(saved)
    x=saved['values']['stack'].flatten(0,1)
    c=saved['values']['context'].flatten(0,1)
    b=saved['values']['box'].flatten(0,1)
    predictions=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for begin in range(0,len(x),16):
            output=model(x[begin:begin+16,-1:].cuda().float(),
                         context=c[begin:begin+16].cuda().float(),
                         context_box=b[begin:begin+16].cuda().float())
            native=F.avg_pool2d(output.float().clamp(0,1),3,3)
            predictions.append(native.cpu().half())
    target=torch.cat(predictions).reshape_as(saved['values']['teacher'])
    new['values']['teacher']=target
    new['teacher_source']=str(a.baseline_checkpoint)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(new,a.out)
    print('saved',a.out,'samples',len(x),'shape',tuple(target.shape),flush=True)


if __name__=='__main__':main()
