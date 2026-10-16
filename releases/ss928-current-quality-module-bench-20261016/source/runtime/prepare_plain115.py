"""Isolate current RAW before history-field correction with frozen SR weights."""
import json,gc,argparse
from pathlib import Path
import numpy as np
import torch
from target_residual import ROOT,current_setup
from prepare_motion14 import DEST
from causal_detail import GuidanceView
from tieddecode_model11 import boards
from input_anchor12 import apply

CACHE115=Path('/tmp/rawir-plain115-20261014')

def run(split='val_continuous'):
    scene='weather_heavy_c32';kind,config,df,box,count,base,original,profile=current_setup(scene)
    source=df(config,'train' if split=='train' else 'val');apply(source,scene);del original
    for path in sorted((DEST/scene/split).glob('capture_*')):
        meta=json.loads((path/'complete.json').read_text());row=meta['row']
        out=CACHE115/split/path.name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        data=GuidanceView(source,list(range(9-count,9)))
        parent,model=boards(scene,base,count,kind,box(row),500);del parent
        values=np.lib.format.open_memmap(out/'plain_current.npy',mode='w+',dtype=np.float16,shape=(len(meta['frames']),1024,1280))
        for i,f in enumerate(meta['frames']):
            rr=dict(row,frame_id=f);data.current_record,data.current_crop,data.cache=rr,(0,0,1024,1280),{}
            context,_=data.context_for(rr);context=context[None].cuda().half()
            current=data._nine_without_field()[-1:][None].cuda().half()
            inputs=current.expand(1,model.source_count,1024,1280).contiguous()
            with torch.inference_mode():full=model(inputs,context)
            if tuple(full.shape)!=(1,1,3072,3840):raise ValueError('Original SR geometry required')
            values[i]=full[0,0,::3,::3].float().cpu().numpy()
            if i%20==0:print('PLAIN115',path.name,f,flush=True)
        values.flush();(out/'complete.json').write_text(json.dumps(dict(frames=meta['frames'],original_SR_weights_changed=False,
            GT_used_for_inference=False,original_input=True,main_profile=profile,branch='current before field correction, repeated original count',input_count=model.source_count)))
        del data,model,values;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val_continuous'],default='val_continuous');args=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory);run(args.split)

