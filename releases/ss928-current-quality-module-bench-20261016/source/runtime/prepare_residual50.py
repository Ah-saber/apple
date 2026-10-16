"""Replay the preserved full-field controller for patch training and video comparison."""
import json,gc
from pathlib import Path
import numpy as np
import torch
from target_residual import ROOT
from prepare_motion14 import DEST
from quality_control43 import QualityControl43

OUT50=ROOT/'MOTION50-20261014'
CACHE50=Path('/tmp/rawir-residual50-20261014')
def run():
    scene='weather_heavy_c32'
    for split in ['train','val_continuous']:
        for path in sorted((DEST/scene/split).glob('capture_*')):
            meta=json.loads((path/'complete.json').read_text());out=CACHE50/split/path.name;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            arrays={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ['baseline','current_only','gated_history','past_background','activity','occupancy']}
            source=np.lib.format.open_memmap(out/'source43.npy',mode='w+',dtype=np.float32,shape=(len(meta['frames']),1024,1280))
            controller=QualityControl43(scene)
            for i,f in enumerate(meta['frames']):
                v={k:torch.from_numpy(a[i].astype(np.float32))[None,None].cuda() for k,a in arrays.items()}
                source[i]=controller.native(v,initial=i==0 or f!=meta['frames'][i-1]+1)[0,0].cpu().numpy()
            source.flush();(out/'complete.json').write_text(json.dumps(meta,indent=2))
            print('SOURCE50_DONE',split,path.name,flush=True)
            del source,controller;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory);run()
