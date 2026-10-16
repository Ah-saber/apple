"""Prepare label-free RAW confirmation; current SR cannot veto a weak body."""
import json
from pathlib import Path
import numpy as np
import cv2
from prepare_motion14 import DEST
from prepare_confidence76 import CACHE76
from road_residual53 import support53
from gate_body80 import physical_body_gate80

CACHE80=Path('/tmp/rawir-body80-inputs-20261014')
def run():
    scene='weather_heavy_c32';top,bottom=176,654
    mask=support53((top,0,bottom-top,1280),'cpu')[0,0].numpy()>.9
    for split in ['train','val_continuous']:
        for path in sorted((DEST/scene/split).glob('capture_*')):
            out=CACHE80/split/path.name;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            p=np.load(CACHE76/split/path.name/'presence76.npy',mmap_mode='r')
            now=np.load(CACHE76/split/path.name/'instant_presence76.npy',mmap_mode='r')
            result=np.lib.format.open_memmap(out/'signed_gate80.npy',mode='w+',dtype=np.float16,shape=p.shape)
            for i in range(len(p)):
                result[i]=0;result[i,top:bottom]=physical_body_gate80(p[i,top:bottom],now[i,top:bottom],mask)
            result.flush();(out/'complete.json').write_text(json.dumps(dict(GT_used=False,SR_used=False,background='available past64 from76',current_RAW_mean=3,coherent='two strong connected pixels and radius1 extension')))
            print('BODY80_GATE',split,path.name,flush=True)
if __name__=='__main__':
    cv2.setNumThreads(2);run()
