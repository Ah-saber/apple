import os
import sys
from pathlib import Path
import torch

ROOT=Path('/data/zhangbenzhuang/huawei_sr')
V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path.insert(0,str(V13/'runtime'))
torch.set_num_threads(2)
from run_compact import setup
for group in ['day','light','heavy']:
    _,base,config,_,path,df,box_for=setup(group)
    q=base.quarter
    print('GROUP',group,'SOURCE',path,'CURRENT',base.current_only,flush=True)
    for name in ['front','body','tail','output']:
        print(name,getattr(q,name),flush=True)
    data=df(config,'test')
    for scene in sorted({r['scene_id'] for r in data.records}):
        row=next(r for r in data.records if r['scene_id']==scene)
        print('BOX',scene,box_for(row),flush=True)
os.environ['RAWIR_V09_CODE']=str(ROOT/'runs/SS928-BOARD-V09-20260928/code_snapshot')
sys.path[:0]=[str(ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'),str(ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot')]
from reference_layout import load_candidate as refload
from lowres_reference import load_candidate as lowload
for scene in ['ordinary','special']:
    model=refload(scene,'after12','rows32',ROOT/'runs/SS928-BOARD-V09-20260928') if scene=='ordinary' else lowload(scene,5,'rows32',ROOT/'runs/SS928-BOARD-V09-20260928',trained=True)
    print('NIGHT',scene,model,flush=True)
