"""Real train calibration and test references for the preserved v0.12 night paths."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from run_compact import ROOT, CODE, sha


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    v09=ROOT/'runs/SS928-BOARD-V09-20260928'
    v11=ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot'
    v12=ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'
    os.environ['RAWIR_V09_CODE']=str(v09/'code_snapshot')
    night_code=ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925'
    sys.path[:0]=[str(v12),str(v11),str(night_code/'src')]
    from reference_layout import load_candidate as load_reference
    from lowres_reference import load_candidate as load_lowref
    from output_candidates import prepare_inputs
    from ir_sr.training import dataset_for_config
    torch.set_num_threads(2); torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    a.out.mkdir(parents=True,exist_ok=True)
    report={'weights_changed':False,'NPU_measured':False,'scenes':{},'source_code':str(night_code)}
    for scene in ['ordinary','special']:
        run=('SS928-QUALITY-NIGHT-NINE-FRAME-TRAJECTORY-20260925-2K-R1' if scene=='ordinary' else
             'SS928-QUALITY-SPECIAL-NIGHT-NINE-FRAME-20260925-2K-R1')
        checkpoint=ROOT/'runs'/run/'checkpoints/step_000002000.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=state['config']
        model=(load_reference(scene,'after12','rows32',v09) if scene=='ordinary' else
               load_lowref(scene,5,'rows32',v09,trained=True)).eval()
        target='night_'+scene; out=a.out/target;out.mkdir(exist_ok=True)
        entry={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
               'path':'reference_after12' if scene=='ordinary' else 'lowref_k5',
               'calibration':[],'test':[],'config':config}
        for split,count in [('train',4),('test',12)]:
            data=dataset_for_config(config,split)
            candidates=[r for r in data.records if r['scene_id']==target and
                        (split!='train' or (r['frame_id']>=8 and len(set(data.frame_ids(r)))==9))]
            assert len(candidates)>=count
            selected=([candidates[i] for i in np.linspace(0,len(candidates)-1,count,dtype=int)]
                      if split=='train' else candidates[:count])
            for i,row in enumerate(selected):
                stack=data.normalized_stack(row,(0,0,1024,1280))[None].cuda()
                c,_=data.context_for(row,crop_tlhw=(0,0,1024,1280));c=c[None].cuda()
                raw,context=prepare_inputs(model,stack,c)
                stages={};hooks=[]
                if split=='test' and i==0:
                    modules={'front':model.front,'head':model.core.model.head,'body':model.core.model.body,
                             'reference_encoder':model.core.model.global_reference.encoder,
                             'reference_project':model.core.model.global_reference.project,
                             'tail':model.core.model.tail,'output_before_clip':model.output.conv}
                    for name,module in modules.items():
                        hooks.append(module.register_forward_hook(lambda m,args,value,n=name:stages.update({n:value.detach().float().cpu().numpy()})))
                with torch.inference_mode():
                    output=model(raw,context).float()
                    assert tuple(output.shape)==(1,1,3072,3840)
                    native=F.avg_pool2d(output,3,3).cpu().numpy()
                for hook in hooks:hook.remove()
                history=data.frame_ids(row)
                originals=np.stack([data.base._raw(dict(row,frame_id=f)) for f in history])
                values={'nine_raw':raw.cpu().numpy(),'reference_thumb':context.cpu().numpy(),
                        'normalized_nine_fp32':stack.cpu().numpy(),'raw_u16':originals,'pc_native_gray':native}
                if split=='test':
                    values['gt_u8']=np.array(Image.open(Path(config['data_root'])/row['target']['path']))
                path=out/f'{split}_{i:02d}.npz';np.savez_compressed(path,**values)
                entry['calibration' if split=='train' else 'test'].append({'path':str(path),'sha256':sha(path),
                    'sample':row,'history_frame_ids':history,'unique_history':len(set(history)),
                    'normalization':[data.normalization_for(dict(row,frame_id=f)) for f in history],
                    'input_shapes':{'nine_raw':list(raw.shape),'reference_thumb':list(context.shape)}})
                if stages:
                    path=out/'pc_layers_source_fp16.npz';np.savez_compressed(path,**stages)
                    entry['layers']={'path':str(path),'sha256':sha(path),'precision':'FP16 source, dumped as FP32',
                        'sample_id':row['sample_id'],'stages':{k:list(v.shape) for k,v in stages.items()}}
                print('NIGHT_CALIBRATION',target,split,i,history,flush=True)
        (out/'manifest.json').write_text(json.dumps(entry,indent=2))
        report['scenes'][target]=entry
        (a.out/'manifest.json').write_text(json.dumps(report,indent=2))
    print('NIGHT_CALIBRATION_COMPLETE',flush=True)


if __name__=='__main__':main()
