"""Export complete and phase-only half-grid models for SS928 profiling."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Rows32(nn.Module):
    def __init__(self):
        super().__init__()
        rows=32
        group=rows//2
        kernel=torch.zeros(group*4,rows,3,6)
        for fy in range(group):
            for py in range(2):
                for px in range(2):
                    channel=fy*4+py*2+px
                    for sy in range(3):
                        yy=fy*6+py*3+sy
                        for sx in range(3):
                            kernel[channel,yy%rows,yy//rows,px*3+sx]=1
        self.register_buffer('kernel',kernel)

    def forward(self,phases):
        n,c,h,w=phases.shape
        packed=phases.reshape(n,c,h//16,16,w).permute(0,3,1,2,4)
        packed=packed.reshape(n,c*16,h//16,w)
        blocked=F.conv_transpose2d(packed,self.kernel,stride=(3,6))
        return blocked.permute(0,2,1,3).reshape(n,1,h*6,w*6)


class BoardHalf(nn.Module):
    def __init__(self,base,box,layout):
        super().__init__()
        self.base=base
        self.register_buffer('box',box)
        self.layout=layout
        self.rows32=Rows32()

    def forward(self,stack,context):
        model=self.base
        q=model.quarter
        if model.current_only:
            stack=stack[:,-1:].expand_as(stack)
        features=q.body(q.front(stack))
        reference=q.reference(context.to(dtype=q.front.first.weight.dtype,
                              memory_format=torch.channels_last),self.box,
                              features.shape[-2:]).to(features.dtype)
        phases=q.output.conv(q.tail(features+reference))
        if getattr(q,'raw_skip',False):
            phases=phases+F.pixel_unshuffle(stack[:,-1:],2)
        phases=phases.clamp(0,1)*255
        if self.layout=='phases':
            return phases
        if self.layout=='rows32':
            return self.rows32(phases)
        return F.interpolate(F.pixel_shuffle(phases,2),scale_factor=3,
                             mode='nearest')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--night-runtime',type=Path,required=True)
    p.add_argument('--config-checkpoint',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path,required=True)
    p.add_argument('--student-checkpoint',type=Path,required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--onnx-site',type=Path)
    p.add_argument('--save-probe',action='store_true')
    a=p.parse_args()
    sys.path[:0]=[str(a.code/'src'),str(a.night_runtime),str(Path(__file__).parent)]
    if a.onnx_site:
        sys.path.append(str(a.onnx_site))
    import onnx
    from ir_sr.model import inference_model
    from ir_sr.training import dataset_for_config
    from half_student import make_student
    from train_quarter_student import BoxQuarter,box_for
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    state=torch.load(a.config_checkpoint,map_location='cpu',weights_only=False)
    reference_state=torch.load(a.reference_checkpoint,map_location='cpu',weights_only=False)
    ck=torch.load(a.student_checkpoint,map_location='cpu',weights_only=False)
    assert ck.get('grid')=='half'
    reference=inference_model(reference_state['config'],
                              reference_state['model']).global_reference
    base=BoxQuarter(make_student(copy.deepcopy(reference),ck),
                    current_only=ck.get('current_only',False))
    base.load_state_dict(ck['model'],strict=True)
    base=base.eval().cuda().to(memory_format=torch.channels_last)
    data=dataset_for_config(state['config'],'test')
    rows=[row for row in data.records if row['scene_id']==a.scene]
    assert rows
    boxes=torch.cat([box_for(row) for row in rows])
    assert torch.allclose(boxes,boxes[:1].expand_as(boxes)),boxes
    box=boxes[:1].cuda()
    row=rows[0]
    stack=data.normalized_stack(row,(0,0,1024,1280))[None].cuda()
    context,_=data.context_for(row)
    context=context[None].cuda()
    a.out.mkdir(parents=True,exist_ok=True)
    manifest={'scene':a.scene,'student':str(a.student_checkpoint),
              'student_sha256':sha(a.student_checkpoint),
              'context_box':box.cpu().flatten().tolist(),'files':[]}
    with torch.inference_mode():
        outputs={}
        for layout in ('nearest','rows32','phases'):
            model=BoardHalf(base,box,layout).eval().cuda()
            output=model(stack,context).float()
            outputs[layout]=output
            path=a.out/f'{a.scene}_{layout}.onnx'
            torch.onnx.export(model,(stack,context),str(path),
                input_names=['nine_raw','reference_thumb'],
                output_names=['display_gray' if layout!='phases' else 'four_phases'],
                opset_version=17,dynamo=False)
            onnx.checker.check_model(str(path))
            manifest['files'].append({'layout':layout,'path':str(path),
                'sha256':sha(path),'shape':list(output.shape),
                'nodes':len(onnx.load(str(path)).graph.node)})
        assert outputs['nearest'].shape==outputs['rows32'].shape==(1,1,3072,3840)
        difference=(outputs['nearest']-outputs['rows32']).abs()
        manifest['rows32_vs_nearest']={'mae_gray':float(difference.mean()),
            'max_abs_gray':float(difference.max())}
        assert manifest['rows32_vs_nearest']['max_abs_gray']<.01
        if a.save_probe:
            probe_raw=torch.full_like(stack,0.25)
            probe_context=torch.full_like(context,0.35)
            expected=BoardHalf(base,box,'rows32').eval().cuda()(
                probe_raw,probe_context).float().cpu().numpy()
            probe=a.out/'rows32_probe.npz'
            np.savez_compressed(probe,nine_raw=probe_raw.cpu().numpy(),
                reference_thumb=probe_context.cpu().numpy(),expected=expected)
            manifest['probe']={'path':str(probe),'sha256':sha(probe)}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
