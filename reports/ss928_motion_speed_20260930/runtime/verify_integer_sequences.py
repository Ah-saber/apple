"""Verify all weather output layouts on full sequences; leave files for user review."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PREVIOUS=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-BOARD-V13-20260930')
sys.path.insert(0,str(PREVIOUS/'runtime'))
from run_compact import setup,GROUPS,OLD,sha
from prepare_deployment import ONNX_SITE,export_model
from compact_model import BoardCompact
from integer_output import IntegerOutput


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);sys.path[:0]=[str(OLD),str(ONNX_SITE)]
    import onnxruntime as ort
    torch.set_num_threads(2);torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'SDK_uint8_Gather_and_Transpose':'unmeasured','packaged':False,'pushed':False,'scenes':{}}
    for group in ['light','heavy']:
        checkpoint=PREVIOUS/(group+'_temporal')/'best.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);GROUPS[group].update(ck['spec'])
        model,base,config,teacher,base_path,df,box_for=setup(group)
        model.load_state_dict(ck['model']);model=model.cuda().float().eval();data=df(config,'test')
        for scene in GROUPS[group]['scenes']:
            row=next(r for r in data.records if r['scene_id']==scene);box=box_for(row).cuda()
            board=BoardCompact(model,box).cuda().half().eval();integer=IntegerOutput(board,4).cuda().eval()
            with torch.inference_mode():
                for frame in range(120):
                    rr=dict(row,frame_id=frame)
                    x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda().half();c,_=data.context_for(rr);c=c[None].cuda().half()
                    expected=board(x,c).byte();actual=integer(x,c)
                    assert tuple(actual.shape)==(1,1,3072,3840)
                    assert torch.equal(expected,actual),(scene,frame)
                    if frame%30==0:print('BYTE_SEQUENCE',scene,frame,flush=True)
                graph=export_model(integer,(x,c),a.out/f'{scene}_quarter_integer_byte_fp16.onnx',['nine_raw','reference_thumb'],'display_gray')
                options=ort.SessionOptions();options.intra_op_num_threads=2
                session=ort.InferenceSession(graph['path'],sess_options=options,providers=['CPUExecutionProvider'])
                output=session.run(None,{'nine_raw':x.cpu().numpy(),'reference_thumb':c.cpu().numpy()})[0]
                assert tuple(output.shape)==(1,1,3072,3840) and output.dtype==np.uint8
                d=np.abs(output.astype(np.float32)-actual.cpu().numpy())
            record={'source_checkpoint':str(checkpoint),'source_sha256':sha(checkpoint),'frames':120,
                'same_backend_equal_pixels':True,'graph':graph,'ORT_final_frame':119,
                'ORT_CPU_vs_PyTorch_GPU_byte_mae':float(d.mean()),'ORT_CPU_vs_PyTorch_GPU_byte_max':float(d.max()),
                'comparison':'integer early-Cast layout vs FP16 rows32 followed by the same truncating Cast'}
            report['scenes'][scene]=record;(a.out/'sequence_byte_verification.json').write_text(json.dumps(report,indent=2))
            print('BYTE_SCENE_COMPLETE',scene,json.dumps(record),flush=True)
    print('BYTE_SEQUENCE_VERIFICATION_COMPLETE',flush=True)


if __name__=='__main__':main()
