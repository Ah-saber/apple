"""Verify both exact integer layouts on all complete real 120-frame sequences."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930';MOTION=ROOT/'runs/SS928-MOTION-SPEED-20260930'
sys.path[:0]=[str(V13/'runtime'),str(MOTION/'runtime')]
from run_compact import GROUPS,sha
from baseline_contract import load_group,independent_control
from equivalent_model import BoardHalfOptimized
from compact_model import BoardCompact
from integer_output import IntegerOutput,ByteReorder
from speed_variants import RepeatByteReorder,NightByte
from probe_integer_output import ByteAfterRows
from prepare_deployment import ONNX_SITE,export_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--night',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ONNX_SITE))
    torch.set_num_threads(2);torch.manual_seed(930);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'packaged':False,'pushed':False,'input_boundary':'original nine-frame and reference inputs retained','scenes':{}}
    if a.night:
        os.environ['RAWIR_V09_CODE']=str(ROOT/'runs/SS928-BOARD-V09-20260928/code_snapshot')
        sys.path[:0]=[str(ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'),str(ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot'),str(ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925/src')]
        from reference_layout import load_candidate as refload
        from lowres_reference import load_candidate as lowload
        from output_candidates import prepare_inputs
        from ir_sr.training import dataset_for_config
        for scene in ['ordinary','special']:
            name='night_'+scene
            source=refload(scene,'after12','rows32',ROOT/'runs/SS928-BOARD-V09-20260928') if scene=='ordinary' else lowload(scene,5,'rows32',ROOT/'runs/SS928-BOARD-V09-20260928',trained=True)
            manifest=json.loads((V13/'night_calibration_final'/name/'manifest.json').read_text());data=dataset_for_config(manifest['config'],'test')
            row=next(r for r in data.records if r['scene_id']==name)
            gather=NightByte(source).cuda().eval();gather.source.output.output=ByteReorder(2).cuda()
            repeat=NightByte(source).cuda().eval();floating=NightByte(source).cuda().eval();floating.source.output.output=RepeatByteReorder(2,to_byte=False).cuda()
            def inputs(frame):
                rr=dict(row,frame_id=frame);stack=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda();c,_=data.context_for(rr,crop_tlhw=(0,0,1024,1280))
                return prepare_inputs(source,stack,c[None].cuda())
            verify(name,ByteAfterRows(source),{'gather':gather,'repeat':repeat,'float_repeat':floating},inputs,data,row,a.out,report,legacy=True)
    else:
        for group in ['day','light','heavy']:
            compact,base,config,_,basepath,df,box_for=load_group(group);independent_control(group,compact)
            data=df(config,'test')
            for scene in GROUPS[group]['scenes']:
                row=next(r for r in data.records if r['scene_id']==scene);box=box_for(row).cuda()
                for kind,source,factor in [('half',BoardHalfOptimized(base,box),2)]+([] if group=='day' else [('quarter',BoardCompact(compact,box),4)]):
                    source=source.cuda().half().eval();gather=IntegerOutput(source,factor).cuda().eval()
                    repeat=copy.deepcopy(gather);repeat.output=RepeatByteReorder(factor).cuda();floating=copy.deepcopy(gather);floating.output=RepeatByteReorder(factor,to_byte=False).cuda()
                    def inputs(frame):
                        rr=dict(row,frame_id=frame);x=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda().half();c,_=data.context_for(rr)
                        return x,c[None].cuda().half()
                    verify(scene+'_'+kind,ByteAfterRows(source),{'gather':gather,'repeat':repeat,'float_repeat':floating},inputs,data,row,a.out,report)
    print('ALL_SEQUENCE_BYTE_VERIFICATION_COMPLETE',flush=True)


def verify(name,source,variants,inputs,data,row,out,report,legacy=False):
    from ir_sr.sequence_normalization import regional_key
    norm=data.base.sequence_normalization
    key=regional_key(row) if norm.regional else row['domain']+'/'+row['sequence_id']
    frame_count=int(norm.by_key[key]['frames'])
    source.requires_grad_(False)
    for model in variants.values():model.requires_grad_(False)
    counts={k:0 for k in variants};history=[]
    with torch.inference_mode():
        for frame in range(frame_count):
            args=inputs(frame);expected=source(*args)
            assert expected.shape==(1,1,3072,3840) and expected.dtype==torch.uint8
            ids=data.frame_ids(dict(row,frame_id=frame));history.append([int(v) for v in ids])
            assert max(ids)<=frame
            for label,model in variants.items():
                actual=model(*args)
                target=source.source(*args) if label=='float_repeat' else expected
                assert torch.equal(actual,target),(name,label,frame)
                counts[label]+=1
            if frame%30==0:print('ALL_SEQUENCE',name,frame,flush=True)
    args=tuple(v.clone() for v in args)
    graphs={}
    for label,model in variants.items():
        path=out/(name+'_'+label+'_integer.onnx')
        if legacy:
            import onnx
            torch.onnx.export(model,args,str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
            graph=onnx.load(str(path));onnx.checker.check_model(graph);ops=sorted({n.op_type for n in graph.graph.node});assert 'GridSample' not in ops
            meta={'path':str(path),'sha256':sha(path),'operators':ops,'retains_original_night_Resize':True}
        else:meta=export_model(model,args,path,['nine_raw','reference_thumb'],'display_gray')
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2
        session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        cpu=session.run(None,{k:v.cpu().numpy() for k,v in zip(['nine_raw','reference_thumb'],args)})[0]
        target=source.source(*args) if label=='float_repeat' else expected
        delta=np.abs(cpu.astype(np.float32)-target.detach().cpu().numpy().astype(np.float32))
        meta['ORT_final_frame']={'frame':frame_count-1,'mean_gray':float(delta.mean()),'max_gray':float(delta.max()),'shape':list(cpu.shape),'dtype':str(cpu.dtype)}
        assert cpu.shape==(1,1,3072,3840) and cpu.dtype==(np.float16 if label=='float_repeat' else np.uint8)
        graphs[label]=meta
    report['scenes'][name]={'frames':frame_count,'same_backend_exact_frames':counts,'history_frame_ids':history,
        'all_frames_use_only_present_or_past':True,'graphs':graphs}
    (out/'all_sequence_verification.json').write_text(json.dumps(report,indent=2))
    print('ALL_SEQUENCE_VERIFIED',name,counts,flush=True)


if __name__=='__main__':main()
