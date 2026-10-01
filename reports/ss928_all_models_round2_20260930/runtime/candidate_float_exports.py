"""Float-output alternatives for reviewable motion candidates, with byte-interface controls."""
import argparse,copy,json,os,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
sys.path[:0]=[str(V13/'runtime'),str(ROOT/'runs/SS928-MOTION-SPEED-20260930/runtime')]
from baseline_contract import load_group
from run_compact import GROUPS,sha
from short_history import ShortHistory
from evaluate_short import BoardShort
from motion_residual import MotionResidual
from evaluate_residual import BoardAdaptive
from speed_variants import RepeatByteReorder,NightByte
from finalize_candidates import benchmark
from prepare_deployment import export_model,ONNX_SITE
from adaptive_history import GatedNightFront
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ONNX_SITE))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'packaged':False,'pushed':False,'candidates':{}}
    def run(name,byte,floating,directory,extra,night=False):
        controls=[]
        for path in sorted(directory.glob('test_*.npz')):
            sample=np.load(path);args=tuple(torch.from_numpy(sample[k]).cuda() for k in ['nine_raw','reference_thumb'])
            if not night:args=tuple(v.half() for v in args)
            with torch.inference_mode():
                expected=byte(*args);actual=floating(*args);assert actual.dtype==torch.float16 and torch.equal(expected,actual.byte()),(name,path)
            controls.append({'sample':str(path),'sample_sha256':sha(path),'float_truncation_exact_byte':True})
        byte.requires_grad_(False);floating.requires_grad_(False);args=tuple(v.clone() for v in args)
        path=a.out/(name+'_float.onnx')
        if night:
            import onnx
            torch.onnx.export(floating,args,str(path),input_names=['nine_raw','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False);g=onnx.load(str(path));onnx.checker.check_model(g)
            meta={'path':str(path),'sha256':sha(path),'operators':sorted({n.op_type for n in g.graph.node}),'retains_original_night_Resize':True}
        else:meta=export_model(floating,args,path,['nine_raw','reference_thumb'],'display_gray')
        timing=benchmark(floating,args);timing['precision']='candidate original mixed input/weight types preserved' if night else 'FP16'
        report['candidates'][name]={'full_output_shape':[1,1,3072,3840],'output_dtype':'float16','source':extra,'interface_controls':controls,'graph':meta,'GPU':timing}
        (a.out/'candidate_float_exports.json').write_text(json.dumps(report,indent=2));torch._dynamo.reset();torch.cuda.empty_cache();print('CANDIDATE_FLOAT',name,timing['mean_ms'],flush=True)
    for group in ['light','heavy','day']:
        _,base,config,_,_,df,box_for=load_group(group);data=df(config,'test')
        path=a.root/(f'half3_{group}_3/best.pt' if group!='day' else 'residual_nonzero_day_half/best.pt');state=torch.load(path,map_location='cpu',weights_only=False)
        model=ShortHistory(base,3,'half') if group!='day' else MotionResidual(base,'half',state['threshold']);model.load_state_dict(state['model'])
        for scene in GROUPS[group]['scenes']:
            row=next(r for r in data.records if r['scene_id']==scene);box=box_for(row).cuda();byte=(BoardShort(model,box) if group!='day' else BoardAdaptive(model,box)).cuda().half().eval();floating=copy.deepcopy(byte)
            if group!='day':floating.model.output=RepeatByteReorder(2,to_byte=False).cuda()
            else:floating.output=RepeatByteReorder(2,to_byte=False).cuda()
            directory=V13/'deployment'/scene
            if not (directory/'test_00.npz').exists():directory=V13/'deployment_corrected'/scene
            run(scene+'_motion',byte,floating,directory,{'checkpoint':str(path),'sha256':sha(path)})
    os.environ['RAWIR_V09_CODE']=str(ROOT/'runs/SS928-BOARD-V09-20260928/code_snapshot')
    sys.path[:0]=[str(ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'),str(ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot'),str(ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925/src')]
    from reference_layout import load_candidate as refload
    from lowres_reference import load_candidate as lowload
    records=json.loads((a.root/'night_motion_weak/night_motion.json').read_text())['scenes']
    for scene in ['ordinary','special']:
        name='night_'+scene;r=records[name];source=refload(scene,'after12','rows32',ROOT/'runs/SS928-BOARD-V09-20260928') if scene=='ordinary' else lowload(scene,5,'rows32',ROOT/'runs/SS928-BOARD-V09-20260928',trained=True)
        threshold=r['thresholds_input_units'][r['chosen_percentile']];source.front=GatedNightFront(source.front,threshold,strength=.25).cuda().eval();byte=NightByte(source).cuda().eval();floating=copy.deepcopy(byte);floating.source.output.output=RepeatByteReorder(2,to_byte=False).cuda()
        run(name+'_motion',byte,floating,V13/'night_calibration_final'/name,{'threshold':threshold,'strength':.25,'weights':'original night sources, no new trained weights'},night=True)
    print('CANDIDATE_FLOAT_EXPORTS_COMPLETE',flush=True)
if __name__=='__main__':main()
