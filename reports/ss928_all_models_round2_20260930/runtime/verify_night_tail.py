"""Real full-sequence numerical control for the algebraic night-tail alternative."""
import argparse,json,os,sys
from pathlib import Path
import torch
ROOT=Path('/data/zhangbenzhuang/huawei_sr');V13=ROOT/'runs/SS928-BOARD-V13-20260930'
os.environ['RAWIR_V09_CODE']=str(ROOT/'runs/SS928-BOARD-V09-20260928/code_snapshot')
sys.path[:0]=[str(ROOT/'runs/SS928-BOARD-V12-20260929/code_snapshot'),str(ROOT/'runs/SS928-BOARD-V11-20260928/code_snapshot'),str(ROOT/'code/worktrees/ss928-quality-special-night-nine-frame-20260925/src'),str(V13/'runtime')]
from reference_layout import load_candidate as refload
from lowres_reference import load_candidate as lowload
from output_candidates import prepare_inputs
from ir_sr.training import dataset_for_config
from ir_sr.sequence_normalization import regional_key
from speed_variants import NightFixed
from linear_tail_corrections import CorrectedLinearTail
from float_speed_profile import profile
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    report={'NPU_measured':False,'packaged':False,'pushed':False,'scenes':{}}
    for scene in ['ordinary','special']:
        name='night_'+scene;source=refload(scene,'after12','rows32',ROOT/'runs/SS928-BOARD-V09-20260928') if scene=='ordinary' else lowload(scene,5,'rows32',ROOT/'runs/SS928-BOARD-V09-20260928',trained=True)
        model=NightFixed(source,scene).cuda().eval();model.fused=CorrectedLinearTail(source.core.model.tail,source.output.conv).cuda().eval()
        manifest=json.loads((V13/'night_calibration_final'/name/'manifest.json').read_text());data=dataset_for_config(manifest['config'],'test');row=next(r for r in data.records if r['scene_id']==name);count=int(data.base.sequence_normalization.by_key[regional_key(row)]['frames']);records=[]
        with torch.inference_mode():
            for frame in range(count):
                rr=dict(row,frame_id=frame);stack=data.normalized_stack(rr,(0,0,1024,1280))[None].cuda();c,_=data.context_for(rr,crop_tlhw=(0,0,1024,1280));args=prepare_inputs(source,stack,c[None].cuda())
                expected=source(*args).byte();actual=model(*args);delta=(expected.float()-actual.float()).abs()
                r={'frame':frame,'mean_gray':float(delta.mean()),'max_gray':float(delta.max()),'changed_fraction':float((delta!=0).float().mean())};records.append(r)
                assert r['max_gray']<=2 and r['mean_gray']<.1,(name,r)
        report['scenes'][name]={'actual_frames':count,'per_frame':records,'mean_gray':sum(r['mean_gray'] for r in records)/count,'max_gray':max(r['max_gray'] for r in records),'source_profile':profile(source,args),'fused_profile':profile(model,args),'limits':'floating point algebra holds before rounding; FP16 arithmetic order and fixed reference sampling can change integer output; GPU previously slower; keep only as board-test alternative'}
        (a.out/'night_tail_full_sequence.json').write_text(json.dumps(report,indent=2));print('NIGHT_TAIL_VERIFIED',name,count,report['scenes'][name]['max_gray'],flush=True)
if __name__=='__main__':main()
