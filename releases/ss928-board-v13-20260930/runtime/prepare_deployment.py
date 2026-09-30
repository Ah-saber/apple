"""Verify equivalent graphs and export real train calibration plus PC layer references."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from run_compact import setup, GROUPS, ROOT, CODE, OLD, sha, score
from equivalent_model import BoardHalfOptimized
from compact_model import FixedSample

ONNX_SITE=ROOT/'runs/TASK-019-export-dependencies/site-packages'


def arrays(values):
    return {k:v.detach().float().cpu().numpy() for k,v in values.items()}


def export_model(model, inputs, path, names, output_name):
    import onnx
    torch.onnx.export(model,inputs,str(path),input_names=names,output_names=[output_name],
                      opset_version=17,dynamo=False)
    graph=onnx.load(str(path)); onnx.checker.check_model(graph)
    kinds={node.op_type for node in graph.graph.node}
    assert 'GridSample' not in kinds and 'Resize' not in kinds, kinds
    return {'path':str(path),'sha256':sha(path),'operators':sorted(kinds),
            'nodes':len(graph.graph.node),
            'inputs':[{ 'name':v.name,'type':v.type.tensor_type.elem_type,
                       'shape':[d.dim_value for d in v.type.tensor_type.shape.dim]} for v in graph.graph.input],
            'outputs':[{ 'name':v.name,'type':v.type.tensor_type.elem_type,
                        'shape':[d.dim_value for d in v.type.tensor_type.shape.dim]} for v in graph.graph.output]}


def compare_sample(box,height,width):
    module=FixedSample(box,height,width).cuda()
    source=torch.rand(1,12,64,64,device='cuda')
    y0,x0,y1,x1=box.unbind(1)
    yy=(torch.arange(height,device='cuda')+.5)/height
    xx=(torch.arange(width,device='cuda')+.5)/width
    gy=y0[:,None,None]+yy[None,:,None]*(y1-y0)[:,None,None]
    gx=x0[:,None,None]+xx[None,None,:]*(x1-x0)[:,None,None]
    grid=torch.stack((gx.expand(-1,height,-1)*2-1,gy.expand(-1,-1,width)*2-1),-1)
    expected=F.grid_sample(source,grid,mode='bilinear',padding_mode='border',align_corners=False)
    delta=(module(source)-expected).abs()
    assert float(delta.max())<2e-5
    return {'mean_abs':float(delta.mean()),'max_abs':float(delta.max())}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--groups',nargs='+',choices=GROUPS,default=list(GROUPS))
    p.add_argument('--with-test-inputs',action='store_true'); a=p.parse_args()
    sys.path[:0]=[str(OLD),str(ONNX_SITE)]
    torch.set_num_threads(2); torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    a.out.mkdir(parents=True,exist_ok=True)
    manifest={'source_commit':'0c1f674d7b5f1d1aed8e75ac64082c97e0dedc9e',
              'baseline_release_commit':'b70a69380c8af36129d9195e9145daf5824328c0',
              'NPU_measured':False,'scenes':{},'calibration_uses_GT':False}
    for group in a.groups:
        unused,base,config,teacher,base_path,df,box_for=setup(group)
        del unused
        base=base.cuda().float().eval().to(memory_format=torch.channels_last)
        test=df(config,'test'); train=df(config,'train')
        for scene in GROUPS[group]['scenes']:
            out=a.out/scene; out.mkdir(exist_ok=True)
            tests=[r for r in test.records if r['scene_id']==scene]
            box=box_for(tests[0]).cuda()
            assert all(torch.allclose(box_for(r),box.cpu()) for r in tests)
            entry={'config_checkpoint':str(teacher),'config_sha256':sha(teacher),
                   'student':str(base_path),'student_sha256':sha(base_path),'context_box':box.cpu().flatten().tolist(),
                   'normalization_sha256':sha(config['sequence_normalization_index']),
                   'dynamic_correction':config.get('causal_dynamic_correction'),
                   'sampling_control':compare_sample(box,512,640),'graphs':[],
                   'calibration':[],'test':[],'equivalence_120frames':[]}
            (out/'config.json').write_text(json.dumps(config,indent=2))
            candidates=[r for r in train.records if r['scene_id']==scene and r['frame_id']>=8 and len(set(train.frame_ids(r)))==9]
            assert len(candidates)>=4, (scene,len(candidates))
            selected=[candidates[i] for i in np.linspace(0,len(candidates)-1,4,dtype=int)]
            optimized=BoardHalfOptimized(base,box).cuda().eval()
            optimized_half=copy.deepcopy(optimized).half()
            project_after=BoardHalfOptimized(base,box,projection_after=True).cuda().eval()
            maximum=0.
            for i,row in enumerate(selected):
                stack=train.normalized_stack(row,(0,0,1024,1280))[None].cuda()
                context,_=train.context_for(row); context=context[None].cuda()
                history=train.frame_ids(row)
                raw=np.stack([train.base._raw(dict(row,frame_id=f)) for f in history])
                norm=[train.normalization_for(dict(row,frame_id=f)) for f in history]
                with torch.inference_mode():
                    original=F.avg_pool2d(optimized(stack,context),3,3)
                path=out/f'train_{i:02d}.npz'
                np.savez_compressed(path,nine_raw=stack.cpu().numpy(),reference_thumb=context.cpu().numpy(),
                    raw_u16=raw,pc_native_gray=original.cpu().numpy())
                entry['calibration'].append({'path':str(path),'sha256':sha(path),'sample':row,
                    'history_frame_ids':history,'history_unique':len(set(history)),
                    'normalization':norm,'deployment_context_box':box.cpu().flatten().tolist(),
                    'training_context_box':box_for(row).flatten().tolist(),
                    'frame_mean_abs_variation_normalized':float((stack[:,1:]-stack[:,:-1]).abs().mean())})
                print('CALIBRATION',scene,i,history,flush=True)
            with torch.inference_mode():
                for i,row in enumerate(tests):
                    stack=test.normalized_stack(row,(0,0,1024,1280))[None].cuda()
                    context,_=test.context_for(row); context=context[None].cuda()
                    original=base.native(stack,context,box).float().clamp(0,1)*255
                    actual=optimized(stack,context)
                    native=F.avg_pool2d(actual,3,3)
                    delta=(native-original).abs()
                    maximum=max(maximum,float(delta.max()))
                    projected=F.avg_pool2d(project_after(stack,context),3,3)
                    pd=(projected-original).abs()
                    gt=np.array(Image.open(Path(config['data_root'])/row['target']['path']))
                    path=out/f'test_{i:02d}.npz'
                    values={'pc_native_gray':original.cpu().numpy(),'optimized_native_gray':native.cpu().numpy(),
                            'gt_u8':gt}
                    if a.with_test_inputs:
                        values['nine_raw']=stack.cpu().numpy(); values['reference_thumb']=context.cpu().numpy()
                        values['raw_u16']=np.stack([test.base._raw(dict(row,frame_id=f)) for f in test.frame_ids(row)])
                    np.savez_compressed(path,**values)
                    entry['test'].append({'path':str(path),'sha256':sha(path),'sample':row,
                        'history_frame_ids':test.frame_ids(row),'equivalent_mae_gray':float(delta.mean()),
                        'equivalent_max_gray':float(delta.max()),'after_projection_mae_gray':float(pd.mean()),
                        'after_projection_max_gray':float(pd.max())})
                    if i==0:
                        stages=optimized.stages(stack,context)
                        path=out/'pc_layers_fp32.npz'; np.savez_compressed(path,**arrays(stages))
                        entry['layers']={'path':str(path),'sha256':sha(path),'sample_id':row['sample_id'],
                            'stage_statistics':{k:{'shape':list(v.shape),'minimum':float(v.min()),
                                'maximum':float(v.max()),'mean':float(v.mean())} for k,v in stages.items()}}
                        names=['nine_raw','reference_thumb']
                        for precision in ['fp32','fp16']:
                            m=copy.deepcopy(optimized).to(dtype=torch.float32 if precision=='fp32' else torch.float16)
                            inp=(stack.to(m.base.quarter.front.first.weight.dtype),context.to(m.base.quarter.front.first.weight.dtype))
                            for layout in ['rows32','phases','byte']:
                                m.layout=layout
                                path=out/f'{scene}_equivalent_{precision}_{layout}.onnx'
                                entry['graphs'].append(export_model(m,inp,path,names,'display_gray' if layout!='phases' else 'four_phases'))
                        if group=='day':
                            alt_run=ROOT/'runs/SS928-FIVE-SCENE-HALF-DAY-RAWSKIP-20260929/best.pt'
                            alt=torch.load(alt_run,map_location='cpu',weights_only=False)
                            from half_student import make_student
                            from train_quarter_student import BoxQuarter
                            alt_base=BoxQuarter(make_student(copy.deepcopy(base.quarter.reference),alt),current_only=True)
                            alt_base.load_state_dict(alt['model'],strict=True); alt_base=alt_base.cuda().float().eval()
                            single=BoardHalfOptimized(alt_base,box,single_frame=True).cuda().float().eval()
                            expected=alt_base.native(stack,context,box).float().clamp(0,1)*255
                            d=(F.avg_pool2d(single(stack[:,-1:],context),3,3)-expected).abs()
                            assert float(d.max())<.01
                            entry['single_frame']={'checkpoint':str(alt_run),'sha256':sha(alt_run),
                                'mae_gray':float(d.mean()),'max_gray':float(d.max()),'temporal_model_changed':False}
                            for precision in ['fp32','fp16']:
                                m=copy.deepcopy(single).to(dtype=torch.float32 if precision=='fp32' else torch.float16)
                                inp=(stack[:,-1:].to(m.base.quarter.front.first.weight.dtype),context.to(m.base.quarter.front.first.weight.dtype))
                                entry['graphs'].append(export_model(m,inp,out/f'day_current_folded_{precision}_rows32.onnx',
                                                                    ['current_raw','reference_thumb'],'display_gray'))
                        print('EXPORT',scene,len(entry['graphs']),flush=True)
                row=tests[0]
                for frame in range(120):
                    rr=dict(row,frame_id=frame)
                    stack=test.normalized_stack(rr,(0,0,1024,1280))[None].cuda()
                    context,_=test.context_for(rr); context=context[None].cuda()
                    expected=base.native(stack,context,box).float().clamp(0,1)*255
                    actual=F.avg_pool2d(optimized(stack,context),3,3)
                    d=(actual-expected).abs()
                    entry['equivalence_120frames'].append({'frame':frame,'mae_gray':float(d.mean()),'max_gray':float(d.max())})
                    half_result=F.avg_pool2d(optimized_half(stack.half(),context.half()).float(),3,3)
                    hd=(half_result-expected).abs()
                    entry['equivalence_120frames'][-1].update({'fp16_vs_fp32_mae_gray':float(hd.mean()),
                                                              'fp16_vs_fp32_max_gray':float(hd.max())})
                    if group=='day':
                        alt_expected=alt_base.native(stack,context,box).float().clamp(0,1)*255
                        sd=(F.avg_pool2d(single(stack[:,-1:],context),3,3)-alt_expected).abs()
                        entry['equivalence_120frames'][-1].update({'single_frame_mae_gray':float(sd.mean()),
                                                                  'single_frame_max_gray':float(sd.max())})
                        assert float(sd.max())<.01
                    if frame%30==0: print('EQUIVALENCE',scene,frame,float(d.max()),flush=True)
            entry['equivalence_max_gray']=max(maximum,max(v['max_gray'] for v in entry['equivalence_120frames']))
            assert entry['equivalence_max_gray']<.01, entry['equivalence_max_gray']
            (out/'manifest.json').write_text(json.dumps(entry,indent=2))
            manifest['scenes'][scene]=entry
            (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print('SCENE_DONE',scene,entry['equivalence_max_gray'],flush=True)
    print('DEPLOYMENT_PREPARATION_COMPLETE',flush=True)


if __name__=='__main__': main()
