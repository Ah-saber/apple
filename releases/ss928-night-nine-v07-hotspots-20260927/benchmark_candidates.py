import argparse,fcntl,json,sys,time,gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from operator_candidates import Statistics,OutputPermutation,CandidateSystem
pkg=Path('/tmp/ss928_release_verification_20260926/ss928-night-nine-system-speed-20260926')
sys.path.insert(0,str(pkg/'runtime'))
from load_system_release import load_release
parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
lock=Path('/data/zhangbenzhuang/huawei_sr/runs/TASK-019-gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(2.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
scene=args.scene;frame=20 if scene=='ordinary' else 60
v=np.load(f'/data/zhangbenzhuang/huawei_sr/runs/SS928-NIGHT-NINE-SPEED-20260925/TRIMMED-TEST-VECTORS/{scene}_frame_{frame}.npz')
x=torch.from_numpy(v['nine_raw']).cuda();ctx=torch.from_numpy(v['reference_thumb']).cuda()
base,_=load_release(scene,'front_f32')
def delta(a,b):
    a,b=a.float(),b.float();d=(a-b).abs();return {'mean_gray':float(d.mean()),'max_gray':float(d.max()),'rounded_byte_changed_fraction':float((a.round()!=b.round()).float().mean()),'rounded_byte_max_error':float((a.round()-b.round()).abs().max())}
def timed(model,inputs,warm=20,n=100):
    for _ in range(warm):model(*inputs)
    torch.cuda.synchronize();times=[]
    for _ in range(3):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(n):model(*inputs)
        end.record();end.synchronize();times.append(start.elapsed_time(end)/n)
    return {'three_batch_mean_ms':times,'mean_ms':float(np.mean(times)),'warmup':warm,'runs_per_batch':n}
report={'scene':scene,'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,'TF32':False,'NPU_measured':False,'micro':[],'full':[]}
def save():Path(args.output).write_text(json.dumps(report,indent=2))
with torch.inference_mode():
    # Real complete-size intermediate, not a small tensor speed claim.
    c,m=base.core,base.core.model;features=m.body(m.head(c.half_input(base.front(x))));ref=m.global_reference.encode(c.half_input(ctx));ref=m.global_reference.project(ref+ref.mean((-2,-1),keepdim=True));ref=torch.nn.functional.interpolate(ref,size=features.shape[-2:],mode='bilinear',align_corners=False);packed=m.upsample[0](m.tail(features+ref));expected=base(x,ctx)
    stats_expected=Statistics(base.front.stats_weight,'dense')(x);out_expected=torch.nn.functional.pixel_shuffle(packed,6)
    for mode in ('dense','temporal_first','temporal_f32','pack_first'):
        model=Statistics(base.front.stats_weight,mode);out=model(x);diff=(out.float()-stats_expected.float()).abs();row={'kind':'statistics','mode':mode,'input_shape':list(x.shape),'output_shape':list(out.shape),'mean_normalized_error':float(diff.mean()),'max_normalized_error':float(diff.max()),'eager':timed(model,(x,))};report['micro'].append(row);save();print(scene,'micro',mode,row,flush=True)
    for mode in ('shuffle6','reshape','deconv6','deconv2_shuffle3','deconv3_shuffle2','shuffle2_3','shuffle3_2'):
        model=OutputPermutation(mode).cuda().half();out=model(packed);assert torch.equal(out,out_expected),mode
        row={'kind':'output','mode':mode,'input_shape':list(packed.shape),'full_output_shape':list(out.shape),'bitexact':True,'eager':timed(model,(packed,))};report['micro'].append(row);save();print(scene,'micro',mode,row,flush=True)
    specs=[('front_f32',{}),('stats_temporal',{'statistics':'temporal_first'}),('stats_temporal_f32',{'statistics':'temporal_f32'}),('stats_pack',{'statistics':'pack_first'}),('output_deconv6',{'output':'deconv6'}),('output_deconv2_shuffle3',{'output':'deconv2_shuffle3'}),('output_deconv3_shuffle2',{'output':'deconv3_shuffle2'}),('output_shuffle2_3',{'output':'shuffle2_3'}),('output_shuffle3_2',{'output':'shuffle3_2'}),('output_reshape',{'output':'reshape'}),('input_nchw_half',{'input_half':True}),('output_half',{'output_half':True}),('scale_packed_f32',{'scale':'packed_f32'}),('scale_packed_half',{'scale':'packed_half'})]
    for name,kw in specs:
        model=CandidateSystem(base,**kw).eval();inputs=(x.half() if kw.get('input_half') else x,ctx);out=model(*inputs);row={'name':name,'options':kw,'source_delta':delta(out,expected),'eager':timed(model,inputs)};report['full'].append(row);save();print(scene,'full',name,row,flush=True)
        if name in ('front_f32','stats_pack','output_deconv6','output_shuffle2_3','input_nchw_half','output_half','scale_packed_f32'):
            torch.compiler.reset();torch._inductor.config.triton.cudagraphs=False
            compiled=torch.compile(model,fullgraph=True);row['compiled']=timed(compiled,inputs,30,200);row['compiled_vs_own_source']=delta(compiled(*inputs),out);save();print(scene,'compiled',name,row['compiled'],flush=True);del compiled;torch.compiler.reset();gc.collect();torch.cuda.empty_cache()
        del model,out
print('COMPLETE',args.output,flush=True)
