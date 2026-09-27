import sys,json,hashlib,collections
from pathlib import Path
import numpy as np
sys.path.append('/data/zhangbenzhuang/miniconda3/envs/bfstvsr/lib/python3.10/site-packages')
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnx.reference import ReferenceEvaluator
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from rewrite_hotspots import rewrite_statistics,rewrite_output
from rewrite_interfaces import change
root=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-FOLLOWUP-20260927');out=root/'onnx';out.mkdir(exist_ok=True);checks=[];manifest=[]
def make_model(nodes,initializers,shape,output_shape,dtype):
    g=helper.make_graph(nodes,'independent_control',[helper.make_tensor_value_info('input',dtype,shape)],[helper.make_tensor_value_info('output',dtype,output_shape)],initializers)
    m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)]);m.ir_version=8;return m

def evaluate(m,x):
    onnx.checker.check_model(m);return ReferenceEvaluator(m).run(None,{'input':x})[0]
def save(m,name):
    path=out/name;onnx.checker.check_model(m);onnx.save(m,path);manifest.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nodes':dict(collections.Counter(n.op_type for n in m.graph.node))})
rng=np.random.default_rng(20260927)
for npdtype,typ in [(np.float32,TensorProto.FLOAT),(np.float16,TensorProto.FLOAT16)]:
    suffix='fp32' if typ==TensorProto.FLOAT else 'fp16';shape=[1,36,3,5];x=rng.normal(0,.5,shape).astype(npdtype)
    old=make_model([helper.make_node('DepthToSpace',['input'],['output'],blocksize=6,mode='CRD')],[],shape,[1,1,18,30],typ)
    expected=x.reshape(1,1,6,6,3,5).transpose(0,1,4,2,5,3).reshape(1,1,18,30)
    assert np.array_equal(evaluate(old,x),expected)
    for mode in ('reshape','deconv6','deconv2_shuffle3','deconv3_shuffle2','shuffle2_3','shuffle3_2'):
        new=rewrite_output(old,mode);actual=evaluate(new,x);assert np.array_equal(actual,expected),(mode,suffix)
        checks.append({'kind':'output','dtype':suffix,'mode':mode,'bitexact':True})
        full=make_model([helper.make_node('DepthToSpace',['input'],['output'],blocksize=6,mode='CRD')],[],[1,36,512,640],[1,1,3072,3840],typ)
        save(rewrite_output(full,mode),f'micro_output_{mode}_{suffix}.onnx')
    save(make_model([helper.make_node('DepthToSpace',['input'],['output'],blocksize=6,mode='CRD')],[],[1,36,512,640],[1,1,3072,3840],typ),f'micro_output_baseline_{suffix}.onnx')
    temporal=np.zeros((3,9),npdtype);temporal[0,8]=1;temporal[1,:5]=120/727;temporal[1,5]=127/727;temporal[2,6:8]=123/373;temporal[2,8]=127/373
    weights=np.zeros((12,9,2,2),npdtype)
    for g in range(3):
        for dy in range(2):
            for dx in range(2):weights[4*g+2*dy+dx,:,dy,dx]=temporal[g]
    def stats_model(h,w):return make_model([helper.make_node('Conv',['input','stats'],['output'],kernel_shape=[2,2],strides=[2,2])],[numpy_helper.from_array(weights,'stats')],[1,9,h,w],[1,12,h//2,w//2],typ)
    for mode in ('temporal_first','temporal_f32','pack_first'):
        old=stats_model(6,10);new=rewrite_statistics(old,mode)
        for feed in ('constant','noise','dark','ramp'):
            x=(np.ones((1,9,6,10))*.3 if feed=='constant' else rng.random((1,9,6,10))*(.03 if feed=='dark' else 1)).astype(npdtype)
            if feed=='ramp':x=np.linspace(0,1,x.size,dtype=npdtype).reshape(x.shape)
            expected=evaluate(old,x);actual=evaluate(new,x);error=np.abs(actual.astype(np.float32)-expected.astype(np.float32));assert error.max()<.001,(mode,suffix,error.max())
            checks.append({'kind':'statistics','dtype':suffix,'mode':mode,'feed':feed,'max_normalized_error':float(error.max()),'mean_normalized_error':float(error.mean())})
        save(rewrite_statistics(stats_model(1024,1280),mode),f'micro_statistics_{mode}_{suffix}.onnx')
    save(stats_model(1024,1280),f'micro_statistics_baseline_{suffix}.onnx')
# Interface/scale independent controls, include Clip-before-shuffle native pattern.
for clip_before in (False,True):
    nodes=[helper.make_node('Cast',['input'],['half'],to=TensorProto.FLOAT16)]
    init=[numpy_helper.from_array(np.asarray(0,np.float16),'zero_half'),numpy_helper.from_array(np.asarray(1,np.float16),'one_half'),numpy_helper.from_array(np.asarray(0,np.float32),'zero'),numpy_helper.from_array(np.asarray(1,np.float32),'one'),numpy_helper.from_array(np.asarray(255,np.float32),'scale')]
    if clip_before:nodes.extend([helper.make_node('Clip',['half','zero_half','one_half'],['clip_half']),helper.make_node('DepthToSpace',['clip_half'],['spatial'],blocksize=6,mode='CRD'),helper.make_node('Cast',['spatial'],['normalized'],to=TensorProto.FLOAT)])
    else:nodes.extend([helper.make_node('DepthToSpace',['half'],['spatial'],blocksize=6,mode='CRD'),helper.make_node('Cast',['spatial'],['raw_float'],to=TensorProto.FLOAT),helper.make_node('Clip',['raw_float','zero','one'],['normalized'])])
    nodes.append(helper.make_node('Mul',['normalized','scale'],['output']))
    m=make_model(nodes,init,[1,36,3,5],[1,1,18,30],TensorProto.FLOAT);x=rng.normal(.5,.3,(1,36,3,5)).astype(np.float32);expected=evaluate(m,x)
    for mode in ('scale_packed_f32','output_half'):
        new=change(m,mode);actual=evaluate(new,x);target=expected.astype(np.float16) if mode=='output_half' else expected;assert np.array_equal(actual,target)
        checks.append({'kind':'interface','mode':mode,'clip_before_shuffle':clip_before,'bitexact_to_declared_dtype':True})
# Whole source graphs: these keep source Resize. Deployment should rewrite actual native front_f32.
pkg=Path('/tmp/ss928_release_verification_20260926/ss928-night-nine-system-speed-20260926')
for scene in ('ordinary','special'):
    model=onnx.load(Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-SYSTEM-SPEED-20260926/source_onnx')/f'{scene}_balanced.onnx')
    for mode in ('temporal_first','temporal_f32','pack_first'):save(rewrite_statistics(model,mode),f'{scene}_statistics_{mode}.onnx')
    for mode in ('reshape','deconv6','deconv2_shuffle3','deconv3_shuffle2','shuffle2_3','shuffle3_2'):save(rewrite_output(model,mode),f'{scene}_output_{mode}.onnx')
    for mode in ('input_nchw_half','output_half','scale_packed_f32'):save(change(model,mode),f'{scene}_{mode}.onnx')
(root/'graph_checks.json').write_text(json.dumps({'independent_controls':checks,'NPU_measured':False},indent=2));(root/'source_export_manifest.json').write_text(json.dumps({'models':manifest,'SDK_compiled':False,'NPU_measured':False},indent=2));print('COMPLETE',len(checks),'controls,',len(manifest),'graphs',flush=True)
