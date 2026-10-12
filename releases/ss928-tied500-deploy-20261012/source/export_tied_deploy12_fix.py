"""Export the frozen user-reviewed TIED500 complete graphs and matched real inputs."""
import argparse,json,copy,gc,shutil,torch,numpy as np,cv2
from pathlib import Path
from collections import Counter
from target_residual import ROOT,current_setup
import onnx,onnxruntime as ort
from causal_detail import GuidanceView
from tieddecode_model11 import boards
from run_reduced_v2 import read_gt

DEST=ROOT/'TIED500-DEPLOY-20261012-FIX'

def describe(v):
    t=v.type.tensor_type;shape=[d.dim_value for d in t.shape.dim]
    size={onnx.TensorProto.FLOAT16:2,onnx.TensorProto.FLOAT:4}.get(t.elem_type)
    return dict(name=v.name,shape=shape,dtype=onnx.TensorProto.DataType.Name(t.elem_type),bytes=int(np.prod(shape))*size if size else None)

def run(scene):
    out=DEST/scene;out.mkdir(parents=True,exist_ok=False)
    kind,config,df,box,count,base,original,profile=current_setup(scene)
    GuidanceView.sensor=False;test=GuidanceView(df(config,'test'),list(range(9-count,9)))
    record=next(r for r in test.records if r['scene_id']==scene)
    quiet,candidate=boards(scene,base,count,kind,box(record),500)
    retained=candidate.reference
    del quiet,base,original;gc.collect();torch.cuda.empty_cache()
    models={'retained_original':retained,'tied500':candidate}
    train=GuidanceView(df(config,'train'),list(range(9-count,9)))
    rows=[r for r in train.records if r['scene_id']==scene]
    unique={ (r['domain'],r['sequence_id'],int(r['frame_id'])):r for r in rows if int(r['frame_id'])>=8 }
    ordered=[unique[k] for k in sorted(unique)]
    if len(ordered)<4:raise ValueError('Four distinct actual training frames required')
    selected=[ordered[i] for i in np.linspace(0,len(ordered)-1,4).astype(int)]
    def inputs(data,row):
        data.current_record,data.current_crop,data.cache=dict(row),(0,0,1024,1280),{}
        raw=data.inputs(profile)[None].cuda()[:,-candidate.source_count:].half()
        context,_=data.context_for(row);context=context[None].cuda().half()
        return raw,context
    calib=[]
    for i,row in enumerate(selected):
        raw,context=inputs(train,row)
        name='calibration_%02d.npz'%i
        np.savez_compressed(out/name,raw_stack=raw.cpu().numpy(),reference_thumb=context.cpu().numpy())
        calib.append(dict(file=name,split='train',sample=row,input_frames=candidate.source_count,profile=profile))
    smoke=dict(record,frame_id=37)
    raw,context=inputs(test,smoke)
    np.savez_compressed(out/'smoke_inputs.npz',raw_stack=raw.cpu().numpy(),reference_thumb=context.cpu().numpy())
    gt=read_gt(config,smoke,37)
    if gt.shape!=(1024,1280):raise ValueError('Full original GT expected')
    cv2.imwrite(str(out/'smoke_GT.png'),np.rint(np.clip(gt,0,255)).astype(np.uint8))
    contracts={}
    from graph_contract import materialize_conv_aliases
    for role,model in models.items():
        path=out/(role+'.onnx')
        with torch.inference_mode():
            direct=model(raw,context).cpu().numpy()
            torch.onnx.export(model,(raw,context),str(path),input_names=['raw_stack','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
        graph=onnx.load(path);materialize_conv_aliases(graph);onnx.checker.check_model(graph);onnx.save(graph,path)
        session_options=ort.SessionOptions();session_options.intra_op_num_threads=2;session_options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=session_options,providers=['CPUExecutionProvider'])
        pc=session.run(None,{'raw_stack':raw.cpu().numpy(),'reference_thumb':context.cpu().numpy()})[0]
        if pc.shape!=(1,1,3072,3840) or pc.dtype!=np.float16 or not np.isfinite(pc).all():raise ValueError('Full finite FP16 output required')
        np.savez_compressed(out/(role+'_pc_output.npz'),display_gray=pc)
        delta=np.abs(pc.astype(np.float32)-direct.astype(np.float32))
        contracts[role]=dict(file=path.name,inputs=[describe(v) for v in graph.graph.input],outputs=[describe(v) for v in graph.graph.output],
            opset=17,operators=dict(Counter(n.op_type for n in graph.graph.node)),onnx_checker_passed=True,PC_backend='onnxruntime CPU',PC_output=role+'_pc_output.npz',
            direct_PyTorch_vs_PC_MAE_gray=float(delta.mean()),direct_PyTorch_vs_PC_max_gray=float(delta.max()),
            smoke_frame=37,weights_embedded=True,SDK_compiled=False,NPU_measured=False)
        del session,pc,direct,delta,graph;gc.collect()
        print('TIEDDEPLOY12_GRAPH',scene,role,json.dumps(contracts[role]),flush=True)
    checkpoint=ROOT/('tieddecode11_weights_'+scene)/'last_step500.pt'
    shutil.copy2(checkpoint,out/'branch_step500.pt')
    for name in ['GPU-TIMING.json','FULL-QUALITY.json','summary.json']:
        shutil.copy2(ROOT/'TIEDDECODE500-USER-REVIEW-20261012-FIX'/scene/name,out/('review_'+name))
    summary=dict(scene=scene,status='ready_for_external_SDK_and_board_measurement',checkpoint='tieddecode11 fixed500',training_repeated=False,
        input_frames=candidate.source_count,profile=profile,graphs=contracts,calibration=calib,smoke=dict(file='smoke_inputs.npz',split='test',sample=smoke,GT='smoke_GT.png',metric_roi_tlhw=record['eval_crop_tlhw'],GT_full_frame=True),
        checkpoint_branch_only=True,ONNX_is_self_contained=True,quality_qualified=False,SDK_compiled=False,NPU_measured=False)
    (out/'manifest.json').write_text(json.dumps(summary,indent=2));print('TIEDDEPLOY12_DONE',scene,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);run(p.parse_args().scene)
