"""Learn GT motion without forcing wrong teacher target signals; Original spatial weights; matching body feature teacher and settled GT/static losses."""
import copy
import json
from collections import Counter
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from target_residual import ROOT,current_setup
from causal_detail import GuidanceView,reference_board
from compact_student import owner,front_feature,tile_forward,transfer_learned,rewrite,set_alpha,CurriculumConv
from streaming_ring_data import InputModel
from target_validation_long import validation
from target_validation_v3 import score,cohorts
from train_balanced_detail import permitted
from npu_graph_candidate import optimize
from run_reduced_v2 import read_gt
from gap_metrics_v5 import GapMetrics
from detail_progress import flattened,guards
from gt_tracks import save_gallery
from graph_contract import inspect,materialize_conv_aliases
import onnx
from settled_training9 import settle,center_temporal
import slim_body9
from signed_departure_data9 import apply as repair_departure

def previous(scene,base,count,kind,box):
    model=reference_board(base,count,kind,box,scene)
    if scene=='weather_medium':
        ck=torch.load(ROOT/('combined_masked_axial_student_'+scene)/'components.pt',map_location='cpu',weights_only=False)
        rewrite(model,'axial');transfer_learned(model,ck['core']['model'],'axial')
        transfer_learned(model,ck['temporal']['model'],'temporal');set_alpha(model,1)
    elif scene=='night_special':
        ck=torch.load(ROOT/('masked_front_student_'+scene)/'best.pt',map_location='cpu',weights_only=False)
        transfer_learned(model,ck['model'],'temporal')
    elif scene=='weather_heavy_c32':
        ck=torch.load(ROOT/('axial_guarded_student_'+scene)/'best.pt',map_location='cpu',weights_only=False)
        rewrite(model,'axial');transfer_learned(model,ck['model'],'axial');set_alpha(model,1)
    return model.eval().requires_grad_(False)

def fitted_modules(model,mode):return slim_body9.fitted_modules(model)

def transfer(model,state,mode):return slim_body9.transfer(model,state)

def departure_reasons(c,r):
    reasons=guards(c,r)
    if c['targets'].get('polarity_inverted',0)>r['targets'].get('polarity_inverted',0)+1e-6:
        reasons.append('overall_target_polarity')
    for key in ['old_position_error_gray','old_position_signed_ghost_gray']:
        if key in c['frames'] and key in r['frames'] and c['frames'][key]>r['frames'][key]+.005:
            reasons.append(key+'_strict_departure')
    return reasons

def run(scene,mode,steps):
    out=ROOT/('nativewidth9_'+mode+'_student_'+scene);out.mkdir(exist_ok=False)
    saved=torch.load(ROOT/('velocity9_cache_'+scene)/'tiles.pt',map_location='cpu',weights_only=False)
    values=saved['values'];signed_metadata=repair_departure(scene,values);settled_metadata=settle(values);raws=[v['raw'] for v in values]
    if len(values)!=len(raws):raise ValueError('Cache length mismatch')
    kind,config,df,box,count,base,_,profile=current_setup(scene)
    GuidanceView.sensor=False;val=GuidanceView(df(config,'val'),list(range(9-count,9)))
    row=next(r for r in val.records if r['scene_id']==scene)
    reference=previous(scene,base,count,kind,box(row))
    native=[v for v in values if v['geometry'][3]==1]
    if not native:raise ValueError('Native training tiles missing')
    plan=slim_body9.build_plan(reference,native,width_fraction=.75)
    plan.update(native_fit_tiles=len(native),fit_scope='Native training only; retained model covariance; no test samples')
    torch.save(plan,out/'pruning_plan.pt')
    student=slim_body9.rewrite(copy.deepcopy(reference).float(),plan)
    front=student.board.source.front.temporal_conv if kind else owner(student).front.first
    front.weight.requires_grad_(not bool(kind));dc=front.weight.detach().sum((1,2,3),keepdim=True).clone()
    for module in fitted_modules(student,mode):module.requires_grad_(True)
    for module in owner(student).body.modules():
        if isinstance(module,CurriculumConv):module.original.requires_grad_(False)
    set_alpha(student,1)

    keys=['gt_motion','raw','body_teacher','field','prior','teacher','gt','target','ghost','confidence','support']+['pair_'+k for k in ['raw','field','prior','gt','target','ghost','confidence']]
    rich=[];tiny_pool=[];ghost_pool=[]
    for i,(v,raw) in enumerate(zip(values,raws)):
        v['raw']=raw
        if v['regions'] or v['ghost'].abs().sum()>0:rich.append(i)
        if any(r['tiny'] for r in v['regions']):tiny_pool.append(i)
        if v['ghost'].abs().sum()>0:ghost_pool.append(i)
        support=torch.zeros_like(v['target'])
        for region in v['regions']:
            support.flatten()[region['target']]=1;support.flatten()[region['ring']]=1
        v['support']=support

        for key in keys:v[key]=v[key].clone()
        for region in v['regions']:
            for key in ['target','ring','gap']:region[key]=region[key].clone()
    shadow=copy.deepcopy(reference).float().requires_grad_(False)
    with torch.no_grad():
        for v in values:
            for prefix in ['', 'pair_']:
                feature=front_feature(shadow,v[prefix+'raw'].cuda())
                output,matched_body=tile_forward(shadow,feature,v[prefix+'field'].cuda(),v[prefix+'prior'].cuda())
                v[prefix+'teacher']=output.clamp(0,1).cpu()
                v[prefix+'body_teacher']=matched_body.cpu()
    del shadow
    keys.append('pair_teacher')
    ref=validation({'currentK':InputModel(reference,val,profile,kind)},config,val,[scene],box)[scene]['currentK']
    parameters=[p for p in student.parameters() if p.requires_grad]
    train_names={n for n,p in student.named_parameters() if p.requires_grad};initial={n:student.state_dict()[n].float().detach().clone() for n in train_names}
    optimizer=torch.optim.Adam(parameters,lr=.0001);rng=np.random.default_rng(1011);torch.manual_seed(1011)
    logs=[];best=score(ref);best_step=None
    def actual(record,alpha=1.):
        model=slim_body9.rewrite(previous(scene,base,count,kind,box(record)),plan)
        state=dict(student.state_dict())
        if alpha!=1.:
            for n in train_names:state[n]=initial[n]*(1-alpha)+state[n]*alpha
        transfer(model,state,mode)
        return model.eval().requires_grad_(False)
    def check_one(step,alpha):
        nonlocal best,best_step
        model=actual(row,alpha)
        metric=validation({'candidate':InputModel(model,val,profile,kind)},config,val,[scene],box)[scene]['candidate']
        reasons=departure_reasons(metric,ref);allowed=permitted(metric,ref) and not reasons;s=score(metric)
        ck=dict(scene=scene,mode=mode,step=step,count=count,profile=profile,
            model={k:((initial[k]*(1-alpha)+v*alpha) if k in train_names else v).detach().cpu().clone() for k,v in student.state_dict().items()},blend_alpha=alpha,
            validation=metric,reference_validation=ref,eligible=allowed,validation_rejected_by=reasons)
        torch.save(ck,out/'last.pt');torch.save(ck,out/('step_'+str(step)+'_alpha_'+str(alpha)+'.pt'))
        if allowed and (best_step is None or s<best-.001):best,best_step=s,step;torch.save(ck,out/'best.pt')
        logs.append(dict(step=step,blend_alpha=alpha,eligible=allowed,score=s,metrics=metric,rejected_by=reasons))
        (out/'training.json').write_text(json.dumps(dict(scene=scene,mode=mode,steps=steps,count=count,profile=profile,
            hidden_width_fraction=.75,native_fit_tiles=len(native),hidden_channel_plan=[{k:v for k,v in layer.items() if k!='regression'} for layer in plan['layers']],sampling='Every batch: four native and four scaled tiles; tiny/departure pools fall back within each scale group',native_tiles=sum(v['geometry'][3]==1 for v in values),parameters=sum(p.numel() for p in parameters),tiles=len(values),rich_tiles=len(rich),tiny_tiles=len(tiny_pool),departure_tiles=len(ghost_pool),
            validation_trials=logs,reference_validation=ref,teacher_scope='Only static support outside 15x15 expanded GT target/departure masks',
            data_change='Matched reference on both causal crops; relative signed/absolute departure, static temporal and weak-structure constraints; total front DC preserved while spatial weights can redistribute' , baseline='R8 axial combo for medium; R8 front for special; R7 axial for C32; original retained for others', trainable_scope='Three-quarter hidden channels; native training covariance; front/head/body/tail/output; night temporal front frozen; reference rows selected and frozen; alpha1 only' ,
            reference_frozen_after_rewrite=True,training_teacher=saved['metadata'],body_teacher_recomputed='Both current and pair use matched retained FP32 reference; old base features replaced',departure_data_repair=signed_metadata,settled_departure=settled_metadata,static_temporal_drift_removed=True,total_front_DC_preserved=True,additional_inference_operations=0,
            additional_input_channels=0,test_used_for_selection=False,NPU_measured=False,SDK_compiled=False,packaged=False,pushed=False),indent=2))
        print('NATIVEWIDTH9_VALIDATION',scene,mode,step,alpha,allowed,s,'reference',score(ref),reasons,flush=True)
        del model
    def check(step):
        for alpha in [1.]:check_one(step,alpha)
    hardness=np.ones(len(values),dtype=np.float64)
    def draw(pool,n):
        pool=np.asarray(pool or list(range(len(values))));w=hardness[pool];return rng.choice(pool,n,p=w/w.sum())
    native_pool=[i for i,v in enumerate(values) if v['geometry'][3]==1]
    scaled_pool=[i for i,v in enumerate(values) if v['geometry'][3]>1]
    def subset(pool,selected):return [i for i in pool if i in set(selected)]
    native_tiny=subset(native_pool,tiny_pool);native_ghost=subset(native_pool,ghost_pool)
    scaled_tiny=subset(scaled_pool,tiny_pool);scaled_ghost=subset(scaled_pool,ghost_pool)
    if not native_pool or not scaled_pool:raise ValueError('Both native and scaled training data required')
    for step in range(1,steps+1):
        # Four native + four scaled examples. Empty rare pools fall back within their scale group.
        chosen=np.concatenate([draw(native_pool,2),draw(native_tiny or native_pool,1),draw(native_ghost or native_pool,1),
                               draw(scaled_pool,2),draw(scaled_tiny or scaled_pool,1),draw(scaled_ghost or scaled_pool,1)])
        batch={key:torch.cat([values[i][key] for i in chosen]).cuda() for key in keys}
        batch=slim_body9.batch_for(student,batch,plan)
        feature=front_feature(student,batch['raw'])
        pred,body=tile_forward(student,feature.float(),batch['field'],batch['prior']);pred=pred.clamp(0,1)
        pair_front=front_feature(student,batch['pair_raw'])
        pair_pred,_=tile_forward(student,pair_front.float(),batch['pair_field'],batch['pair_prior']);pair_pred=pair_pred.clamp(0,1)
        error=(pred-batch['gt'])*255;target=batch['target'].abs();ghost=batch['ghost'].abs()
        inner=(slice(None),slice(None),slice(12,-12),slice(12,-12))
        def weighted(a,w):
            a,w=a[inner],w[inner];return (a*w).sum()/w.sum().clamp_min(1e-8)
        static=(1-F.max_pool2d(((target+ghost+batch['support'])>0).float(),15,1,7)).clamp(0,1)*batch['confidence']
        foreground=weighted(error.abs(),target)
        depart=weighted((error-F.avg_pool2d(error,21,1,10)).abs(),ghost)
        trust=weighted((pred-batch['teacher']).abs()*255,static)
        mask=F.avg_pool2d(static,2)[...,6:-6,6:-6]
        distill=(body-batch['body_teacher']).abs().mean(1,keepdim=True)[...,6:-6,6:-6]*255
        distill=(distill*mask).sum()/mask.sum().clamp_min(1e-8)
        rings=[];tinies=[];polarities=[];anchors=[];target_errors=[]
        for n,i in enumerate(chosen):
            p=pred[n].flatten();g=batch['gt'][n].flatten();tile_errors=[]
            for r in values[i]['regions']:
                gap=p[r['target']].mean()-torch.quantile(p[r['ring']],.5);term=(gap-r['gap']).abs()*255/(1+r['gap'].abs()*255/15)
                rings.append(term);tile_errors.append(term.detach())
                polarities.append(F.relu(.25*r['gap'].abs()-r['gap'].sign()*gap)*255)
                anchors.append((torch.quantile(p[r['ring']],.5)-torch.quantile(g[r['ring']],.5)).abs()*255)
                target_errors.append((p[r['target']]-g[r['target']]).abs().mean()*255)

                if r['tiny']:tinies.append(term)
            gh=batch['ghost'][n].abs()
            if gh.sum()>0:
                centered=error[n]-F.avg_pool2d(error[n:n+1],21,1,10)[0]
                tile_errors.append((((error[n].abs()+centered.abs())*gh).sum()/gh.sum().clamp_min(1e-8)).detach())
            if tile_errors:
                hardness[i]=.95*hardness[i]+.05*min(12.,1.+float(torch.stack(tile_errors).mean())/3.)
        ring=torch.stack(rings).mean() if rings else pred.sum()*0
        tiny=torch.stack(tinies).mean() if tinies else pred.sum()*0
        def average(parts):return torch.stack(parts).mean() if parts else pred.sum()*0
        absolute_depart=weighted(error.abs(),ghost)
        local_gt=(batch['gt']-F.avg_pool2d(batch['gt'],7,1,3))*255
        weak_support=static*((local_gt.abs()>=1)&(local_gt.abs()<3)).float()
        weak_structure=weighted((error-F.avg_pool2d(error,7,1,3)).abs(),weak_support)
        delta_error=((pair_pred-pred)-(batch['pair_gt']-batch['gt']))*255
        occupied=((target+ghost+batch['support']+batch['pair_target'].abs()+batch['pair_ghost'].abs())>0).float()
        pair_static=(1-F.max_pool2d(occupied,15,1,7)).clamp(0,1)*torch.minimum(batch['confidence'],batch['pair_confidence'])
        pair_static*=1-F.max_pool2d(batch['gt_motion'],7,1,3)
        delta_static=center_temporal(delta_error,pair_static)
        temporal_static=weighted(delta_static.abs(),pair_static)
        temporal_motion=weighted(delta_error.abs(),target+ghost+batch['pair_target'].abs()+batch['pair_ghost'].abs())
        centered=error-F.avg_pool2d(F.pad(error,(10,10,10,10),mode='reflect'),21,1)
        reference_error=(batch['teacher']-batch['gt'])*255
        reference_centered=reference_error-F.avg_pool2d(F.pad(reference_error,(10,10,10,10),mode='reflect'),21,1)
        sign=batch['ghost'].sign()
        signed_depart=weighted(F.relu(centered*sign),ghost)
        opposite_depart=weighted(F.relu(-centered*sign),ghost)
        opposite_guard=weighted(F.relu(F.relu(-centered*sign)-F.relu(-reference_centered*sign)),ghost)
        depart_guard=weighted(F.relu(centered.abs()-reference_centered.abs())+F.relu(F.relu(centered*sign)-F.relu(reference_centered*sign)),ghost)
        reference_delta=((batch['pair_teacher']-batch['teacher'])-(batch['pair_gt']-batch['gt']))*255
        reference_delta=center_temporal(reference_delta,pair_static)
        temporal_guard=weighted(F.relu(delta_static.abs()-reference_delta.abs()),pair_static)
        reference_weak=reference_error-F.avg_pool2d(reference_error,7,1,3)
        weak_guard=weighted(F.relu((error-F.avg_pool2d(error,7,1,3)).abs()-reference_weak.abs()),weak_support)
        loss=80*(depart_guard+opposite_guard)+80*temporal_guard+40*weak_guard+32*(signed_depart+opposite_depart)+40*temporal_static+16*temporal_motion+4*foreground+8*ring+24*tiny+16*average(polarities)+4*average(anchors)+4*average(target_errors)+32*depart+8*absolute_depart+40*trust+10*distill+30*weak_structure+.25*error[inner].abs().mean()

        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5);optimizer.step()
        with torch.no_grad():front.weight.add_((dc-front.weight.sum((1,2,3),keepdim=True))/(front.weight.shape[1]*front.weight.shape[2]*front.weight.shape[3]))
        for group in optimizer.param_groups:group['lr']=.00001+.00009*(1-step/steps)
        if step%500==0:print('NATIVEWIDTH9_TRAIN',scene,mode,step,float(loss),flush=True)
        if step in [1000,steps]:check(step)
    if best_step is None:
        (out/'summary.json').write_text(json.dumps(dict(scene=scene,mode=mode,eligible_for_review=False,
            reason='No validation score improvement within quality and strict departure guards',
            NPU_measured=False,packaged=False,pushed=False),indent=2));return
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);student.load_state_dict(ck['model'])
    del values,raws,val,reference,optimizer,parameters,saved
    data=GuidanceView(df(config,'test'),list(range(9-count,9)));record=next(r for r in data.records if r['scene_id']==scene)
    current,_=optimize(previous(scene,base,count,kind,box(record)),kind);candidate,changes=optimize(actual(record),kind)
    boards={'currentK':current,'candidate':candidate}
    from ir_sr.sequence_normalization import regional_key
    norm=data.base.sequence_normalization;key=regional_key(record) if norm.regional else record['domain']+'/'+record['sequence_id']
    frames=int(norm.by_key[key]['frames']);metric=GapMetrics(record['eval_crop_tlhw'],boards)
    metric.prepare_background([read_gt(config,record,f) for f in range(frames)])
    with torch.inference_mode():
        for f in range(frames):
            rr=dict(record,frame_id=f);data.current_record,data.current_crop,data.cache=rr,(0,0,1024,1280),{}
            raw=data.inputs(profile)[None].cuda();context,_=data.context_for(rr);context=context[None].cuda()
            if not kind:raw,context=raw.half(),context.half()
            outputs={name:model(raw,context) for name,model in boards.items()}
            for output in outputs.values():
                if output.shape!=(1,1,3072,3840) or output.dtype!=torch.float16:raise ValueError('Output contract changed')
            metric.update(f,read_gt(config,record,f),{n:a[0,0,::3,::3].float().cpu().numpy() for n,a in outputs.items()})
    report=metric.report();report.update(cohorts=cohorts(report['targets'],boards))
    reasons=departure_reasons(flattened(report,'candidate'),flattened(report,'currentK'))
    report.update(scene=scene,mode=mode,checkpoint_step=best_step,blend_alpha=ck['blend_alpha'],full_guards=not reasons,rejected_by=reasons,
        eligible_for_review=not reasons,channels=count,profile=profile,graph_changes=changes,
        additional_inference_operations=0,test_used_for_selection=False,NPU_measured=False,SDK_compiled=False,
        automatically_adopted=False,packaged=False,pushed=False)
    if not reasons:
        for name,model in [('control',current),('candidate',candidate)]:
            path=out/(name+'.onnx');model.requires_grad_(False)
            torch.onnx.export(model,(raw,context),str(path),input_names=['raw_stack','reference_thumb'],output_names=['display_gray'],opset_version=17,dynamo=False)
            graph=onnx.load(path);materialize_conv_aliases(graph);onnx.save(graph,path);g=inspect(graph)
            g['node_counts']=dict(Counter(n.op_type for n in graph.graph.node));report.setdefault('graphs',{})[name]=g
    (out/'full_field_results.json').write_text(json.dumps(report,indent=2));save_gallery(metric,out)
    (out/'summary.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ['frames','targets']},indent=2))
    if not reasons:
        from review_nativewidth9_wide import run as render_review
        render_review(scene,'best.pt')
    print('NATIVEWIDTH9_COMPLETE',scene,mode,not reasons,reasons,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--mode',choices=['spatial','core'],required=True);p.add_argument('--steps',type=int,default=6000);a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    run(a.scene,a.mode,a.steps)


