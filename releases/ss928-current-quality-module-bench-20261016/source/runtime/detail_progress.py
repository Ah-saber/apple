"""Compact progress and full-sequence guards; no hashes or model reruns."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCENES=['night_ordinary','day_normal','weather_light','weather_medium','weather_heavy','weather_heavy_c32','night_special']

def flattened(summary,name):
    m=summary['summary'][name]
    return dict(frames={k:v['mean'] for k,v in m['frame_metrics'].items()},targets={k:v['mean'] for k,v in m['target_metrics'].items()},cohorts=summary.get('cohorts',{}).get(name,{}))

def guards(c,r):
    reasons=[]
    if c['frames']['psnr_db'] < r['frames']['psnr_db']-.05:reasons.append('PSNR')
    for k,scale,margin in [('static_temporal_error_gray',1.1,.02),('weak_structure_error_gray',1.05,.02),('old_position_signed_ghost_gray',1,.02),('old_position_error_gray',1.05,.02)]:
        if k in c['frames'] and k in r['frames'] and c['frames'][k]>r['frames'][k]*scale+margin:reasons.append(k)
    if c['targets'] and r['targets'] and c['targets']['gap_error_gray']>r['targets']['gap_error_gray']+.02:reasons.append('gap')
    for group in ['tiny','weak']:
        a,b=c['cohorts'].get(group,{}),r['cohorts'].get(group,{})
        if a.get('count') and b.get('count'):
            if a['gap_error_gray']>b['gap_error_gray']+.05:reasons.append(group+'_gap')
            if a['polarity_inverted']>b['polarity_inverted']+1e-6:reasons.append(group+'_inversion')
    return reasons

def collect():
    result={}
    for batch,family in [('CAUSAL-DETAIL','causal_detail_'),('SENSOR-DETAIL','sensor_detail_'),('NATIVE-DETAIL','native_detail_'),('MATCHED-DETAIL','matched_detail_'),('TINY-MATCHED','tiny_matched_'),('BALANCED-DETAIL','balanced_detail_'),('CONTEXTUAL-DETAIL','contextual_detail_'),('DETAIL-GUARD','guarded_detail_'),('POINT-FIELD','point_field_'),('QUICK-FIELD','quick_field_'),('GPU-FIELD','gpu_field_'),('GPU-LEGACY','gpu_legacy_')]:
        statepath=ROOT/(batch+'-STATE.json')
        state=json.loads(statepath.read_text()) if statepath.exists() else {}
        rows=[]
        for scene in SCENES:
            out=ROOT/(family+scene); p=out/'summary.json'
            if not p.exists():continue
            s=json.loads(p.read_text()); row=dict(scene=scene)
            if 'summary' in s:
                r,c=flattened(s,'currentK'),flattened(s,'candidate'); reasons=guards(c,r)
                row.update(reference=r,candidate=c,full_guards=not reasons,rejected_by=reasons,timing=s.get('timing'),CPU_complete_RAW=s.get('CPU_complete_RAW'),serial_chain=s.get('serial_chain'))
                row['eligible_for_review']=not reasons and (s.get('validation_improved',False) or bool(s.get('validation_selected')))
            else:row.update(validation_improved=False,skipped=s.get('skipped',False),reason=s.get('reason','No validation improvement within guards'))
            native=ROOT/('native_targets_'+scene)/'summary.json'
            if family=='native_detail_' and native.exists():row['native_training_additions']=json.loads(native.read_text()).get('native_tiny_observations_added')
            rows.append(row)
        result[batch]=dict(status=state.get('status'),current_job=state.get('current_job',state.get('current_scene')),completed=len(rows),scenes=rows)
    (ROOT/'DETAIL-PROGRESS.json').write_text(json.dumps(result,indent=2))
    for batch,d in result.items():
        print(batch,d['status'],d['current_job'],'completed',d['completed'])
        for r in d['scenes']:
            if 'candidate' in r:
                a,b=r['reference'],r['candidate']
                print(r['scene'],'review',r['eligible_for_review'],'gap',a['targets'].get('gap_error_gray'),b['targets'].get('gap_error_gray'),'tiny',b['cohorts'].get('tiny'),'ghost',a['frames'].get('old_position_signed_ghost_gray'),b['frames'].get('old_position_signed_ghost_gray'),'rejected',r['rejected_by'],'CPU',r.get('CPU_complete_RAW'))
            else:print(r['scene'],r.get('reason'),'native_added',r.get('native_training_additions'))
    return result

if __name__=='__main__':
    import contextlib,io,sys
    if '--brief' in sys.argv:
        with contextlib.redirect_stdout(io.StringIO()):result=collect()
        for name,entry in result.items():
            print(name,entry['status'],entry['current_job'],'completed',entry['completed'],'review',[r['scene'] for r in entry['scenes'] if r.get('eligible_for_review')])
    else:collect()









