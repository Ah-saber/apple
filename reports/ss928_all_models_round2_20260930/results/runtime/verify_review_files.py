"""File-level review manifest, exported graph checks, and decoded video frame counts."""
import hashlib,json,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-ALL-MODELS-ROUND2-20260930')
sys.path.insert(0,'/data/zhangbenzhuang/huawei_sr/runs/TASK-019-export-dependencies/site-packages')
import onnx
lock=(ROOT/'review_verification.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (ROOT/'review_file_verification.json').exists():
    print('REVIEW_ALREADY_COMPLETE',flush=True);sys.exit(0)
while not (ROOT/'targets_tiny_half1_day/signed_target_summary.json').exists() or not (ROOT/'targets_tiny_night_weak/signed_target_summary.json').exists() or not (ROOT/'night_tail_full_sequence/night_tail_full_sequence.json').exists():time.sleep(15)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
directories=['runtime','exact_layout_five_r2','exact_layout_night_r2','day_backup','speed_five_r2','speed_quarter_correct','speed_night_actual','speed_tail_corrected','float_speed_profile','adaptive_probe_correct','adaptive_probe_nonzero_r3','adaptive_correct_light','adaptive_correct_heavy','adaptive_nonzero_day','residual_eval_light','residual_eval_heavy','residual_nonzero_eval_day','half3_eval_light','half3_eval_heavy','half3_eval_day','night_motion_r2','night_motion_weak','targets_adaptive_light','targets_adaptive_heavy','targets_adaptive_day','targets_adaptive_nonzero_day','targets_residual_light','targets_residual_heavy','targets_residual_day','targets_half3_light','targets_half3_heavy','targets_half3_day','targets_night_strong','targets_night_weak','correct_fp32_eval_light','correct_fp32_eval_heavy','targets_fp32_light','targets_fp32_heavy']
directories+=['residual_'+g+'_'+k for g in ['light','heavy'] for k in ['half','quarter']]+['residual_nonzero_day_half']+['half3_'+g+'_3' for g in ['day','light','heavy']]
directories+=['day_backup_float','night_tail_full_sequence','targets_tiny_adaptive_light','targets_tiny_adaptive_heavy','targets_tiny_adaptive_day','targets_tiny_residual_day','targets_tiny_half3_light','targets_tiny_half3_heavy','targets_tiny_half3_day','targets_tiny_night_strong','targets_tiny_night_weak']
directories+=['half1_'+g+'_1' for g in ['day','light','heavy']]+['half1_eval_'+g for g in ['day','light','heavy']]+['targets_half1_'+g for g in ['day','light','heavy']]+['targets_tiny_half1_'+g for g in ['day','light','heavy']]
files=[];graphs=[];videos=[]
for directory in directories:
    parent=ROOT/directory
    assert parent.is_dir(),('unfinished directory',directory)
    for path in sorted(parent.rglob('*')):
        if not path.is_file():continue
        if directory=='speed_five_r2' and path.suffix=='.onnx' and '_quarter_' in path.name:continue
        if path.suffix not in ['.py','.json','.onnx','.mp4','.png'] and path.name!='best.pt':continue
        rel=str(path.relative_to(ROOT));files.append({'path':rel,'bytes':path.stat().st_size,'sha256':sha(path)})
        if path.suffix=='.onnx':
            model=onnx.load(str(path));onnx.checker.check_model(model)
            graphs.append({'path':rel,'operators':sorted({n.op_type for n in model.graph.node}),'checker_passed':True})
        if path.suffix=='.mp4':
            obj=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)]))['streams'][0]
            expected=60 if 'night_ordinary' in rel else 120
            assert int(obj['nb_read_frames'])==expected,(rel,obj)
            videos.append({'path':rel,'expected_real_frames':expected,**obj})
for name in ['INVALID_EARLY_WEATHER_CONFIG.json']:
    path=ROOT/name;files.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
report={'complete':True,'packaged':False,'pushed':False,'NPU_measured':False,'files':files,'graphs':graphs,'videos':videos,'limits':'only source-level graph checker and full video decode count; no SDK compile, calibration quantization or SS928 inference'}
(ROOT/'review_file_verification.json').write_text(json.dumps(report,indent=2))
(ROOT/'review_files.txt').write_text('\n'.join(v['path'] for v in files)+'\nreview_file_verification.json\n')
print('REVIEW_FILE_VERIFIED',len(files),'graphs',len(graphs),'videos',len(videos),flush=True)
