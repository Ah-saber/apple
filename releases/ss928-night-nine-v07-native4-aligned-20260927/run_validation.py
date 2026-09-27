"""Run required validation serially on the leased GPU and preserve individual logs."""
import argparse,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['quality','export','media'],required=True);a=p.parse_args();base=Path(__file__).parent;root=Path('/data/zhangbenzhuang/huawei_sr');run=root/'runs/SS928-V07-NATIVE4-20260927';py='/data/zhangbenzhuang/miniconda3/envs/test/bin/python';os.environ['CUDA_VISIBLE_DEVICES']='GPU-655fc72e-3e1f-236c-ee8b-5ee23e9bd264';jobs=[]
fast=['previous_combo','combo_aligned16_nearest','combo_input9_half_aligned16_nearest','combo_input16_half_native_stats_aligned16_fixed','combo_project_after_resize_aligned16_nearest','native4_linear_ref_dither_float_trained','combo_input9_half_scaled9_exact','combo_input16_half_native_stats_scaled9_exact']
if a.phase=='quality':
 for scene in ('ordinary','special'):
  jobs.append((f'{scene}_paired_final','benchmark_joint.py',['--scene',scene,'--tag','paired_final_benchmark','--cases']+fast))
  jobs.append((f'{scene}_combined_quality','evaluate_joint.py',['--scene',scene,'--tag','combined_final_quality','--variants','baseline']+fast))
  jobs.append((f'{scene}_exact_full','check_exact_sequences.py',['--scene',scene,'--cases','combo_scaled9_exact','combo_aligned16_nearest','combo_input9_half_scaled9_exact','combo_input16_half_native_stats_scaled9_exact','combo_input9_half_aligned16_nearest','combo_input16_half_native_stats_aligned16_fixed']))
 jobs.append(('weak','audit_weak.py',[]))
 for scene in ('ordinary','special'):jobs.append((f'{scene}_weight8','check_output_weight8.py',['--scene',scene]))
elif a.phase=='export':
 cases=['previous_combo','combo_scaled9_exact','combo_aligned16_nearest','combo_aligned16_fixed','combo_aligned16_folded_fixed','combo_relu_stats32_scaled9_exact','combo_project_after_resize_scaled9_exact','combo_input9_half_scaled9_exact','combo_input16_half_native_stats_scaled9_exact','combo_input9_half_aligned16_nearest','combo_input16_half_native_stats_aligned16_fixed','native4_linear_trained','native4_linear_ref_dither_float_trained','native4_relu8_ref_dither_trained','combo_tail5_exact_edges','combo_tail5_interior']
 for scene in ('ordinary','special'):jobs.append((f'{scene}_export','export_candidates.py',['--scene',scene,'--cases']+cases))
else:
 for scene in ('ordinary','special'):
  for case in ('combo_input9_half_aligned16_nearest','native4_linear_ref_dither_float_trained'):jobs.append((f'{scene}_{case}_video','make_full_videos.py',['--scene',scene,'--variant',case]))
for tag,script,args in jobs:
 t=time.monotonic();print('START',tag,flush=True)
 with (run/f'{tag}.log').open('w') as log:result=subprocess.run([py,str(base/script)]+args,stdout=log,stderr=subprocess.STDOUT,cwd=root)
 print('END',tag,'exit',result.returncode,'seconds',round(time.monotonic()-t,2),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
print('COMPLETE_PHASE',a.phase,flush=True)
