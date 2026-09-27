import os,subprocess
from pathlib import Path
base=Path(__file__).parent;root=Path('/data/zhangbenzhuang/huawei_sr');run=root/'runs/SS928-V07-NATIVE4-20260927';py='/data/zhangbenzhuang/miniconda3/envs/test/bin/python';os.environ['CUDA_VISIBLE_DEVICES']='GPU-655fc72e-3e1f-236c-ee8b-5ee23e9bd264';cases=['combo_stats3_relu32_aligned16_half_output_fixed']
for scene in ('ordinary','special'):
 jobs=[('benchmark_joint.py',['--scene',scene,'--tag','combined_npu_benchmark','--cases','previous_combo','combo_aligned16_nearest']+cases),('evaluate_joint.py',['--scene',scene,'--tag','combined_npu_quality','--variants','baseline','previous_combo']+cases),('check_exact_sequences.py',['--scene',scene,'--tag','combined_npu_full_sequences','--cases']+cases),('export_candidates.py',['--scene',scene,'--tag','combined_npu_export_manifest','--cases']+cases)]
 for script,args in jobs:
  tag=scene+'_combined_npu_'+script.removesuffix('.py');print('START',tag,flush=True)
  with (run/(tag+'.log')).open('w') as log:result=subprocess.run([py,str(base/script)]+args,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  print('END',tag,result.returncode,flush=True)
  if result.returncode:raise SystemExit(result.returncode)
with (run/'weak_with_combined_npu.log').open('w') as log:subprocess.run([py,str(base/'audit_weak.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
print('COMPLETE',flush=True)
