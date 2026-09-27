import subprocess
p='/data/zhangbenzhuang/miniconda3/envs/test/bin/python'
for scene in ['ordinary','special']:subprocess.run([p,'/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927/code_snapshot/export_finish_probes.py','--scene',scene],check=True)
