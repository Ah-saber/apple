"""Correct the initial wide-loop metadata before fixed 8000-step continuation."""
from pathlib import Path
import torch,json,hashlib
root=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-EXTREME-20260927');archive=root/'budget_audit';archive.mkdir(exist_ok=True);rows=[]
for scene in ('ordinary','special'):
 for kind in ('k4','3x3'):
  path=root/f'{scene}_quarter_w32_{kind}_body1_024000.pt';old=torch.load(path,map_location='cpu',weights_only=True);assert old['successful_optimizer_updates']<=16000;raw_archive=archive/(path.stem+'_original_mislabeled.pt')
  if raw_archive.exists():raise FileExistsError(raw_archive)
  raw_archive.write_bytes(path.read_bytes());old['original_mislabeled_step']=old['step'];old['step']=16000;old['attempted_iterations']=16000;dest=root/f'{scene}_quarter_w32_{kind}_body1_016000_stage1.pt';torch.save(old,dest);rows.append({'original_file':path.name,'original_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actual_attempted_iterations':16000,'actual_successful_optimizer_updates':old['successful_optimizer_updates'],'corrected_stage1_file':dest.name,'planned_continuation':8000})
(root/'BUDGET_AUDIT.json').write_text(json.dumps({'issue':'Initial wide script changed metadata/scheduler to24000 but range endpoint remained16001; corrected and continued8000 steps. Original raw files preserved.','models':rows},indent=2))
print('ARCHIVED_AND_CORRECTED_STAGE1',len(rows))
