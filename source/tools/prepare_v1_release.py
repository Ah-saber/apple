"""Freeze configurations only after the independent middle-target build passes."""
import hashlib
import json
from pathlib import Path

code=Path(__file__).resolve().parents[1]
root=Path('/data/zhangbenzhuang/huawei_sr/runs')
middle=root/'MIDGT-TRAIN-V1-20260922'
index=middle/'index.json'
data=json.loads(index.read_text())
assert data['status']=='complete' and data['source_after_verified'] and data['total_frames']==2820
assert json.loads((middle/'preview_regression.json').read_text())['status']=='passed'
digest=hashlib.sha256(index.read_bytes()).hexdigest()
jobs=[]
for name in ('day','light_medium','heavy','night'):
    original=code/f'configs/train/b0_{name}_p256_d1.json'
    config=json.loads(original.read_text())
    for key in list(config):
        if key.startswith('comparison_'):
            del config[key]
    config.update(model=f'RT4KSR_B0_{name}_AugMidGT_v1',geometric_augmentation=True,
                  augmentation_protocol='uniform eight exact rot90/flip transforms after aligned crop; shared RAW/display/middle; train only',
                  auxiliary_raw_weight=0 if name=='day' else .1,
                  loss='display_L1 + auxiliary_raw_weight * normalized_middle_RAW_L1',
                  initialization='random; same seed and original B0 main-weight initialization as baseline',
                  auxiliary_resolution='input RAW grid: area3 of aligned denoised HR RAW crop',
                  auxiliary_attachment='shared body output before ISP tail; residual conv + pixelshuffle2; removed at inference',
                  version='aug_midgt_v1_20260922')
    if name!='day':
        config.update(middle_gt_root=str(middle),middle_gt_index_sha256=digest)
    else:
        config['auxiliary_resolution']=None;config['auxiliary_attachment']=None
        config['day_policy']='augmentation only; current traditional release has no day teacher denoising implementation'
    path=code/f'configs/train/v1_{name}.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    jobs.append({'name':name,'config':str(path.relative_to(code)),
                 'run':str(root/f'AUG-MIDGT-V1-20260922-{name.upper()}')})
plan={'version':'aug_midgt_v1_20260922','gpu_uuid':config['gpu_uuid'],'jobs':jobs,
      'policy':'serial, all preflights before training, stop on failure, no automatic restart'}
path=code/'configs/train/v1_queue.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'configured','jobs':jobs,'middle_gt_index_sha256':digest}))
