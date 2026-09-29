#!/bin/bash
set -euo pipefail
while ! test -f /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-LIGHT-MEDIUM/exit.json; do sleep 30; done
/data/zhangbenzhuang/miniconda3/envs/test/bin/python - <<'CHECK'
import json
from pathlib import Path
p=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-LIGHT-MEDIUM')
assert json.loads((p/'exit.json').read_text())['exit_code']==0
assert json.loads((p/'completed.json').read_text())['steps']==12000
CHECK
export CUDA_VISIBLE_DEVICES=GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1
exec /data/zhangbenzhuang/miniconda3/envs/test/bin/python /data/zhangbenzhuang/huawei_sr/code/worktrees/ss928-five-scene-nine-20260929/tools/run_managed_training.py --config /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS/light_medium_20k.json --output /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-LIGHT-MEDIUM-20K --parent-run /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-LIGHT-MEDIUM --max-wall-seconds 7200
