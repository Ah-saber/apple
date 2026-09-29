#!/bin/bash
set -euo pipefail
while ! test -f /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-HEAVY-C32/exit.json; do sleep 30; done
/data/zhangbenzhuang/miniconda3/envs/test/bin/python - <<'CHECK'
import json
from pathlib import Path
p=Path('/data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-HEAVY-C32')
assert json.loads((p/'exit.json').read_text())['exit_code']==0
assert json.loads((p/'completed.json').read_text())['steps']==12000
CHECK
export CUDA_VISIBLE_DEVICES=GPU-655fc72e-3e1f-236c-ee8b-5ee23e9bd264
exec /data/zhangbenzhuang/miniconda3/envs/test/bin/python /data/zhangbenzhuang/huawei_sr/code/worktrees/ss928-five-scene-nine-20260929/tools/run_managed_training.py --config /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS/heavy_c32_20k.json --output /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-HEAVY-C32-20K --parent-run /data/zhangbenzhuang/huawei_sr/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-HEAVY-C32 --max-wall-seconds 7200
