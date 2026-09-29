#!/bin/bash
set -euo pipefail
R=/data/zhangbenzhuang/huawei_sr
Q=$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS
P=$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-DAY-FULLNIGHT-LR1E4
while ! test -f "$P/exit.json"; do sleep 30; done
/data/zhangbenzhuang/miniconda3/envs/test/bin/python - "$P" <<CHECK
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/"exit.json").read_text())["exit_code"]==0;assert json.loads((p/"completed.json").read_text())["steps"]==12000
CHECK
export CUDA_VISIBLE_DEVICES=GPU-655fc72e-3e1f-236c-ee8b-5ee23e9bd264
exec /data/zhangbenzhuang/miniconda3/envs/test/bin/python "$R/code/worktrees/ss928-five-scene-day-nine-20260929/tools/run_managed_training.py" --config "$Q/day_fullnight_lr1e4_20k.json" --output "$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-DAY-FULLNIGHT-LR1E4-20K" --parent-run "$P" --max-wall-seconds 7200
