#!/bin/bash
set -euo pipefail
R=/data/zhangbenzhuang/huawei_sr
Q=$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS
P=$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-LIGHT-FULLNIGHT
while ! test -f "$P/exit.json"; do sleep 30; done
/data/zhangbenzhuang/miniconda3/envs/test/bin/python - "$P" <<CHECK
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/"exit.json").read_text())["exit_code"]==0;assert json.loads((p/"completed.json").read_text())["steps"]==12000
CHECK
export CUDA_VISIBLE_DEVICES=GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1
exec /data/zhangbenzhuang/miniconda3/envs/test/bin/python "$R/code/worktrees/ss928-five-scene-day-nine-20260929/tools/run_managed_training.py" --config "$Q/light_fullnight_20k.json" --output "$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-LIGHT-FULLNIGHT-20K" --parent-run "$P" --max-wall-seconds 7200
