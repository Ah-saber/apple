#!/bin/bash
set -euo pipefail
R=/data/zhangbenzhuang/huawei_sr
Q=$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS
E=$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-BEST-TEST-FULLNIGHT
while ! test -f "$E/summary.json"; do sleep 30; done
export CUDA_VISIBLE_DEVICES=GPU-7e9093df-d21b-0d14-a009-078bf6dd94f1
exec /data/zhangbenzhuang/miniconda3/envs/test/bin/python "$Q/render_and_measure.py" \
    --root "$R" \
    --code "$R/code/worktrees/ss928-five-scene-day-nine-20260929" \
    --out "$R/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-BEST-VIDEOS-FULLNIGHT" \
    --best --split test \
    --groups DAY-FULLNIGHT LIGHT-FULLNIGHT HEAVY-FULLNIGHT C32-FULLNIGHT
