#!/usr/bin/env bash
set -euo pipefail

group="${1:?light, heavy, or day}"
steps="${2:-12000}"
root=/data/zhangbenzhuang/huawei_sr
run="$root/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-CONFIGS"
python_bin=/data/zhangbenzhuang/miniconda3/envs/test/bin/python
code="$root/code/worktrees/ss928-five-scene-nine-20260929"
night_runtime="$root/runs/SS928-EXTREME-20260927/code_snapshot/runtime"

case "$group" in
    light)
        base="$root/runs/SS928-FIVE-SCENE-QUARTER-LIGHT-20260929"
        reference="$root/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-LIGHT-FULLNIGHT-20K/checkpoints/best.pt"
        scenes=(weather_light weather_medium)
        ;;
    heavy)
        base="$root/runs/SS928-FIVE-SCENE-QUARTER-HEAVY-20260929"
        reference="$root/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-HEAVY-FULLNIGHT-20K/checkpoints/best.pt"
        scenes=(weather_heavy weather_heavy_c32)
        ;;
    day)
        base="$root/runs/SS928-FIVE-SCENE-QUARTER-DAY-20260929"
        reference="$root/runs/SS928-FIVE-SCENE-NINE-SCRATCH-20260929-DAY-FULLNIGHT-LR1E4/checkpoints/best.pt"
        scenes=(day_normal)
        ;;
    *) echo "unknown group: $group" >&2; exit 2 ;;
esac

exec "$python_bin" "$run/refine_quarter_detail.py" \
    --code "$code" --night-runtime "$night_runtime" \
    --config-checkpoint "$reference" \
    --reference-checkpoint "$reference" \
    --base-checkpoint "$base/best.pt" \
    --train-cache "$base/train_cache.pt" \
    --scenes "${scenes[@]}" \
    --out "$root/runs/SS928-FIVE-SCENE-DETAIL-${group^^}-20260929" \
    --steps "$steps" --motion-weight 0.5 --teacher-weight 0.5
