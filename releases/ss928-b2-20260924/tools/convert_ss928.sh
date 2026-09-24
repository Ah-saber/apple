#!/usr/bin/env bash
set -euo pipefail
release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_group=${1:?Usage: convert_ss928.sh GROUP NORMALIZED_INPUT OUTPUT_DIR}
case "$model_group" in day|light_medium|heavy|night) ;; *) echo 'Unknown model group' >&2; exit 2;; esac
normalized_input=$(realpath -- "${2:?normalized FP32 input required}")
result_dir=$(realpath -m -- "${3:?new output directory required}")
: "${ATC:?Set ATC to an executable ATC binary or executable environment-wrapper path}"
[[ -x "$ATC" ]] || { echo 'ATC must be executable' >&2; exit 2; }
[[ $(stat -c %s "$normalized_input") == 5242880 ]] || { echo 'Expected packed FP32 input: 5242880 bytes' >&2; exit 2; }
[[ ! -e "$result_dir" ]] || { echo 'Output directory exists; choose a new experiment directory' >&2; exit 2; }
mkdir -p -- "$result_dir"
compile_mode=${COMPILE_MODE:-6}
args=(--framework=5 --soc_version=SS928V100 --npu_arch=V101
 --model="$release_root/models/$model_group/student_raw_x3_1024x1280.onnx"
 --input_shape=raw:1,1,1024,1280 --input_type=raw:FP32 --output_type=FP32
 --image_list="raw:$normalized_input" --compile_mode="$compile_mode" --gelu_high_precision_mode=1
 --output="$result_dir/model" --log_level=1 --online_model_type=3)
printf '%q ' "$ATC" "${args[@]}" > "$result_dir/command.txt"
printf '\n' >> "$result_dir/command.txt"
sha256sum -- "$release_root/models/$model_group/student_raw_x3_1024x1280.onnx" "$normalized_input" > "$result_dir/source_sha256.txt"
"$ATC" "${args[@]}" 2>&1 | tee "$result_dir/conversion.log"
[[ -f "$result_dir/model.om" ]] || { echo 'Expected model.om missing' >&2; exit 1; }
sha256sum -- "$result_dir/model.om" > "$result_dir/model.sha256"
