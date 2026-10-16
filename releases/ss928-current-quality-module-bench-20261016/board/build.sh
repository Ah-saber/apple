#!/usr/bin/env bash
set -euo pipefail
: "${SS928_SDK:?Set SS928_SDK to the installed SS928V100 SDK root}"
CC=${CC:-aarch64-linux-gnu-gcc}
TASK_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$TASK_ROOT/bin"
"$CC" -O3 -mcpu=cortex-a55 -ffp-contract=off -Wall -Wextra -DIR_GLIBC_217 \
  -I"$SS928_SDK/include/hisilicon/svp_npu" "$TASK_ROOT/board/module_bench.c" \
  -L"$SS928_SDK/lib/linux/hisilicon/svp_npu" \
  -Wl,-rpath,'$ORIGIN:/usr/app/chip/lib:/usr/app/chip/lib/svp_npu' \
  -Wl,--wrap=__libc_start_main -Wl,--allow-shlib-undefined -lsvp_acl -lm \
  -o "$TASK_ROOT/bin/module_bench"
