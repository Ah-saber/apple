# SS928 整模型提速候选

最新交接：[当前问题、已交付版本与下一轮板端验证](HANDOFF-20260927.md)（2026-09-27）。

目标≤16.7 ms，板端尚未验证达标。最终GPU实测普通0.424→0.341 ms、特殊0.431→0.329 ms；180帧源码画质、完整视频及负结果见 [RESULTS.md](RESULTS.md)。

推荐模型保留九帧、原四层主体、原参考分支及全部36输出相位。重新训练的统计前端替换原布局、全分辨率差分、前端残差和反卷积；完整行分组和缩放在模型内完成。一个已有弱窗口响应下降约5.35%，不宣称弱目标完全无损。

**这里的ONNX是源码图，尚未应用部署端的完整Resize兼容处理。直接沿用失败的v0.6 R1转换流程，可能再次出现CPU回退。** 用下述工具接入部署端现有native_r2，保留其已验证的参考/主干兼容处理；无需把native_r2传回。

本目录是增量交付，依赖同仓库相邻 [ss928-night-nine-packed-front-20260926](../ss928-night-nine-packed-front-20260926/README.md) 的冻结运行代码、原权重、真实输入和训练区校准向量。独立运行使用仓库文件，不依赖服务器上的实验目录。重新训练的研究脚本仍需原服务器数据和中间检查点，其训练样本与路径保存在 `evidence/`。

## 文件与接口

| 用途 | 每类场景的ONNX文件名后缀 | 九帧输入 / 输出 |
|---|---|---|
| 原版对照 | `_base.onnx` | `[1,9,1024,1280]` FP32 / FP32 |
| 仅行分组，保留原前端 | `_base_rows16.onnx` | 原FP32接口 |
| 统计前端，同接口对照 | `_balanced.onnx` | 原FP32接口 |
| 统计前端＋行分组 | `_balanced_rows16.onnx` | 原FP32接口 |
| 推荐候选 | `_balanced_low_rows16_nhwc16_half.onnx` | `[1,1024,1280,9]` FP16 / FP16 |
| 直接差分卷积对照 | `_contrast*.onnx` | 按文件名和图内接口查看 |
| 参考激活对照 | `_sigref.onnx` | 原前端、原FP32接口，仅替换参考分支激活和对应重训参数 |
| 字节输出试验 | `_balanced_low_rows16_nhwc16_byte.onnx` | NHWC FP16 / uint8 |

前缀 `ordinary` 为普通夜间，`special` 为特殊夜间。全部输出形状均为 `[1,1,3072,3840]`；64×64参考输入仍为FP32。FP16/FP32分别是16/32位浮点；uint8是0–255字节灰度。NHWC表示帧通道在最后一维，NCHW表示帧通道在第二维。

`onnx/controls/` 是64×96小整图的独立改写检查，供解释器验证，不能拿来代替完整模型测速。`source_operator_inventory.json`覆盖完整源图；`source_export_manifest.json`记录主导出，字节与参考激活另见算子清单及校验清单。

## 接入实际native_r2

在已有ONNX依赖的test环境执行。以下以普通数据举例，特殊数据更换文件前缀及原native模型。`native_r2.onnx`表示部署端自己的已验证文件。

```bash
# 同FP32接口，只换统计前端；保留原参考/Body/Tail/输出。
python tools/splice_native_front.py --native native_r2.onnx \
  --student onnx/ordinary_balanced.onnx --output ordinary_front_r2.onnx

# 在该兼容图上改完整输出重排，接口仍为FP32。
python tools/rewrite_output_rows.py --input ordinary_front_r2.onnx \
  --output ordinary_front_rows_r2.onnx --groups 16

# 推荐FP16入口：先接入九帧NHWC-FP16前端，后续兼容图保留。
python tools/splice_native_front.py --native native_r2.onnx \
  --student onnx/ordinary_balanced_low_rows16_nhwc16_half.onnx \
  --output ordinary_front_nhwc16_r2.onnx

# 缩放前移、完整行分组、输出FP16；所有操作在模型内。
python tools/rewrite_output_rows.py --input ordinary_front_nhwc16_r2.onnx \
  --output ordinary_primary_r2.onnx --groups 16 --packed-half-scale
```

前端接入严格核对原Head、四Body、Tail、输出的冻结参数和属性；拒绝不同模型或未审查的兼容变化。前端源图读取FP16，接入原FP32 Head时仅在边界转换。图中动态路径不能意外跨越被替换前端。

如果SDK不接受推荐输入/输出类型，保留拒绝信息并运行FP32对照；不能静默改类型后仍将结果记作推荐候选。转换后按OM接口重新检查布局、类型、缓冲区大小及输出顺序。

可选参考激活替换，保留actual R2中的Resize兼容路径：

```bash
python tools/rewrite_reference.py --native ordinary_front_nhwc16_r2.onnx \
  --old-source onnx/ordinary_base.onnx \
  --new-source onnx/ordinary_sigref.onnx \
  --output ordinary_reference_r2.onnx
python tools/rewrite_output_rows.py --input ordinary_reference_r2.onnx \
  --output ordinary_reference_primary_r2.onnx --groups 16 --packed-half-scale
```

参考工具匹配原参考卷积参数，更新为重训参数，将已识别的GELU换成Sigmoid×输入，保留原Resize。遇到非标准激活结构或参数不匹配会明确拒绝。该组合属于额外对照，默认保守模型使用原参考网络。

字节输出对照可在前端兼容图上加 `--packed-half-scale --output-byte`。Round、uint8以及Report类型能否在当前SDK中使用尚未验证；不得忽略CPU回退或输出类型变更。

## 输入和校准

继续使用相邻冻结包的原九帧预处理与归一化。每帧一个位置，全部九帧与64×64参考进入模型；新接口适配仅排列和转换这些原输入。

```bash
python tools/prepare_vendor_inputs.py --onnx ordinary_primary_r2.onnx \
  --vector ../ss928-night-nine-packed-front-20260926/input_vectors/ordinary_frame_20.npz \
  --output-dir prepared/ordinary_frame_20
```

读取生成的 `inputs.json` 来设置转换器的输入名、形状和类型。对相邻包 `calibration/` 的独立训练区向量逐个运行同样转换，再用部署端已经验证的多输入校准方式。保留实际SDK参数、编译日志、模型校验值及校准列表；本包不编造新SDK命令。新的统计卷积尤其需要检查量化后的亮度和时间差分。

## 匹配PC参考输出

`reference_outputs/*_balanced_low_rows16_nhwc16_half_reference.npz`对应相邻包普通frame20、特殊frame60输入及推荐源模型，灰度范围0–255。不要拿旧模型参考输出比较新模型。

```bash
python tools/compare_board_output.py --board ordinary_output.f16 --dtype float16 \
  --reference reference_outputs/ordinary_balanced_low_rows16_nhwc16_half_reference.npz \
  --output ordinary_board_vs_source.json
```

工具要求完整图，记录平均绝对误差、平均偏差、最大误差和6×6相位偏差。若输出先转成FP32保存，使用 `--dtype float32`。字节输出使用 `--dtype uint8`，参考会明确按字节取整。可选参考激活组合须用 `runtime/infer.py --case primary_reference`生成对应的新参考。

## 独立PC运行与测速

使用test环境及已授权GPU。原冻结包要与本目录相邻，也可用 `--base-dir`指定。

```bash
CUDA_VISIBLE_DEVICES=<指定GPU> python runtime/infer.py --scene ordinary \
  --case primary \
  --input ../ss928-night-nine-packed-front-20260926/input_vectors/ordinary_frame_20.npz \
  --output generated_ordinary.npz --gpu-lock <项目GPU锁文件>

CUDA_VISIBLE_DEVICES=<指定GPU> python tools/benchmark.py --scene ordinary \
  --vector ../ss928-night-nine-packed-front-20260926/input_vectors/ordinary_frame_20.npz \
  --output ordinary_timing.json --gpu-lock <项目GPU锁文件>
```

运行候选名称包括 `base`、`rows_only`、`front_f32`、`front_rows_f32`、`primary`、`contrast_f32`、`contrast_rows_f32`、`contrast_half`、`reference_only`、`primary_reference`、`byte`。推理默认源码路径；测速使用完整图编译，关闭TF32与CUDA图，30次预热、三轮各200次，保存全部记录。

## 完整视频

四列依次为输入RAW、GT、v0.6对应源码、新保守候选。每列显示1024×1280完整传感器，模型始终生成完整三倍图；视频为三倍输出取整后3×3平均。三倍完整PNG另行保存。

- [普通60帧完整视频](videos/ordinary_full_comparison.mp4)
- [特殊夜间120帧完整视频](videos/special_full_comparison.mp4)

## 新一轮板端记录

50次预热、600次同步推理，记录纯模型均值/P95、CPU回退、输入输出实际类型、逐任务或按模块耗时。原任务号会改变。优先回答：新前端几个任务及耗时、输出是否仍为六个重排反卷积、行分组/转置/恢复成本、输入与Report成本、参考激活替换的任务变化，以及匹配PC输出的亮度和相位误差。

只返回这些文字结果或表格也能继续推进，原native_r2、CSV和output文件不必传回。当前停止用于获得新硬件信息，后续继续优化16.7 ms目标。
