# v0.7 后续：统计与完整输出映射对照

当前默认研究起点是已板测81.228/80.351 ms的 `front_f32`。撤回上一包rows16/NHWC半精度组合的默认推荐。目标16.7 ms未达到，本包新图尚无SS928测量。结论、GPU实测、负结果及边界见 [RESULTS.md](RESULTS.md)，板测原文见 [BOARD_REPORT_v07.md](BOARD_REPORT_v07.md)。

## 原依赖和范围

沿用同仓库 `ss928-night-nine-system-speed-20260926/` 和 `ss928-night-nine-packed-front-20260926/`。本轮未修改训练权重；九帧、参考分支、全部36相位及完整三倍输出均保留。独立热点图仅用于研究编译映射，不能以其时间替代完整模型成绩。

`onnx/ordinary_*`、`special_*` 是完整源码候选，仍含源码Resize，不能直接宣称部署零回退。优先对部署端已经成功编译的 **front_f32兼容ONNX** 用下列工具改写，沿用本轮报告中的SDK编译、校准及运行流程，无需把实际兼容文件传回。

## 优先测试完整输出映射

工具替换唯一shuffle6，后续原裁剪、缩放及输出接口保留。一次只选一个：

```bash
python tools/rewrite_hotspots.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_out_deconv6.onnx --permutation deconv6
python tools/rewrite_hotspots.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_out_reshape.onnx --permutation reshape
python tools/rewrite_hotspots.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_out_2_3.onnx --permutation shuffle2_3
```

另有 `deconv2_shuffle3`、`deconv3_shuffle2`、`shuffle3_2`。名称表示固定反卷积步长及剩余通道重排倍数，或两级重排顺序。两级重排包含明确的通道顺序调整，避免错误的相位排列。

用部署PC已有ONNX Runtime生成相同真实微基准输入：

```bash
python tools/prepare_micro_vectors.py --model ordinary_front_f32_r2.onnx \
  --vector ordinary_frame_20.npz --output-dir ordinary_micro_inputs
```

该工具从同版本PC兼容图截取完整输出重排前张量，保留原计算至此处；只用于探针输入。生成statistics/output各FP32与FP16紧凑二进制及归一化单位的期望重排输出，仍需运行器按OM实际stride装填。校准用同样流程生成已有训练区四组向量，测试帧不加入校准。已用完整真实向量和独立解释器验证生成路径；本轮环境没有ONNX Runtime，ORT执行路径未在此运行。

微基准 `micro_output_*_{fp16,fp32}.onnx` 输入 `[1,36,512,640]`，输出 `[1,1,3072,3840]`；baseline与候选使用相同输入、类型与范围。检查编译后的六个反卷积是否减少，新增转置/Gather/恢复任务是否抵消收益。固定反卷积在GPU更慢，作为映射试验，不预言NPU更快。

## 再测试统计卷积分解

```bash
python tools/rewrite_hotspots.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_stats_temporal.onnx --statistics temporal_first
python tools/rewrite_hotspots.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_stats_pack.onnx --statistics pack_first
```

`temporal_first`：时间1×1卷积9→3，再groups=3的2×2步长2打包；`pack_first`：groups=9的空间打包9→36，再1×1卷积36→12混合。groups表示每组输入通道独立卷积。`temporal_f32`提供中间FP32对照，部分帧已观察到极小舍入差，不作为位级等价默认。

`micro_statistics_*` 输入 `[1,9,1024,1280]`，输出 `[1,12,512,640]`。原统计常量权重与分解均保留，实际SDK量化可能改变中间舍入与缩放，仍需板端比较。禁止用更小尺寸的时间代替这些完整尺寸热点的时间。

## 单独定位半精度和缩放问题

```bash
python tools/rewrite_interfaces.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_nchw_half.onnx --mode input_nchw_half
python tools/rewrite_interfaces.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_output_half.onnx --mode output_half
python tools/rewrite_interfaces.py --input ordinary_front_f32_r2.onnx \
  --output ordinary_scale_f32.onnx --mode scale_packed_f32
```

输入半精度保持NCHW，模型内先恢复原FP32入口语义，转换器可进一步消除冗余转换；是否消除需看实际图。输出半精度在完整原计算之后转换。缩放前移保留原点运算的精度、常量及顺序，支持Clip已在shuffle前的原R2。

这些对照分别从同一个front_f32生成，不能从primary叠加后声称单因素。检查实际OM接口及stride，按类型生成输入/校准。可复用上一包 `tools/prepare_vendor_inputs.py`；只改输出的参考应使用原front_f32结果转FP16，其他结果各自重新计算。禁止复用不匹配的参考权重或接口。

## 回传信息

同条件基线、编译成功/失败信息、CPU回退、稳态同步均值/P95、独立逐任务profile，及完整板端输出对同版本PC的MAE、偏差和36相位误差。输出热点微基准先筛选，再以完整图复测；不能把微基准之和称为完整模型时间。

若需要检验连续帧，完整原始输入/GT应保持原序列、预处理与历史帧边界，不能从压缩对照视频恢复GT。本包180帧检查是源码回归，未提供板端视频验收。

## 本轮复现与证据

- `benchmark_candidates.py`：完整尺寸热点及完整模型GPU计时，冻结原输入/参数。
- `check_sequence.py`：普通60/特殊120帧完整源码输出回归。
- `check_and_export.py`：40个独立解释器控制和46个ONNX检查/导出。
- `check_folded_front.py`：统计/首学习卷积合并的未采用方案，原始负结果保留。
- `runtime/operator_candidates.py`：研究运行代码；原交付依赖由其原loader加载。
- `evidence/`：原始批次计时、逐帧差异、图校验与导出SHA。

研究复现脚本引用原服务器数据和冻结实验目录，不冒称所有训练原始数据已包含本包。部署改写工具只需既有test环境ONNX，不依赖GPU研究目录；拒绝覆盖输出。
