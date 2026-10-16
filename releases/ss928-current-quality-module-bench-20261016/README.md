# 当前画质版：七景分模块部署测速包

日期：2026-10-16。

本包用于在SS928部署端编译和测量当前版的神经网络模块。C32对应已交付视频的“小目标均值保持修正”。重天气和特殊夜景使用当前保留分支，其他四景沿用接手前的冻结SR。

**本包提供ONNX、真实输入、PC参考输出、调用清单和板端运行器源码。当前多帧CPU处理尚未移植到ARM，暂时只能得到模块时间，不能得到当前版完整帧速。**

OM和ARM可执行文件使用部署端原有SDK生成。

## 文件入口

| 文件或目录 | 用途 |
| --- | --- |
| `PACKAGE_INDEX.json` | 七景图接口、模块类型和输入输出规格 |
| `models/` | 每张静态ONNX、训练输入、核对输入及PC输出 |
| `workloads/` | 原处理流程的逐帧模块调用顺序和次数 |
| `PC_COMPARISONS.json` | 导出图与实际GPU模块输出的差异 |
| `tools/module_bench.py` | 准备BIN、调用ATC、比对输出、汇总时间 |
| `board/module_bench.c` | 多输入、多输出、FP16/FP32板端同步测速运行器 |
| `board/build.sh` | 使用原SDK交叉编译运行器 |
| `PORTING.md` | 接入完整处理流程仍需移植的CPU和状态处理 |
| `source/runtime/` | 当前流程的实际Python源码快照 |
| `source/run_snapshot/` | 实际加载的权重和选择参数 |

同一个网络有多个分块尺寸或初始化模式，因此会对应多张静态图。图目录编号只用于文件定位；具体模块及输入含义见同目录的`manifest.json`和`argument_spec.json`。

15个原FP64临时接口已显式转为FP32，权重不变。非零样本及数值差异记录见`BOUNDARY_PRECISION.json`。部署输入精度以`manifest.json`为准。

## 七景范围

共80张静态图。下表调用次数取原流程第19帧，同一网络的分块和采样位置调用分别计数。

| 场景 | 静态图数 | 第19帧调用次数 | 本包内容 |
| --- | ---: | ---: | --- |
| 普通夜景、日景、轻天气、中天气 | 各1 | 各1 | 原冻结SR完整图及原真实输入 |
| 重天气 | 3 | 5 | 当前SR和局部修正模块 |
| 重天气C32 | 70 | 228 | 当前SR、边缘SR、多帧目标与背景修正模块 |
| 特殊夜景 | 3 | 5 | 当前SR和运动、背景修正模块 |

新导出模块的校准输入来自原训练数据回放，核对输入来自原验证数据回放，GT不进入推理。两组来自相同采集片段时，RAW可相同，不能当作两组独立画质样本。内容完全相同的输入只保存一份，来源仍分别记录。

## 部署端操作

### 1. 准备真实输入BIN

在PC上安装NumPy，从本目录执行：

```bash
python tools/module_bench.py list
python tools/module_bench.py prepare --out work/inputs
```

每张图生成`smoke/`、`calibration_XX/`、各输入的`.list`和规格文件。输入为小端数据，按各图清单保留FP16或FP32。直接写入SDK，不再次归一化。

`models/<场景>/<图目录>/manifest.json`列出ONNX真实输入。`argument_spec.json`同时说明这些扁平输入在原Python调用中的位置，供完整流程接入使用。

### 2. 用原ATC编译

沿用原`SS928V100 / V101 / M6`工具链。多输入`image_list`的参数格式使用部署端当前SDK规定的格式，数据使用步骤1生成的训练输入列表。

以下例子定位到C32冻结SR主干：

```bash
python tools/module_bench.py compile \
  --graph weather_heavy_c32/graph_000 \
  --atc /path/to/current/atc \
  --image-list '<当前SDK的训练输入列表参数>' \
  --out work/compiled/weather_heavy_c32/graph_000
```

工具按图清单设置输入形状及精度，保存实际命令和`conversion.log`。其余图使用`list`输出的键依次编译，输出目录保持`work/compiled/<场景>/<图目录>/model.om`。

### 3. 编译并运行板端程序

```bash
SS928_SDK=/path/to/SS928V100_SDK_V2.0.2.2 bash board/build.sh
```

将`bin/module_bench`、编译后的`compiled/`及准备好的`inputs/`送到板端。使用板端原有库路径和SDK运行配置，从其工作目录执行：

```sh
SS928_RUNNER=./module_bench OM_ROOT=./compiled INPUT_ROOT=./inputs \
RESULT_ROOT=./board_results WARMUP=50 RUNS=100 \
sh inputs/run_components.sh
```

各图输入预先读入，计时只包围同步SDK执行。运行器按SDK行步长准备数据，输出按逻辑形状去除行填充，保存为`output_XXX.bin`。运行要求AICPU任务为零；结果保留逐次`timing.csv`与`run.json`。

### 4. 比对输出和汇总

把板端结果取回PC：

```bash
python tools/module_bench.py compare \
  --graph weather_heavy_c32/graph_000 \
  --board-output work/board_results/weather_heavy_c32/graph_000 \
  --out work/board_results/weather_heavy_c32/graph_000/compare.json

python tools/module_bench.py summarize \
  --results work/board_results --out work/timing_summary.json
```

纯NPU时间使用部署端原SDK profiler另行采集。将读数按`board/profile_template.csv`格式填写，再通过`--profile-csv`传给`summarize`。同步SDK时间和纯NPU时间分别记录。

汇总工具按实际调用次数计算模块平均时间之和。该值是组件开销估计，完整一帧时间和完整帧P95均保留为空。各模块的P95不能相加。

## 当前速度及接入要求

当前C32原研究流程在RTX 5090与CPU上平均8.27秒/帧，P95为8.59秒/帧，去除视频编码和落盘。模块上板后，还需按[移植清单](PORTING.md)接通RAW处理、历史状态、标签投影和完整输出融合，才能测SS928完整帧速及16.7毫秒目标。

本次保持已查看视频对应的权重和选择参数，未重新训练。
