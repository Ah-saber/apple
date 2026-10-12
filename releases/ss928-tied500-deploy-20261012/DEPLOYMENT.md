# 部署试测步骤

命令从本交付目录执行。使用部署端原有SDK环境。输入转换和比对工具只需NumPy；`requirements.txt`列出该依赖。

## 1. 准备校准与测试输入

```bash
python tools/prepare_inputs.py --scene all --out work/inputs
```

每景生成四个`calibration_XX/`和一个`smoke_inputs/`。每份有`raw_stack.bin`、`reference_thumb.bin`和metadata。BIN为小端FP16，顺序固定为RAW、参考缩略图。每景另有两个训练校准文件列表。

|场景|RAW形状|RAW字节数|参考形状/字节数|
|---|---|---:|---|
|night_ordinary、day_normal、night_special|1×5×1024×1280|13,107,200|1×1×64×64 / 8,192|
|四个weather场景|1×3×1024×1280|7,864,320|1×1×64×64 / 8,192|

输入通道从旧到新，最后一帧为当前帧。四份校准来自本景训练记录；测试第37帧只用于输出检查，不用于校准。预处理配置名、来源记录在各景manifest中。每个角色使用同一组输入。

## 2. 编译两角色完整图

用部署端当前ATC入口及其环境包装器。原工程使用`SS928V100 / V101`、framework5；两输入形状、FP16类型和完整输出规格由本轮manifest读取。`compile_model.py`沿用该入口参数，不在本机安装或调用SDK。

```bash
python tools/compile_model.py --scene weather_light --role tied500 \
  --atc /path/to/current/atc --compile-mode 1 \
  --image-list '<当前SDK格式的两个输入训练校准列表>' \
  --out work/compiled/weather_light/tied500 --print-only
```

将`--image-list`填为当前SDK接受的多输入校准格式，内容使用步骤1生成的四份训练BIN或两个列表；然后去掉`--print-only`执行。控制图将`--role`改为`retained_original`，其他设置相同。保留conversion.log、command.json、编译输出、SDK版本、量化设置和执行位置。新图含Slice/Concat/Transpose等算子；ONNX检查不代表SDK支持。

先测日景和轻天气，确认编译和完整输出，再覆盖其余五景。保留原板端可运行模型。本轮没有生成OM或操作板卡。

## 3. 核对完整输出

以该景`smoke_inputs/`的两个BIN运行编译图，导出完整去量化浮点输出。FP16原始输出应为23,592,960字节。不要在比对前拉伸对比度、转8位或裁剪。

```bash
python tools/compare_output.py --scene weather_light --role tied500 \
  --board-output board_results/weather_light_tied500.bin \
  --dump-dtype FP16 --out board_results/weather_light_tied500_compare.json
```

若SDK实际导出FP32去量化数据，显式用`--dump-dtype FP32`。工具核对字节数，并给出平均/最大/P95灰度差、亮度偏移和RMSE。PC参考来自确切导出ONNX，不是旧模型输出。对照角色同样执行并保留结果。

## 4. 测量完整纯 NPU 时间

每图50次预热，建议1000次正式调用，保存逐次耗时。分别记录完整图纯NPU和SDK同步时间；读文件、传输、图外显示转换分列。两角色使用相同输入、精度、编译设置和完整输出。逐算子时间和CPU回退也随结果回传。

按`board_timing_template.csv`填写实际读数，正式计时不含预热：

```bash
python tools/summarize_timing.py --csv board_results/timing.csv \
  --out board_results/timing_summary.json
```

若CSV中包含预热行，使用`--warmup-rows 50`按每个场景/角色移除。只有完整纯NPU mean、P95都≤16.7ms才达到速度目标。缺少纯NPU读数时保留空值，不能用GPU或SDK同步时间代替。

## 5. 画质与问题回传

提供两角色原始输出、比对JSON、编译日志、计时CSV、SDK及板卡运行配置。编译或运行失败时保留失败输入和日志，由部署端处理板卡；本轮不安排固件、重启或故障恢复操作。

七景完整GT视频用于查看当前算法已有问题。重天气C32的偏亮在保留原模型中已存在。所有GT原图均为完整1024×1280；C32和特殊夜景右侧192像素是原空间测试ROI，不能将ROI外当作缺GT。逐帧指标仍用原split ROI，整图显示不扩大独立测试指标范围。

本交付的固定第37帧用于匹配输出和计时，不代替连续量化画质验收。连续复测沿原manifest和本轮因果预处理生成输入，保持普通夜60帧、其他120帧，同帧GT、同区域和同灰度尺度检查。不能将旧版本预处理输入直接标成本轮输入。
