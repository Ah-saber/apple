# v0.13 板测输入与复现

部署模型包与真实校准数据包分别解压。数据包内保留 `deployment_corrected/<scene>/`、`night_calibration_final/<scene>/`，这些名字与 `MODEL_INDEX.json` 对应。源记录内的绝对路径用于追溯；解压后的输入按目录中的相对文件名选择。

## 先核对完整图和输入

1. 从 `MODEL_INDEX.json` 选择图，核对 SHA256、输入名字、输入类型与尺寸。五类同权重图优先测试完整 `rows32`；`phases` 只有相位输出，作为诊断。
2. 保持部署端既有编译器、设备与测试启动条件，记录版本。新 Gather 图的兼容性待确认；两类夜间保留源图，仍须使用经过同输入数值核对的参考插值兼容改写。
3. 用训练校准的 `train_00`～`train_03.npz` 生成校准输入，各份九帧历史真实独立。两类夜间的逐输入精度可能不同，推荐指定确切 `--model`。

以下命令在带依赖的 `test` 环境执行，`release`、`data` 是解压目录：

```bash
python release/runtime/materialize_calibration.py \
  --input data/deployment_corrected/weather_light/train_00.npz \
  --model release/onnx/equivalent/weather_light/weather_light_equivalent_fp16_rows32.onnx \
  --out calibration/weather_light_00 --inputs-only

python release/runtime/regenerate_reference.py \
  --model release/onnx/equivalent/weather_light/weather_light_equivalent_fp16_rows32.onnx \
  --sample data/deployment_corrected/weather_light/test_00.npz \
  --out pc_exact_weather_light.npz
```

生成的 `.bin` 为 NCHW 连续小端浮点，实际类型与形状写入 `metadata.json`。`regenerate_reference.py` 使用确切 ONNX 及其声明的输入类型，生成完整 PC 参考；记录模型哈希和后端版本。输入转换不会自动生成该精度或该候选的参考输出。

日间 `current_folded` 指定图后工具会选择历史末帧。使用 `--inputs-only`，另以确切图生成 PC 输出；九帧稳帧模型的输出不能作它的参考。两组天气小模型亦须各自生成对应权重的 PC 输出。

## 逐层定位与连续帧

数据包中五类 `pc_layers_detailed_fp32.npz`、`pc_layers_detailed_fp16.npz` 包含参考编码、各金字塔分支、投影、放大、主体、融合、尾部、裁剪前相位等。夜间为 `pc_layers_source_fp16.npz`。这些参考计算的精度在文件名和记录中说明，数组统一以 FP32 保存。

先对同一份 `test_00.npz` 留出部署图的对应层。板端整数输出须按实际量化参数反量化，并核实实际布局，再使用：

```bash
python release/runtime/compare_layers.py \
  --pc data/deployment_corrected/weather_light/pc_layers_detailed_fp16.npz \
  --board board_weather_light_dequantized.npz --out layer_difference.json
```

不能凭输出长度猜通道排序，也不能用全局亮度平移掩盖相位偏差。记录第一个显著偏差层，以及每通道、四相位误差。完成浮点图核对后再比较 SDK 量化图与 GT。

`cast_probe/` 为两张 8×8 逐值探针，包含截断及就近偶数舍入参考。先测实际设备 Cast 规则，再测完整字节图的布局、数值、CPU/AICPU 归属与 Report。字节图的正确性与性能均未在本轮板端验证。

新天气视频保留完整 120 帧四栏；日间失败候选存于 `videos/rejected/`。原日间默认结果和两类夜间整图视频见此前同分支发布包。本轮 84 份测试输入提供同帧 GT 校准对照，连续板端视频应按记录中的原序列从首帧因果处理至末帧，核对真实运动、静态区域、建筑、弱目标、亮度与局部对比。

## 速度边界和回传内容

正式成绩只统计模型图内完整 1024×1280 输入至 3072×3840 输出，包含图内 Report，排除文件读取和主机拷贝。保持统一预热与至少 600 次同步执行，另给出独立纯 NPU 均值和 P95及完整原始采样。

每图回传源图/兼容改写图/量化图/OM 哈希、编译日志、算子与 CPU/AICPU 归属、逐任务时间、输出铺排及 Report 时间、同帧输出和连续帧画质。两类夜间及五类新图须分别计时；GPU 时间、诊断图时间及旧版夜间时间不作为新完整图成绩。

## 独立权重复现

无需服务器教师权重，使用本包五类源码与权重：

```bash
python release/runtime/reproduce_model.py --release release \
  --scene weather_heavy_c32 --kind equivalent \
  --sample data/deployment_corrected/weather_heavy_c32/test_00.npz \
  --out source_c32.npz --precision fp32
```

`--kind fused` 为 20k 天气候选，`temporal` 为 8k 时序续训候选，`current` 为日间原画质备选的首层折叠。该源复现入口需要 GPU 的 `test` 环境，禁用 TF32；直接运行 ONNX 的 PC 参考入口可用 CPU。原训练复现的源代码、参数、数据索引与缓存哈希均随训练记录保留，原服务器路径保持有效。
