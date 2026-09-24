# 部署与板端验收

以下命令以 `releases/ss928-b2-20260924/` 为当前目录。使用部署端既有Python环境和SS928工具链；本包不安装依赖。准备输入需NumPy；CPU参考推理还需ONNX Runtime；新序列离线归一化还需OpenCV。导出所需环境见NORMALIZATION_ONNX.md。

## 1. 文件与输入核验

```bash
python tools/deploy_io.py verify
python tools/prepare_inputs.py --output work/inputs
```

第二条命令将5个验证样例、7个训练标定样例转为归一化FP32文件，每个文件逐字节校验。输出 `work/inputs/inputs.json` 记录分组、来源、范围和SHA256。验证样例在 `work/inputs/samples/`，初始标定样例在 `work/inputs/calibration/`。命令拒绝覆盖已有目录。

例：`work/inputs/samples/weather_heavy.f32` 对应重度模型；`day_normal.f32` 对应白天；`weather_light.f32`、`weather_medium.f32` 对应轻中度；`night_ordinary.f32` 对应夜间。

## 2. 生成当前模型的CPU参考输出

```bash
python tools/deploy_io.py infer \
  --model models/heavy/student_raw_x3_1024x1280.onnx \
  --input work/inputs/samples/weather_heavy.f32 \
  --output work/heavy_cpu.f32
```

CPU输出是数值比较参考，不能作为GT计算模型画质。应保留未截断FP32输出，不能先转8bit再比较。

## 3. SS928转换

沿用部署端原有ATC环境。`tools/convert_ss928.sh` 使用原交接包 `scripts/convert_v02.sh` 中记录过的参数组合：SS928V100、V101、framework=5、compile_mode=6、gelu_high_precision_mode=1、online_model_type=3。新图尚未验证该工具链兼容性。

先初始化已有ATC环境，将ATC设为可执行二进制或可执行环境包装脚本的绝对路径。示例：

```bash
ATC=/absolute/path/to/atc bash tools/convert_ss928.sh \
  heavy work/inputs/calibration/heavy_0.f32 work/atc_heavy
```

脚本使用单个真实训练输入做首次转换检查，并保存命令、日志、输入/ONNX/OM的SHA256。支持 `COMPILE_MODE` 环境变量切换模式；不同模式使用新目录。文件路径按实际安装位置填写，不要直接使用示例路径。

初步转换成功后，按该版本ATC实际支持的方式使用同组训练输入进行代表性标定。包内7帧只是启动检查子集，未证明量化充分性；C32与特殊夜间未提供越过空间留出边界的整帧标定样例。需要在部署端用合法完整输入单独补充这两类验证。不要将验证样例作为唯一量化标定来源。

## 4. 板端输入输出

复用部署端已验证的 `svp_acl` 运行器，模型路径指向本轮新OM。输入名 `raw`，小端FP32、连续NCHW `[1,1,1024,1280]`，5,242,880字节；输出名 `display`，FP32 `[1,1,3072,3840]`，47,185,920字节。板端可能有stride/padding，导出前按实际张量描述取有效元素。

没有板端连接配置或SDK，故本包不包含伪装为已验证的OM或新编译运行器。旧运行器不能硬编码旧模型内部布局；输入输出缓冲大小应由本轮模型描述核验。关闭额外均值/缩放预处理，避免重复归一化。FP32输入输出接口不代表NPU内部计算精度。

## 5. 与当前CPU输出比较

```bash
python tools/deploy_io.py compare \
  --actual work/heavy_board.f32 \
  --reference work/heavy_cpu.f32 \
  --output work/heavy_comparison.json
```

记录最大绝对误差、MAE、RMSE、均值偏差。CPU ONNX相对PyTorch已通过数值校验，详见各模型 `verification.json`；此结果不能代替NPU量化误差检查。板端误差容忍阈值尚未约定，先返回实际误差及同帧图。

## 6. 必须记录的运行耗时

填写 `tools/board_timing_template.json`。固定batch1及上述输入输出尺寸；预热50次后，测3轮、每轮200帧；异步执行必须等待完成后停止计时。每轮分别保存均值、P50、P95、最大值和逐帧值，最后汇总。

- 仅模型：输入已在设备端，开始执行至同步完成；必须包含图内整图参考、融合及完整三倍输出。≤16.7ms的目标只针对这一项。
- 完整单帧：RAW在主机内存就绪，至三倍uint8图像在主机内存可用。
- 局部模块：归一化参数应用、输入布局处理、主机到设备拷贝、NPU模型、设备到主机拷贝、输出布局处理、clip/乘255/取整转8bit。若能分析图内耗时，分别记录主干卷积、参考分支、Resize、末端卷积和像素重排；编译器融合后的任务按真实粒度记录，不虚构拆分。
- 另列：整段离线归一化统计、模型加载、第一次推理、文件读写。离线统计不应隐藏在模型速度中，也不能因为预先完成就宣称全流程实时。
- 保存设备型号、SDK/ATC版本、OM/ONNX/输入SHA256、编译精度、时钟/功耗/并行业务状态。性能采样与开启profiling的采样分开。

## 7. 回传材料

每组返回转换命令和日志、OM校验值、运行器版本、计时JSON及逐帧记录、至少一份未截断FP32板端输出及其输入SHA、CPU比较JSON。画质比较补充原RAW、对应GT、原模型和新模型同帧/同区域输出以及连续视频。画质素材可留服务器，提供准确路径。原模型、原数据与已有结果继续保留。
