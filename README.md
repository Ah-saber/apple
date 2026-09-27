# v0.8 九帧完整推理：极限速度候选与部署诊断

2026-09-28 完成本轮训练、保留前端常规卷积改写、低分辨率网络、GT弱变化监督、参考/输出融合及布局对照。完整GPU校准快速版普通0.1875ms、特殊夜间0.1378ms，相对同轮三层源码均值低约35.5%/52.2%；夜间画质下降约0.48dB，保真候选另存。264模型ONNX、110模块探针、132份匹配源输出、14段整图视频、180帧输入重建与独立依赖证据见 [本轮交付](releases/ss928-night-nine-v08-extreme-20260928/README.md)。SS92816.7ms尚未实板验证；输入网络外场校正及计时边界已明确记录。首轮优先8类完整图及针对性探针，按实际任务表做第二轮优化。

# 九帧主体压缩、量化校准与16槽输入

2026-09-27 已继续完成18个训练/续训结果、200000实际优化步及四主体校准，完整GPU同接口两层快约9%～14%、三层快约4%～6%，保留夜间弱运动退化证据、36个源ONNX和八段整图视频。部署候选与当前未解决项见 [本轮交付](releases/ss928-night-nine-v07-body-input16-20260927/README.md)。SS92816.7ms仍待实板验证。

# 九帧继续优化：输出对齐、四相位训练与NPU完整候选

2026-09-27 已完成新训练、180帧画质与GPU复测。Half输入／16输出通道组合普通0.321～0.322ms、夜间0.305～0.320ms；保留Float32接口对照。针对NPU固定统计和六路输出，另提供3×3正负ReLU32统计＋单反卷积＋Half完整输出组合及独立版本。完整视频、实际证据和待验证状态见 [本轮交付](releases/ss928-night-nine-v07-native4-aligned-20260927/README.md)。NPU16.7ms尚未验证达到。

# v0.7 后续结构优化与完整视频

2026-09-27 已实现统计空间打包、原生均值约束九相位输出、量化权重搜索与保留多尺度的ReLU参考分支。RTX5090完整模型同轮实测普通0.377→0.350 ms、特殊0.383→0.350 ms；完成180帧画质、弱目标与完整整图视频，保留失败方案。部署候选、实际数值与当前未解决项见 [本轮交付与交接](releases/ss928-night-nine-v07-structural-20260927/README.md)。新图尚无SS928结果，16.7 ms未达到。

# v0.7 实板反馈与后续热点候选

2026-09-27 实板确认新前端普通/特殊81.228/80.351 ms；撤回rows16/NHWC半精度默认推荐。已继续实现统计与完整输出替代映射、独立精度对照，完成GPU实测及180帧源码回归，见 [本轮交付与最新交接](releases/ss928-night-nine-v07-hotspots-20260927/README.md)。新图尚需板端任务映射验证，16.7 ms未达到。

# 整模型提速候选与板端复测

2026-09-27 最新交接见 [当前问题与下一步](releases/ss928-night-nine-system-speed-20260926/HANDOFF-20260927.md)。

2026-09-26 新增 [整模型提速交付](releases/ss928-night-nine-system-speed-20260926/README.md)。统计前端、完整行分组、输入输出精度及参考激活均已实施验证；GPU最终实测普通0.424→0.341 ms、特殊0.431→0.329 ms，完整180帧视频与未采用方案均保留。一个已有弱窗口响应降低约5.35%。下一步需要新图的SS928编译、任务分解及匹配画质；16.7 ms尚未达到。原版本均保留，可按提交回溯。

# v0.6 报告后续模型试验

2026-09-26 新增 [模型侧增量候选](releases/ss928-night-nine-v06-model-controls-20260926/README.md)。部署报告中 v0.6 整模型同步耗时为 105.450／106.594 ms。新输出候选源码画质已核对完整 180 帧，但 GPU 由 0.425／0.432 增至 0.524／0.522 ms，SS928 收益待实测。提供保留部署端实际 R2 兼容图的改写工具，无需将 R2 文件传回。16.7 ms 尚未达到。

# 最新夜间九帧联合去噪候选

2026-09-26 进一步优化见 [联合去噪候选与完整验证](releases/ss928-night-nine-packed-front-20260926/README.md)。同条件 FP32 完整整图输出 GPU 实测：普通夜间 0.702→0.439 ms，特殊夜间 0.673→0.429 ms。完整 180 帧源码画质、整图视频、独立训练区校准与原始逐次计时已保存。SS928 新算子映射与板端耗时待验证，16.7 ms 尚未验证。

# 夜间轨迹前端学习候选

2026-09-26新增 [轻量轨迹候选与完整验证](releases/ss928-night-nine-trajectory-student-20260926/README.md)。相同GPU接口实测普通0.761→0.691 ms、特殊夜间0.733→0.655 ms；完成180帧质量回归和完整视频。SS928编译与板端耗时尚未验证，16.7 ms目标未达成。原factor24版本保留如下。

# 最新夜间九帧提速候选

2026-09-26 最新候选位于 [releases/ss928-night-nine-factor24-20260926/](releases/ss928-night-nine-factor24-20260926/README.md)。包含两类夜间整图 ONNX、运行权重、预处理、真实输入、当前参考输出、完整视频和测速证据。SS928 兼容性与 16.7 毫秒目标尚待部署端验证。

# 最新四组提速模型

2026-09-24部署候选位于 [releases/ss928-b2-20260924/](releases/ss928-b2-20260924/README.md)。包含白天、轻中度、重度、夜间ONNX与交接说明。可直接下载 [四组独立部署压缩包](packages/SS928-B2-FOUR-SCENES-DEPLOY-20260924-v2.zip)（约11.3MB），并用同目录.sha256校验。以下旧S03包完整保留供对照。

# S03 轻度＋中度天气部署测速包

本包对应训练完成 200000 步的 S03：16 通道、4 块、主干去 LayerNorm 并使用 ReLU、分级 2→3 像素重排、保留整图参考；默认导出使用等价的分级均值池化。整图参考的小网络仍有 GELU，不能把整个图称为“只有 ReLU”。辅助 RAW 头已从部署图移除。推理无需 GT。

## 先选模型

- `models/best_126000/`：按验证裁块 PSNR 选出的第 126000 步。默认先转换此模型。
- `models/last_200000/`：最后第 200000 步，供对照。
- `models/baseline_8000/`：相同 S03 结构的原 8000 步版本，供画质对照。
- 各目录的 `student_raw_x3_1024x1280.onnx` 可交给转换器；`static_fused_weights.pt` 为融合权重；`verification.json` 为 CPU 数值一致性记录；`operator_inventory.json` 为算子及形状清单。
- `weights/` 保存三份原训练检查点；`source/` 保留相应冻结代码；`reports/` 保存训练、质量、速度及选择依据；`normalization/index.json` 保存冻结的逐序列逐帧归一化参数。

训练已完成，完整质量测试、CPU ONNX 一致性检查已完成。SS928 转换和实板速度尚未测试；包内没有伪装为已验证的 OM 文件。

## 本次已知画质情况

长训练提高了裁块指标，但整图指标下降。以中度验证集为例，8000 步整图 PSNR 约 29.76 dB，默认 126000 步约 28.38 dB，平均亮度偏差约 -4.52 灰度级。详细数据见 `reports/RESULTS.md`。最佳权重的“最佳”只按验证裁块指标定义。此次部署可核实速度和转换精度，完整训练不能直接等同于整图画质提高。

整图质量指标：RAW 原尺寸输入→3倍输出→裁剪到[0,1]→面积回缩3倍→同一区域GT。不存在真实3倍高分辨率GT。

## 输入输出契约

- 名称 `raw`，FP32，小端、连续 NCHW `[1,1,1024,1280]`，共 5,242,880 字节。
- 必须使用本版本归一化：`x=(RAW.astype(float32)-offset_t)/scale_segment`，不截断输入。旧固定参数 5079.227/109.563 不适用于本包。
- ONNX 已包含整图缩略图、参考分支和对齐融合，仅有一个输入。无需外部提供参考图或 GT。
- 名称 `display`，FP32，小端、连续 NCHW `[1,1,3072,3840]`，共 47,185,920 字节。网络输出可超出[0,1]。
- 显示转换：`np.rint(np.clip(y,0,1)*255).astype(np.uint8)`。逐值比较时保留未截断浮点输出。
- 板端张量可能包含 stride/padding；先按模型描述提取有效元素，再按上述顺序导出。比较脚本拒绝尺寸错误或非有限值。

## 建议执行顺序

在已配置好依赖的 Python 环境中执行，命令以解压后的包根目录为当前目录；包内不会安装依赖。

```bash
python tools/deploy_io.py verify
python tools/deploy_io.py normalize --raw samples/weather_medium/raw.u16 --metadata samples/weather_medium/metadata.json --output medium_input.bin
python tools/deploy_io.py infer --model models/best_126000/student_raw_x3_1024x1280.onnx --input medium_input.bin --output medium_cpu.bin
python tools/deploy_io.py compare --actual medium_cpu.bin --reference samples/weather_medium/best_126000_output.f32 --output cpu_comparison.json
```

两份 `samples/` 分别为轻度和中度验证样例，包含原始 uint16、已经归一化的 `input.f32`、逐帧参数，以及默认第 126000 步权重的 CPU ONNX 输出；其他两种权重可运行 tools/deploy_io.py infer 生成。参考输出是模型数值参考，无 GT 含义。先直接使用 `input.f32` 排查模型转换；随后验证 RAW 预处理产生相同输入。样例及校准数据的源身份、帧号、SHA256 都记录在 JSON 中。

转换到 SS928 后，将紧凑浮点结果与同版本参考输出比较：

```bash
python tools/deploy_io.py compare --actual board_medium_output.f32 --reference samples/weather_medium/best_126000_output.f32 --output board_comparison.json
```

`calibration/inputs/` 提供训练集不同完整画面序列的归一化输入，`calibration/samples.json` 记录范围和来源。空间裁剪训练序列不生成整图校准样本。校准文件与两份验证样例分开。当前校准子集用于初步转换，量化画质充分性需要板端实测。

## RAW 归一化如何接入

已有样例直接使用 metadata 中的 offset、scale。它们仅适用于对应序列、对应帧，不能复用到另一段视频。

新视频必须先执行同一离线标定算法。源码为 `source/src/ir_sr/sequence_normalization.py`：先按照冻结规则分段；每帧从间隔8的采样点求中位数；对减去帧中心和段参考后的残差截断到[-4,4]求公共偏移；由段平均RAW的0.1%和99.9%分位数确定范围，最小跨度20；得到逐帧 offset 和段内 scale。数学操作及精度以原函数 `calibrate_sequence` 为准。

```bash
python tools/deploy_io.py calibrate-sequence --sequence complete_raw_sequence.npy --output sequence_parameters.json
```

输入为原始 `uint16[T,1024,1280]` 的完整序列。标定需要未来帧，当前属于离线方案，实时因果归一化尚未实现。测速时必须单独记录离线标定耗时、逐帧套用参数耗时；使用预先归一化输入的测速仅代表模型部分。

## SS928 转换要求

沿用部署方现有已验证工具链：SS928V100，`svp_acl`，ATC `framework=5`、`soc_version=SS928V100`、`npu_arch=V101`、opset17、静态 batch1。输入 `raw:1,1,1024,1280`、`input_type=raw:FP32`，输出 FP32，关闭额外输入均值/缩放处理。FP32 接口不代表内部计算为 FP32。

转换参数格式可参考旧工程：`--framework=5 --soc_version=SS928V100 --npu_arch=V101 --input_shape=raw:1,1,1024,1280 --input_type=raw:FP32 --output_type=FP32 --compile_mode=6 --online_model_type=3`。模型路径、输出目录及代表性输入按本包重新设置。旧工程单样本语法为 `--image_list=raw:<绝对路径输入文件>`；多样本列表按安装版本的工具说明生成，本包不推测未经核实的格式。

完整保留转换命令、版本、日志、编译选项和 OM 的 SHA256。先以原导出图转换；如遇算子兼容或量化问题，在独立目录改写并做 PC 数值一致性检查。重点检查参考分支的 AvgPool、Resize、GELU/Erf 和两级像素重排；主干去掉 LayerNorm 并不证明整图兼容。不要重复置换已经完成分级重排的输出卷积通道。

## 必须返回的耗时记录

同一设备负载条件、固定输入输出尺寸、batch1，预热50次后连续至少600次；异步执行需等待完成再计时。记录均值、P50、P95、最大值和全部逐帧记录。

1. 纯模型：输入已在设备端，开始执行至同步完成，包含图内整图参考与输出重排。
2. 完整单帧：RAW在主机内存就绪至3072×3840的uint8输出在主机内存可用。
3. 局部模块：RAW参数应用、输入布局和拷贝、NPU模型、输出拷贝和布局、clip/量化到8bit；离线序列标定、文件读写另列。使用编译器分析工具时，还需记录主干卷积、参考分支、Resize与输出重排的耗时。
4. 记录模型加载和第一次推理耗时；记录并行业务、频率及功耗状态，保留旧业务。
5. 使用不同训练权重比较画质时，明确 ONNX/OM/输入和参考输出身份。相同结构的测速结果不能冒充所有场景均已完成训练验证。

`tools/board_timing_template.csv` 为填写格式。16.7ms是目标，当前尚无SS928实测达标证据。5090数据见 `reports/timing/report.json`，不能直接按比例宣称板端耗时。

## Git 仓库交付说明

本仓库是从已校验部署包展开的独立交付目录，保留默认权重的两份完整浮点参考输出。8000 步和 200000 步参考输出可用 `tools/deploy_io.py infer` 按各自 ONNX 生成。`MANIFEST.json` 只覆盖本 Git 版本实际收录的文件，所有收录文件的校验值已更新。

克隆后运行 `python tools/deploy_io.py verify` 检查内容。模型使用、SS928 转换和耗时记录步骤见上文。
