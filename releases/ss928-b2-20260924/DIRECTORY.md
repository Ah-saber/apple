# 目录与文件

|位置|内容及使用方法|
|---|---|
|README.md|四组模型选择、状态和阅读入口|
|DEPLOYMENT.md|从校验、准备输入到ATC、板端对比、计时与回传|
|NORMALIZATION_ONNX.md|归一化公式、输入输出约定、权重及重导出要求|
|HANDOFF.md|接手上下文、完成状态、限制及后续技能建议|
|MODEL_MANIFEST.json|每组权重/ONNX身份、步数、场景和理论耗时|
|MANIFEST.json / SHA256SUMS.txt|本交付全部文件的字节数及校验值；清单文件互不循环收录|
|models/{day,light_medium,heavy,night}/|ONNX、融合权重、训练检查点、配置、CPU核验、算子与复杂度|
|normalization/|两份冻结归一化索引；不能将一个视频参数复用于其他视频|
|samples/*/|五类完整帧验证样例的raw.u16和metadata；用于生成输入及当前CPU参考，无GT|
|calibration/*/|七个训练完整帧及参数；初始转换用，未证明量化充分性|
|tools/prepare_inputs.py|将RAW样例生成FP32输入并逐文件核对SHA|
|tools/deploy_io.py|文件核验、RAW归一化、CPU推理和部署误差比较|
|tools/reexport.py|直接从包内训练权重导出静态ONNX并做CPU核验|
|tools/convert_ss928.sh|调用部署方既有ATC并保存命令、日志、校验值|
|tools/board_timing_template.json|整体、模型及局部模块计时回传模板|
|source/src/ir_sr/|训练模型、部署封装和归一化实现快照|
|reports/cross_scene/|本轮白天、重度、夜间验证及原始指标|
|reports/light_medium/|先前轻中度结构对照；本包只选其中B2权重|
|reports/TARGET.md|16.7ms目标、用户确认的理论换算范围及结果|
|reports/environment.json|实际导出环境版本|
|reports/delivery_verification.json|交付工具、样例及CPU输出自检记录|
|work/|部署端生成的输入、输出、编译和误差结果；不纳入Git|

服务器完整训练、同帧图和视频路径在reports内。原始数据与冻结GT位于 `/data/zhangbenzhuang/huawei_sr/data/`，未复制进本交付。旧S03部署版本仍在Git仓库根目录，不覆盖。

模型适配源于RT4KSR（Apache-2.0）；来源和许可证位于source/third_party/RT4KSR/。
