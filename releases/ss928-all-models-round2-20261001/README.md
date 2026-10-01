# SS928 全模型优化：板测交付入口

2026-10-01，用户审查后明确授权推送。本次沿用 `codex/ss928-night-nine-factor24-20260926` 分支，新增独立目录，原图、原权重及原部署保留。

七类保留原权重、原浮点输出接口的完整 GPU 前向均值下降 12.4%～26.2%。天气三帧主体、日间残差补偿、夜间弱历史混合已完成连续帧实验，小目标拖影有部分改善。日间和特殊夜间拖影候选增加耗时及部分结构误差，保持独立。SS928 新图尚未实测，完整纯 NPU 均值/P95 ≤16.7 ms 仍待验证。

## 先使用的图

[MODEL_INDEX.json](MODEL_INDEX.json) 按七类列出 21 张完整浮点图：每类原图、原权重新布局、拖影候选各一张。路径相对于仓库根目录，包含确切 SHA-256、输入名称/精度/形状、输出及对应记录。五类新布局名称中的 `_integer.onnx` 是实验历史文件名，索引已核实该图实际输出为 Float16。

1. 先比较每类 `original_float` 和 `layout_float`，保持输入、输出精度、编译条件、计时边界一致，确定布局改动的板端收益。
2. 再分别验证 `motion_float`，同时观察量化后的目标当前位置、旧位置残留、对比、建筑弱结构及静态变化。天气用三帧主体；日间用残差补偿；夜间混合强度为 0.25。C32 有效标注不覆盖所选天气道路，特殊夜间仅右侧 192 像素有效。
3. 原始输入仍为九帧和参考，输出均为 `1×1×3072×3840`。参考和原有预处理的历史范围保留，三倍输出沿用最近邻复制。

[板测步骤与计时边界](BOARD-VALIDATION.md) 要求核对 SDK 算子支持和实际执行位置，预热至少 50 次，再保存至少 1000 次完整前向耗时，分别报告均值、P95、逐算子时间、图内输出转换和 CPU 回退。图外 RAW 读取及主机传输单独记录。

## 真实输入与参考输出

复用上一轮已交付的 `SS928-V13-REAL-CALIBRATION-AND-GT-20260930.tar`，其五类输入在 `deployment_corrected/<scene>/`，夜间在 `night_calibration_final/<scene>/`。每类训练校准 `train_00`～`train_03.npz` 使用九个真实不同历史帧；测试输入为 `test_00`～`test_11.npz`。原输入定义保持不变，各候选须用确切新图重新生成参考输出。大型数据包继续独立保存，此次没有重复压缩。

从仓库根目录，在具备 NumPy、ONNX、ONNX Runtime 的 `test` 环境执行；`data` 为原真实数据包解压目录：

```bash
python releases/ss928-all-models-round2-20261001/verify_delivery.py

python releases/ss928-board-v13-20260930/runtime/materialize_calibration.py \
  --input data/deployment_corrected/weather_light/train_00.npz \
  --model reports/ss928_all_models_round2_20260930/results/exact_layout_five_r2/weather_light_half_float_repeat_integer.onnx \
  --out calibration/weather_light_00 --inputs-only

python releases/ss928-board-v13-20260930/runtime/regenerate_reference.py \
  --model reports/ss928_all_models_round2_20260930/results/candidate_float_exports/weather_light_motion_float.onnx \
  --sample data/deployment_corrected/weather_light/test_00.npz \
  --out pc_weather_light_motion.npz
```

夜间两输入的精度按索引和图内声明转换，不统一强制为半精度。按原因果预处理重放整段序列：日间及天气各 120 帧、普通夜间 60 帧、特殊夜间 120 帧。

## 审查与复现证据

- [全模型审查及连续图片/视频](REVIEW.md)
- [完整数值和逐次 GPU 耗时](RESULTS.md)
- [本轮实验源码](../../reports/ss928_all_models_round2_20260930/runtime/)
- [完整图、权重、训练记录及核验](../../reports/ss928_all_models_round2_20260930/results/)
- [上一轮拖影定位及整数布局实验](../../reports/ss928_motion_speed_20260930/REVIEW.md)
- [早期天气配置作废记录](../../reports/ss928_all_models_round2_20260930/results/INVALID_EARLY_WEATHER_CONFIG.json)
- [早期日间目标区域作废记录](../../reports/ss928_all_models_round2_20260930/results/INVALID_EARLY_DAY_TARGET_ROI.json)

天气四分之一网格、日间当前帧备选、字节布局及夜间尾部合并保留在实验目录，完整质量和耗时取舍见 RESULTS；它们没有统一替换主模型。目录中的 141 张图包括实验对照，板测先按上述 21 张索引进行，避免误用退步或作废候选。

审查原件及 JSON 中 `pushed=false`、`packaged=false` 记录核验时点的状态，继续保留。本交付入口记录之后的用户推送授权；实验原件内容未重写。此处 REVIEW、RESULTS 为便于仓库浏览生成的链接副本，数值及结论保持原件。

训练和完整视频复现依赖记录中的服务器数据与 `test` 环境；ONNX 图及其参考生成可独立使用。全部图已通过 ONNX 检查，28 段完整视频已解码核验，代表连续图已人工检查；未逐帧人工观看全部视频，目标区域仍是 GT 辅助提取，测试序列属于开发对照。部署和盲测结论尚未取得。
