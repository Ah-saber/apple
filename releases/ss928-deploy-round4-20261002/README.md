# SS928 第四轮部署试测材料

用户于2026-10-02明确要求推送部署需要的模型、问题和文档，准备部署试测。所有模型独立保存，不替换原部署；此目录不需要训练服务器的代码或权重即可使用。

[MODEL_INDEX.json](MODEL_INDEX.json) 列出七类各一张第三轮同权重控制和一张第四轮候选，共14张完整ONNX。图均输出 `1×1×3072×3840` Float16。候选改为分块参考采样和普通卷积输出复制，原学习权重保持；同直接运行后端660次对照全部逐像素一致。

本次使用原半网格九帧及原夜间九帧权重，先验证同权重布局收益。既有天气三帧图继续保留；尚未把本轮分块改写与天气三帧权重组合验证，不能混用两类权重的指标。

| 场景 | 完整卷积乘加减少 | 模型目录 |
|---|---:|---|
| 日间 | 7.24% | [day_normal](models/day_normal/) |
| 轻天气 | 12.39% | [weather_light](models/weather_light/) |
| 中天气 | 12.39% | [weather_medium](models/weather_medium/) |
| 重天气 | 12.39% | [weather_heavy](models/weather_heavy/) |
| 重天气C32 | 12.55% | [weather_heavy_c32](models/weather_heavy_c32/) |
| 普通夜间 | 5.43% | [night_ordinary](models/night_ordinary/) |
| 特殊夜间 | 2.36% | [night_special](models/night_special/) |

**以上是计算量减少，实际速度未测。** 目标仍为完整纯NPU均值/P95均≤16.7毫秒。先编译与单次完整输出核对，再对每类 `control.onnx` 和 `candidate.onnx` 同条件计时及量化画质比较。

- [部署顺序、输入准备、参考与输出比对](DEPLOYMENT.md)
- [已知拖影、量化和板卡问题](KNOWN-ISSUES.md)
- [真实校准及测试输入的112项身份记录](CALIBRATION_INDEX.json)
- [原v0.13部署反馈](evidence/REPORT-v0.13-all-in-one.md)
- [本轮计算量及一致性证据](evidence/STRUCTURAL-COMPARISON.json)
- [14张图真实输入CPU推理与部署工具检查](evidence/TOOL-QA.json)

真实数据复用已交付 `SS928-V13-REAL-CALIBRATION-AND-GT-20260930.tar`，8,420,392,960字节，SHA-256为 `d748b9582cca5aab1cd85760c866e351de8925ff916590515cf7b9be74483a59`。无需新打包或重复传输此数据；准备输入工具会逐项校验源NPZ、历史身份、图哈希、形状和精度。

这批主要测试结构布局能否改善板端速度；拖影没有新增改善结论。失败的夜间前端、未实测的新增检测器和训练脚本不在此部署目录。

在test环境从仓库根目录校验：

```bash
python releases/ss928-deploy-round4-20261002/runtime/verify_delivery.py
```

校验工具仅需NumPy；PC参考生成还需要ONNX Runtime。使用部署端原test环境和既有SDK编译/运行入口，具体命令见部署文档。本目录尚无新SDK编译、NPU计时或量化连续画质结果。

交付前，已从原包读取七类真实训练/测试输入，完成14张图的CPU参考与BIN核对；控制/候选CPU完整输出最大差0.125灰度。输出比较工具以CPU结果模拟板端文件做流程检查，未因此取得NPU或板卡证据。
