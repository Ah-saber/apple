# TIED500 七景 SS928 部署试测

本轮交付固定为用户已查看完整视频的 **TIEDDECODE500**。七景各提供保留原模型和当前候选，共14张完整 ONNX。没有重新训练，也没有按测试结果选择权重。该候选仍有小目标、拖影、噪声和对比度问题；本次用于实测 SDK 支持、量化输出及完整纯 NPU 时间。

- [部署步骤](DEPLOYMENT.md)：准备输入、编译、输出比对和计时。
- [已知问题与 GPU 结果](KNOWN-ISSUES.md)。
- [模型索引](MODEL_INDEX.json)：确切场景、输入帧数及文件。
- [七景完整 GT 对比视频](videos/REVIEW.html)：普通夜60帧，其他各120帧，共780帧。GT显示完整图像。

每景 `models/<scene>/` 包含：

|文件|用途|
|---|---|
|retained_original.onnx|与视频相同的保留原5或3帧模型，作为板端对照|
|tied500.onnx|固定 TIED500 完整推理图，包含所有部署权重|
|calibration_00～03.npz|四份不同真实训练帧的两个输入，用于编译器校准|
|smoke_inputs.npz|固定测试第37帧的两个输入，供两角色共同使用|
|retained_original_pc_output.npz、tied500_pc_output.npz|对应完整 ONNX 在 ONNX Runtime CPU 的输出|
|smoke_GT.png|该帧完整1024×1280 GT|
|manifest.json|输入规格、来源、算子、输出字节数及导出差异|
|branch_step500.pt|研究用分支权重；不是单独可运行的完整模型|
|review_*.json|已完成的完整画质及 GPU 测时记录|

普通夜、日景、特殊夜输入5帧，天气输入3帧。所有图输入为 FP16 NCHW `raw_stack`、`reference_thumb`；输出为 FP16 `display_gray`，形状 `1×1×3072×3840`，原始字节数 **23,592,960**。输入已完成本轮归一化和因果预处理，不能再次归一化或用旧 NPZ 替换。

14张图已通过 ONNX 检查，并生成同输入的 PC 输出。直接 PyTorch 与 ONNX Runtime 的最大差为0.375～0.625灰度，逐图值见manifest。这不是板端量化允许误差。SDK 尚未编译，SS928 尚未测时。目标为完整纯 NPU **mean和P95均≤16.7ms**。

交付约377 MB，包含模型、真实输入、参考输出和七景 GT 对比视频。为避免下载历史大包，可使用原分支的稀疏检出：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  --branch codex/ss928-night-nine-factor24-20260926 \
  https://github.com/Ah-saber/apple.git apple
cd apple
git sparse-checkout set releases/ss928-tied500-deploy-20261012
```

`source/` 保存本轮分支、图优化和输入处理实现片段。其研究复现仍依赖原训练服务器的原参考模型和配置；部署运行完整ONNX不依赖这些训练目录。导出入口为 `export_tied_deploy12_fix.py`，原导入失败记录保存在 `evidence/`。
