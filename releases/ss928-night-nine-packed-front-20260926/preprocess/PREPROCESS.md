# 夜间九帧输入生成约定

这里保存普通夜间与特殊夜间训练/测试时使用的源代码和校准文件快照。`ordinary/config.json`、`special/config.json` 中的绝对路径指向原服务器；在部署端应分别映射到此目录的 `shared/index.json` 与 `special/index.json`。源代码快照用于核对数值规则，不能把其中的数据集绝对路径当成部署端路径。

共同规则：按时间从旧到新组成九帧，帧顺序对应输入通道 0～8。序列片段起点之前的历史帧重复片段首帧，不使用后续帧。每帧依其校准记录的 `offset`、`scale` 转为 `(raw.astype(float32) - offset) / scale`。整帧分辨率为 1024×1280，送模型时补 batch 维，张量形状 `[1,9,1024,1280]`。`reference_thumb` 来自当前帧在整幅允许区域上的 64×64 `INTER_AREA` 缩图，使用当前帧同一组 `offset`、`scale` 归一化，形状 `[1,1,64,64]`。验证样例中的两个输入均已完成这些步骤。

## 普通夜间

使用 `shared/index.json` 的逐帧归一化参数。九帧分别归一化后，按 `shared/temporal_stack.py` 的 `causal_dynamic_correct` 做因果动态校正：前六帧中位图作参考，每帧估计共同偏移、行、列及宽尺度残差；宽尺度残差使用 OpenCV 高斯标准差 4，并乘 0.9。生成缩略图时使用当前原始 RAW 及其归一化参数，不应用该九帧动态校正。

## 特殊夜间

使用 `special/index.json` 的逐帧共享归一化参数及 `special/fixed_field.npy`、`special/raw_reference.npy`。按 `special/shared_raw_normalization.py` 中 `correct(..., input_noise=True)` 对每帧原始 RAW 减固定场，并利用参考图估计、限制动态行列偏移；随后分别归一化，组成九帧。特殊夜间不调用普通夜间的 `causal_dynamic_correct`。缩略图来自已完成固定场和动态行列校正的当前帧，再使用当前帧的共享参数归一化。

`shared/sequence_normalization.py`、两个场景的 `data.py` 及配置快照记录了片段边界、允许区域和源数据读取规则。校准文件 SHA-256 见上级目录 `MANIFEST.sha256`。校准索引针对记录中的现有数据序列；接入新数据序列前须建立匹配的校准参数和片段边界，不能直接沿用测试序列参数。
