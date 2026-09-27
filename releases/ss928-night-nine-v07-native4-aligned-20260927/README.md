# 九帧模型继续优化：输出对齐与四相位训练

目标为SS928完整模型推理≤16.7ms，保持去噪、闪烁、同场景连续帧亮度／对比度及弱目标。上一版冻结在apple提交cada77899084bf05e4c768cecc75d43aca389065。本轮实际训练、完整180帧检查与GPU实测已完成，NPU目标尚未达到验证条件。

Half表示16位浮点；NCHW按批次、通道、高、宽排列。输出相位指每个低分辨率特征位置预测的不同输出像素。

## 当前可采用的GPU候选

`combo_input9_half_aligned16_nearest`保留九个已学习的输出相位，补到16通道后取回原九通道；裁剪和255缩放移到小相位张量，完整输出仍为3072×3840。输入为Half NCHW九帧，参考缩图维持Float32。上一版原本在模型内把RAW转为Half；本版输入直接声明Half。接口类型准备排除在模型推理计时之外，所有时域统计与输出展开仍由模型执行。

两轮正反顺序复测，普通0.321～0.322ms，夜间0.305～0.320ms；同轮上一版普通0.347～0.352ms，夜间0.340～0.354ms。两场景亮度／对比度时序指标与上一版相同，两处已记录弱目标窗口响应相同。完整浮点输出仍有少量Half卷积舍入差异，普通最大0.125、夜间最大0.249灰度。不能称逐像素无损。

保留Float32输入的`combo_aligned16_nearest`在同轮下降约3.9～6.6%，可单独分离输出通道对齐与输入类型影响。

详情见 [RESULTS.md](RESULTS.md)、[DEPLOYMENT.md](DEPLOYMENT.md) 和 [HANDOFF.md](HANDOFF.md)。GPU数字不能换算成SS928数字。

## 四相位训练候选

普通／夜间各训练线性四相位和ReLU八通道→四相位4000步。只用冻结教师的训练区特征、裁剪舍入后的像素均值，不用测试GT选择权重。直接四相位复制有连续帧亮度／对比度退化；增加固定小于0.5灰度的相位偏移后，时序指标接近或略好于上一版。`native4_linear_ref_dither_float_trained`有完整序列、弱目标和视频证据，但会改变三倍纹理，GPU耗时也有明显波动，保留作画质取舍候选。

## 完整视频

`videos/`包含普通60帧与夜间120帧，12fps、5120×1080四栏视频。每栏均是完整1024×1280视野：RAW、GT、cada778上一版、新候选。完整三倍结果先舍入，再按3×3平均用于同视野比较；PNG保留3072×3840输出。图像中间帧仅辅助观察，不能代替视频判断闪烁。候选名称与源码执行状态记录在对应video.json。

## 文件与重现

- `runtime/`模型实现；`models/`四个本轮训练检查点和四个上一版冻结检查点。
- `source_onnx/`92个完整／小图源导出；`frozen_source/`原v0.7 36相位对照；`compatibility_controls/`源码派生的工具组合控制图。
- `evidence/`完整原始GPU计时、每帧指标、逐像素差异、弱目标、训练与输出权重8位压力检查。压力检查未模拟SDK激活量化、偏置量化或校准。
- `tools/`受约束输出替换、缩放提前及上一版统计／参考重写工具，未知结构会拒绝。
- 根目录保留实际训练、计时、画质、ONNX与视频脚本；研究数据路径固定，迁移数据时需调整。

使用test环境。Torch源码加载依赖apple同级冻结目录`ss928-night-nine-system-speed-20260926/`及其`ss928-night-nine-packed-front-20260926/`；可用`RAWIR_SYSTEM_RELEASE_DIR`指定前者。独立ONNX与部署改写工具不依赖Torch目录。

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path('runtime').resolve()))
from continuation_candidates import load_continuation, prepare_inputs
model = load_continuation('ordinary', 'combo_input9_half_aligned16_nearest', Path('models'), device='cuda')
x, context = prepare_inputs(model, nine_raw_float32, reference_thumb_float32)
y = model(x, context)  # 完整Float32灰度图
```

`SHA256SUMS`、`MANIFEST.json`覆盖交付文件，未知SDK兼容性与NPU耗时明确标记。原版本和失败实验保留，当前研究数据已经用于结构评估，不是独立盲测集。
