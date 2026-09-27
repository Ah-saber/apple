# 部署端验证：保留原生兼容处理，逐项测模型

## 目标与当前边界

目标SS928完整模型推理≤16.7ms。v0.7板端同步耗时普通81.228／特殊80.351ms，统计卷积约17ms、完整输出路径约47～48ms。当前64个基础源图及补充Half输出图均未经过SS928 SDK，GPU速度不能推算NPU。

本轮已继续做完实际训练、完整180帧画质、逐像素差异、弱目标、输出权重量化压力检查、完整视频。以下候选已实现可测，所有源ONNX输出完整三倍图像。

## 实际候选与要验证的耗时

| 候选 | 修改 | 需要板端判定 |
|---|---|---|
| combo_scaled9_exact | Float32裁剪／255缩放在九相位上完成 | 原Concat＋缩放任务是否减少；变为Float32展开是否反而变慢 |
| combo_aligned16_nearest | 输出九相位补零到16，取九相位重排 | 逻辑16通道能否改变卷积与布局映射；GPU已提速 |
| combo_aligned16_fixed | 16输出通道＋单个固定6×6反卷积 | 能否真正替换六个输出重排任务；GPU变慢，NPU未知 |
| combo_aligned16_folded_fixed | 255吸收到输出权重；小图裁剪＋单反卷积 | 小图灰度处理、完整输出展开；有Half／量化差异 |
| combo_aligned16_half_output_fixed | 权重不乘255，Half小相位裁剪／缩放＋单反卷积，完整Half输出 | 输出Report及展开任务变化；独立Half输出接口 |
| combo_aligned16_nearest_half_output | Float32输入，Half小相位缩放，完整Half输出 | 单独Half输出格式转换影响 |
| combo_relu_stats32_scaled9_exact | 正负成对＋ReLU恢复12统计，补齐32统计通道 | 固定统计卷积是否换映射，输入允许负值；GPU不保证收益 |
| combo_input9_half_scaled9_exact | 单独Half NCHW九帧输入，完整Float32输出 | 输入格式转换能否减少，避免联动Half NHWC与16行输出 |
| combo_input16_half_native_stats_scaled9_exact | Half NCHW前九帧＋七个零通道，原2×2统计卷积后七权重零 | 九输入通道的转换与固定统计映射；输入契约变化 |
| combo_project_after_resize_scaled9_exact | 12通道先放大，再12→16投影 | 参考放大少四通道，增加全分辨率1×1，能否净收益 |
| native4_linear_ref_dither_float_trained | 实训四相位＋固定微小偏移 | 三倍纹理取舍、浮点Tile／Add兼容与输出成本 |
| combo_tail5_exact_edges | Tail3＋输出3组成Conv5，额外完整边缘计算 | 完整边缘切片代价；GPU已变慢，量化压力较大 |

`combo_tail5_interior`损坏图像边缘，保留失败对照，不作为部署质量候选。直接四相位复制版时序亮度／对比度退化，不能只按PSNR采用。

## 六个输出反卷积路径的替换工具

`tools/transplant_output.py`定位并核对冻结的Conv16→36权重／偏置，提取新源图独立输出模块，替换原输出卷积及其下游六路反卷积、拼接、缩放等路径。保持原生前端、Body和参考分支已有兼容节点，按新源图声明输出类型。输出模块入口加对应精度Cast，真实量化与边界差异需复核。

支持常规四／八／九／十六通道3×3输出入口，完整输出形状必须与native一致；未知权重、附加输入、输出形状不符、共享路径和Tail5会拒绝。真实native_r2访问不到，仅完成合成六反卷积路径的控制验证，不能宣称已验证真实R2或SDK。

若起点为v0.7原始native_r2，先采用本包冻结继承工具，加入已训练参考ReLU与空间统计；然后替换输出。只分离输出耗时可从原native起点直接替换，但不能用组合源码的画质结果代替该输出单改版。

```bash
python tools/rewrite_reference_relu.py \
  --native /path/to/ordinary_native_r2.onnx \
  --old-source frozen_source/ordinary_baseline.onnx \
  --new-source source_onnx/ordinary_previous_combo.onnx \
  --output /path/to/new_unique/ordinary_reference.onnx
python tools/rewrite_hotspots.py \
  --input /path/to/new_unique/ordinary_reference.onnx \
  --statistics space_pack \
  --output /path/to/new_unique/ordinary_front.onnx
python tools/transplant_output.py \
  --native /path/to/new_unique/ordinary_front.onnx \
  --frozen-source frozen_source/ordinary_baseline.onnx \
  --source source_onnx/ordinary_combo_aligned16_fixed.onnx \
  --output /path/to/new_unique/ordinary_aligned16_single_deconv.onnx
```

特殊场景替换ordinary为special。Half完整输出改用`combo_aligned16_half_output_fixed`源图；输出类型随源图更新，板端按Half读取，仅表示灰度图的数值类型，模型仍完成全部展开。工具保持native输入类型，不能把输入Half源图当作已经改变native输入的证明。独立Half输入源图需沿用部署侧已验证兼容转换流程，并明确声明其输入类型、通道数及输出类型。

## 数值与测速回传

沿用v0.7同编译／校准／同步计时协议。记录编译失败、AICPU、完整同步耗时和任务分解，重点辨认：输入转换、固定统计、参考放大、输出学习卷积、单固定反卷积、灰度缩放、Report。模型内部Cast／布局／统计／输出展开计入；主机RAW读取、准备输入类型、拷贝及输出落盘排除。

Float32输入形状[1,9,1024,1280]；Half九／十六通道型号相应改变类型／通道，16通道第9～15通道严格为零。参考输入保持Float32[1,1,64,64]。输出均完整[1,1,3072,3840]，灰度0～255，默认Float32，名字含half_output为Half。

训练与GPU结果已提供，下一步未知项具体为SS928转换器如何处理这些已有候选。应同时测独立改动与组合，并逐帧对照原生输出；单个GPU值不能决定NPU路径。没有可靠的16.7ms达标预测。

## 标准3×3统计的完整组合已实现

`combo_stats3_relu32_aligned16_half_output_fixed`已完成两场景完整180帧源推理、弱目标、GPU测量与导出：固定统计2×2改为右下角3×3核＋正负ReLU32，参考ReLU沿用上一版训练参数，输出16通道＋单固定反卷积，完整Half灰度图。GPU普通约0.572、夜间约0.556ms，比上一GPU组合慢；不能称GPU提速版本。这是针对SS928统计和输出热点的完整待测图，NPU收益未知。

从原v0.7 native_r2的reference改写图继续，统计3×3替代space_pack步骤：

```bash
python tools/rewrite_statistics3.py \
  --native /path/to/new_unique/ordinary_reference.onnx \
  --rectified32 \
  --output /path/to/new_unique/ordinary_stats3_relu32.onnx
python tools/transplant_output.py \
  --native /path/to/new_unique/ordinary_stats3_relu32.onnx \
  --frozen-source frozen_source/ordinary_baseline.onnx \
  --source source_onnx/ordinary_combo_stats3_relu32_aligned16_half_output_fixed.onnx \
  --output /path/to/new_unique/ordinary_stats3_relu32_single_deconv_half.onnx
```

`rewrite_statistics3.py`严格核对原统计系数、已知偶数NCHW尺寸；不接受已经space_pack替换后的图。工具的实际R2兼容仍需部署端核实。所有独立版保持可用，用来区分统计、输出展开和Half输出的贡献；编译失败也应保留，不绕过保护。
