# 本轮结果与证据边界

冻结比较版本：apple cada77899084bf05e4c768cecc75d43aca389065。目标SS928完整推理≤16.7ms尚未验证达成。

## GPU同轮正反顺序复测

RTX5090 GPU1、Torch2.7.0+cu128、test环境。30次预热，3×200个单次CUDA事件，TF32关闭，无CUDA图；完整输入与完整三倍输出。仅输入表示准备排除。计时包含模型调用中的CPU派发间隙。不能用GPU数值推算NPU。

| 场景／轮次 | 上一组合 ms | Float32输入、16输出通道 ms | Half输入、16输出通道 ms | Half输入组合下降 |
|---|---:|---:|---:|---:|
| ordinary/paired_final_benchmark | 0.347151 | 0.333489 | 0.320596 | 7.65% |
| ordinary/paired_reverse_benchmark | 0.351951 | 0.334483 | 0.322350 | 8.41% |
| special/paired_final_benchmark | 0.354405 | 0.331154 | 0.319826 | 9.76% |
| special/paired_reverse_benchmark | 0.340455 | 0.320987 | 0.304675 | 10.51% |

第一次缩放提前组合普通0.324ms／特殊0.316ms，后续多轮普通0.348～0.353ms。因此没有采用第一次数值宣称稳定提速。四相位带偏移组合正反顺序普通0.311／0.333ms、夜间0.289／0.321ms，波动同样保留。

## 完整序列画质

普通60帧、夜间120帧，模型每帧完整1024×1280输入、3072×3840输出。显示灰度裁剪舍入后按3×3平均回RAW尺寸。GT只有RAW尺寸，不能凭此判断真实三倍纹理。PSNR／SSIM与时序指标按既有测试ROI：普通[2,0,1020,1278]、夜间[2,1088,1020,192]；完整视频均无裁剪。

| 场景／模型 | PSNR dB | 静态时序误差 灰度 | 亮度时序误差 灰度 | 对比度时序误差 灰度 |
|---|---:|---:|---:|---:|
| ordinary/previous_combo | 27.03020294 | 1.61190593 | 0.53117749 | 0.70557338 |
| ordinary/combo_input9_half_aligned16_nearest | 27.03020264 | 1.61190581 | 0.53117749 | 0.70557338 |
| ordinary/native4_linear_ref_dither_float_trained | 27.03329072 | 1.60427117 | 0.52669203 | 0.70148079 |
| special/previous_combo | 26.09642396 | 1.84791815 | 0.77101640 | 1.09302738 |
| special/combo_input9_half_aligned16_nearest | 26.09642551 | 1.84791732 | 0.77101640 | 1.09302738 |
| special/native4_linear_ref_dither_float_trained | 26.09670334 | 1.84073782 | 0.77070939 | 1.08987158 |

上述时序误差均相对GT计算；静态掩码、8×8分区分位数与公式见evaluate_joint.py。两场景已有数据用于结构选择，未进行独立盲测。

## 完整浮点输出逐帧核对

| 场景／模型 | 所有帧逐像素相同 | 最大灰度差 |
|---|---|---:|
| ordinary/combo_scaled9_exact | True | 0.000000000 |
| ordinary/combo_aligned16_nearest | False | 0.124511719 |
| ordinary/combo_input9_half_scaled9_exact | True | 0.000000000 |
| ordinary/combo_input16_half_native_stats_scaled9_exact | False | 0.249023438 |
| ordinary/combo_input9_half_aligned16_nearest | False | 0.124511719 |
| ordinary/combo_input16_half_native_stats_aligned16_fixed | False | 0.249023438 |
| special/combo_scaled9_exact | True | 0.000000000 |
| special/combo_aligned16_nearest | False | 0.249023438 |
| special/combo_input9_half_scaled9_exact | True | 0.000000000 |
| special/combo_input16_half_native_stats_scaled9_exact | False | 0.249023438 |
| special/combo_input9_half_aligned16_nearest | False | 0.249023438 |
| special/combo_input16_half_native_stats_aligned16_fixed | False | 0.249023438 |

source未编译执行。编译模型与自身源码的灰度误差记录在所有GPU计时文件；最终组合单帧均值差约0.043／0.049灰度。不能把源码逐像素一致等同于编译器或NPU一致。

## 弱目标窗口

| 普通帧号 | 上一组合响应 | Half输入对齐响应 | 四相位偏移响应 |
|---|---:|---:|---:|
| 46 | 0.335238934 | 0.335238934 | 0.332486004 |
| 58 | 0.557518780 | 0.557518780 | 0.562687814 |

这是GT定义的低幅时域残差投影，检测召回率未标注。只检查既有两个32×32窗口，不能覆盖全部弱目标。

## 四相位训练与舍入修正

普通／夜间各训练线性四相位、ReLU八→四相位4000步；192个成对特征块来自训练ROI，缓存教师全图特征和完整参考尺度。监督为冻结教师裁剪／舍入后的四个原生像素均值，不使用GT或测试区选择检查点。损失含均值、时域、弱时域、局部均值与梯度约束。随机种子2783，线性学习率1e-5，ReLU版本2e-5，固定最终4000步。全部四份PT、训练样本和损失日志保留。

直接最近邻复制四相位使每个RAW像素对应九个显示像素同值，舍入后只有整数均值；原九相位允许1/9灰度步进。固定3×3偏移在±4/9灰度之间，有助保留均值小数，但引入固定纹理。它无法恢复原三倍细节；不能仅按RAW指标采用。

## 独立Half完整输出

| 场景／模型 | GPU ms | 相对上一组合PSNR dB | 亮度时序误差变化 | 对比度时序误差变化 |
|---|---:|---:|---:|---:|
| ordinary/combo_aligned16_nearest_half_output | 0.321805 | +0.00095602 | +0.00347269 | +0.00431144 |
| ordinary/combo_aligned16_half_output_fixed | 0.516290 | +0.00095602 | +0.00347269 | +0.00431144 |
| special/combo_aligned16_nearest_half_output | 0.304047 | -0.00035261 | -0.00160169 | -0.00214087 |
| special/combo_aligned16_half_output_fixed | 0.508845 | -0.00035261 | -0.00160169 | -0.00214087 |

两版本输入仍为Float32 NCHW九帧；输出Half完整灰度图。单反卷积版在GPU变慢，针对NPU六路展开和Report转换单独验证。未把上轮多项联动Half NHWC失败实验当作独立Half输出结果。

## 未采用方案与NPU专用对照

- 学习输出卷积与重排直接合成ConvTranspose：GPU约0.69～0.71ms，完整stride6约1.57～1.59ms，拒绝作GPU默认。
- Tail3与输出3组成Conv5：内区等价，忽略边界的全图最大误差普通约103灰度／夜间约63灰度；补齐真实边缘后GPU约0.39～0.43ms，输出权重8位压力误差也比原投影大。近似版本禁止作为保细节版本。
- 固定统计正负成对ReLU、逻辑32通道：完整序列指标保持，GPU部分场景变慢。NPU固定统计任务是否换映射未知。
- 参考投影移到双线性放大之后：保留权重与多尺度分支，Float64数学验证通过；Half会改变舍入，普通对比度时序误差稍升，GPU收益不一致。
- 输入Half NCHW16，前九帧／后七零通道：保持数学统计，GPU无稳定收益；与Half NHWC9分开保存。
- 原统计2×2移到3×3右下角、padding1：要求偶数传感器尺寸，完整原始统计位置与权重保留。正负32通道版本也已实现。名义运算增加2.25倍，GPU变慢，不作为GPU提速结论；测试目的为NPU标准3×3映射是否规避原2×2固定统计约17ms热点。

## 输出权重8位压力检查

只对输出学习卷积与固定展开权重作对称8位压力，分整张权重／每输出通道；反卷积输出轴为1。未模拟激活16位校准、偏置量化和SDK混合精度。原九相位与16对齐版本在这项压力检查相同；Tail融合误差明显更大。固定偏移版本用1/255常量特征配合±4/9核，避免直接存储±4/(9×255)核被整张权重8位量化抹掉。

## 已验证与仍需板端确认

ONNX结构、受约束替换、统计核重排、完整Source推理、训练、GPU计时均分别留存。统计3×3工具24项独立ONNX模块控制；输出缩放工具12项实际输出模块控制；输出替换工具18项合成六反卷积路径控制。实际native_r2访问不到，源码派生控制图保留参考兼容节点仅作工具练习。没有把合成图当作真实板端图。

下一步需要部署端对已有完整组合与独立改动给出SDK转换、AICPU、完整同步耗时和任务分解；尤其单反卷积、3×3统计、Half完整输入／输出和参考放大映射。没有可信的NPU总耗时预测，16.7ms目标尚未达成。

## 统计3×3与完整NPU候选实测

| 场景／模型 | GPU ms | 最大完整输出差 灰度 | 相对上一组合PSNR dB |
|---|---:|---:|---:|
| ordinary/combo_stats3_scaled9_exact | 0.366499 | 0.000000000 | +0.000000000 |
| ordinary/combo_stats3_relu32_scaled9_exact | 0.374527 | 0.000000000 | +0.000000000 |
| ordinary/combo_input16_half_stats3_scaled9_exact | 0.374351 | 0.000000000 | +0.000000000 |
| ordinary/combo_stats3_relu32_aligned16_half_output_fixed | 0.572165 | 0.187011719 | +0.000956021 |
| special/combo_stats3_scaled9_exact | 0.362349 | 0.009727478 | +0.000000000 |
| special/combo_stats3_relu32_scaled9_exact | 0.379152 | 0.009727478 | +0.000000000 |
| special/combo_input16_half_stats3_scaled9_exact | 0.368260 | 0.000000000 | +0.000000000 |
| special/combo_stats3_relu32_aligned16_half_output_fixed | 0.555578 | 0.241699219 | -0.000352651 |

完整NPU候选GPU比上一组合慢，针对的是板端约17ms统计与约47～48ms输出路径的映射。独立版本、完整组合、完整视频和受约束替换均已提供；NPU不能根据这些GPU数字作结论。
