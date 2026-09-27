# 接手状态：九帧NPU模型继续优化

## 用户目标与持续授权

SS928完整模型≤16.7ms，允许少量画质下降，尽量保持已解决去噪／闪烁／连续帧亮度和局部对比度／弱目标。主机读取、输入表示准备和传输不在模型计时范围。用户要求持续实际修改、实际GPU测量、完整整图视频，改到需要板端新信息时提交推送。既有授权持续有效：apple原分支提交推送；本轮明确回复允许上传代码到5090_media#gpu的/data/zhangbenzhuang/huawei_sr并继续训练测速。

不得新建分支、偷偷换远程、挑单轮最短GPU值、凭GPU推算NPU、把统计和完整输出挪到CPU宣称达标。中文直接沟通。真实native_r2／SDK／板端输出仍不可访问，用户已说明；不要反复索要不可访问文件。

## 冻结版本与当前改动

上一版apple提交cada77899084bf05e4c768cecc75d43aca389065，分支codex/ss928-night-nine-factor24-20260926，远程https://github.com/Ah-saber/apple.git。原发布目录ss928-night-nine-v07-structural-20260927保留。

本轮：

1. 普通／夜间各线性四相位和ReLU八→四相位4000步训练；直接四相位亮度／对比度退化，固定±4/9灰度相位偏移修复大部分舍入问题，但改变真实三倍纹理。
2. 输出九相位补到16通道；Float32小相位裁剪与灰度缩放提前；Half NCHW九帧输入组合。当前GPU默认候选combo_input9_half_aligned16_nearest，输出仍完整Float32三倍图。
3. 输出卷积＋重排直接学习反卷积、Tail3＋Out3组成Conv5、真边缘计算、输出单固定反卷积与255权重吸收。失败／变慢／边缘损坏明确保留。
4. 固定统计正负成对ReLU24／32；Half NCHW16输入；参考12通道先放大后投影；独立Half完整输出。
5. 原统计2×2放入3×3右下角、padding1；保持偶数传感器统计位置。三版独立对照与完整combo_stats3_relu32_aligned16_half_output_fixed组合均已实现和测量。
6. transplant_output.py受约束替换native的Conv36及下游六个反卷积路径；rewrite_statistics3.py、move_output_scale.py等有独立CPU数学／ONNX模块控制。真实R2未经验证。

## 实测结论

GPU正／反顺序复测，普通上一组合0.347／0.352ms，新Half输入对齐组合0.321／0.322ms；夜间上一组合0.354／0.340ms，新组合0.320／0.305ms。纯Float32输入输出对齐也有3.9～6.6%收益。缩放提前组合初测普通0.324ms，随后0.348～0.353ms，不能声称稳定收益。

新Half输入对齐组合完整180帧亮度／对比度时序指标保持，两个既有普通弱目标窗口响应相同；Float输出少数像素Half舍入差异普通最大0.125／夜间0.249灰度。缩放提前及Half九输入缩放提前的源码完整逐帧输出与上一组合一致。编译器与NPU一致性另算。

完整NPU测试组合combo_stats3_relu32_aligned16_half_output_fixed的GPU普通0.572／夜间0.556ms，比上一组合慢；目的是改变SS928约17ms固定统计与六路输出的映射，并单测Half完整输出Report。PSNR相对上一组合普通+0.000956／夜间−0.000353dB，Half舍入使时序亮度／对比度小幅变化，完整结果和视频保留。不能称这个组合GPU提速或预言NPU达标。

原v0.7板端仍普通81.228／特殊80.351ms。16.7ms没有验证达到，不能宣布优化完成或全部可能性已穷尽。

## 所有证据与复现

本地research：/mnt/d/work_project/rawir/reports/ss928_v07_native4_20260927。
远程code：/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927/code_snapshot。
远程产物：/data/zhangbenzhuang/huawei_sr/runs/SS928-V07-NATIVE4-20260927。
test解释器：/data/zhangbenzhuang/miniconda3/envs/test/bin/python，Torch2.7.0+cu128；ONNX依赖从bfstvsr site-packages追加，不运行base或安装。
GPU1：GPU-655fc72e-3e1f-236c-ee8b-5ee23e9bd264，独占runs/TASK-019-gpu1.lock，2.5GiB配额，TF32关闭，不并行GPU作业。

上轮审批拒绝/tmp代码目的地，本轮只上传Python代码到明确授权/data目录。整实验目录上传亦被拒绝（包含模型、证据缓存超出代码范围），没有再次上传整目录或绕过。代码同步后执行获准。

完整源图92个，源码派生兼容控制24个，冻结源4个；全部ONNX结构核对。四个新训练PT与四个上一版PT保留。完整视频含GPU默认候选、四相位纹理取舍候选、Half输出候选及完整NPU测试组合，普通60帧／夜间120帧、12fps、5120×1080四栏，每栏完整RAW视野；PNG为3072×3840。

指标与范围见RESULTS.md；板端候选与现成改写命令见DEPLOYMENT.md。GT只有RAW尺寸；普通测试ROI[2,0,1020,1278]、夜间[2,1088,1020,192]，两场景数据已经用于结构评估，非盲测。弱目标投影只覆盖两个窗口，非召回率。

## 下一步具体需要的信息

部署端对现有独立版和完整组合做SDK转换／AICPU／完整同步计时／任务分解，判断3×3或ReLU统计是否规避固定2×2热点、单固定反卷积是否真的替代六路输出、Half NCHW输入与Half完整输出是否减轻格式转换。保留native既有参考兼容处理，用受约束工具；工具拒绝说明实际图不满足已验证条件，不绕过。

这是后续NPU选择与进一步重排的缺失证据。服务器上的实现、训练、完整序列、GPU、弱目标和视频已完成；不要停在“有设想”或“等审批”。若新板端结果返回，按真实新热点继续改。未经新NPU数据，不能把更慢GPU图当成NPU已改进结论。

## 提交推送

用/tmp/rawir-apple-structural-sparse-20260927原分支，按提交回溯，不创建分支；尊重根工作区其他未提交修改。代理127.0.0.1:7890用于GitHub现有连接，命令级覆盖；不改变全局配置，不输出认证信息。以发布目录提交，本轮交付状态与SHA见DELIVERY_STATUS.md（提交后保存）。
