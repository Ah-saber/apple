# SS928时间效率改进结果（2026-09-24）

## 已确认目标

用户确认原41ms仅为模型推理。本轮以历史GPU 0.914313ms对应41ms的比例，达到理论换算≤16.7ms；SS928实测由部署端负责。即GPU均值≤0.372415ms；P95另报以观察稳定性。所有模型输入1024×1280、输出3072×3840、batch1、FP32、TF32关闭、整图参考在模型内。

本轮做了三项同预算V3实验，以及一项保留主干分辨率的V4实验。两轮旧结果均保留，未替换现有部署包。

## 训练后实测与画质

|模型|选定步数|GPU均值/P95 ms|理论换算ms|整图验证PSNR|整图测试PSNR/SSIM|裁块测试PSNR/SSIM|
|---|---:|---:|---:|---:|---:|---:|
|S03画质参考|10000|0.9324/0.9466|原部署41（用户提供）|30.603|31.357/0.96781|31.310/0.94616|
|16通道/4倍打包/3主干块|4000|0.3690/0.3715|16.55|29.898|30.510/0.95725|27.730/0.80802|
|16通道/4倍打包/2主干块|8000|0.3352/0.3375|15.03|30.110|30.930/0.96295|28.318/0.83797|
|8通道/2倍打包/1主干块|8000|0.3445/0.3476|15.45|28.455|28.622/0.95965|28.325/0.91531|

理论换算=候选GPU均值×41/0.914313。另按同轮S03均值换算的结果保存在summary.json。两种换算都不能代替板端实测，未在不同硬件之间假设真实固定倍率。
整图指标仍使用3倍输出截断后回缩与原尺寸GT比较，没有真实3倍GT。裁块指标和真实三倍输出局部图用于检查细节取舍。

## 训练协议

V3：入口打包4倍、末端1×1卷积、16通道，4/3/2主干块各随机初始化训练8000步，第一步起交替原路径与均匀原尺寸路径。原尺寸分数较好，但裁块细节指标损失较大。
V4：保留S03的2倍入口打包，8通道、1个主干块、1×1输出，先原路径8000步，再混合原尺寸2000步，总10000步；训练路径步数对齐S03当前质量参考。数据、GT、冻结归一化、中间RAW辅助损失0.1、几何增强、batch16、seed928、原20万步学习率曲线保持。
每版按验证原尺寸PSNR选择检查点，测试不参与选择。V3与V4同时改变了结构和训练顺序，不能把全部差异单独归因于其中一项。V4选定步数若小于10000，说明验证选择保留了更早的权重。
两项训练顺序测试通过，真实数据预检查跨过8000/8002切换边界；原路径张量一致，梯度与损失有限。

## 分类别及模块记录

### 16通道/4倍打包/3主干块

|划分|类别|原尺寸PSNR|SSIM|亮度偏差|裁块PSNR|裁块SSIM|
|---|---|---:|---:|---:|---:|---:|
|val|weather_light|30.0712|0.953372|1.2947|27.2991|0.791740|
|val|weather_medium|29.7258|0.950220|0.2264|26.9352|0.779742|
|test|weather_light|30.3194|0.956864|2.3723|27.6157|0.808140|
|test|weather_medium|30.7009|0.957628|1.5515|27.8451|0.807902|

|模块|独立采样均值ms|
|---|---:|
|model.down|0.00815|
|model.head.0|0.03900|
|model.body.0.norm|0.00035|
|model.body.0.conv1.rep_conv|0.03372|
|model.body.0.act|0.00427|
|model.body.1.norm|0.00131|
|model.body.1.conv1.rep_conv|0.03738|
|model.body.1.act|0.00413|
|model.body.2.norm|0.00135|
|model.body.2.conv1.rep_conv|0.03761|
|model.body.2.act|0.00417|
|thumbnail_pool.0|0.00416|
|thumbnail_pool.1|0.00872|
|model.global_reference.encoder.0|0.01565|
|model.global_reference.encoder.1|0.00960|
|model.global_reference.encoder.2|0.01812|
|model.global_reference.encoder.3|0.00897|
|reference_pool.0|0.01062|
|reference_pool.1|0.00893|
|model.global_reference.project|0.01550|
|reference_resize|0.01448|
|model.tail.0|0.00578|
|model.tail.1.rep_conv|0.03690|
|model.upsample.0|0.08493|
|model.upsample.1|0.02630|
|model.upsample.2|0.02627|
|model.upsample.3|0.02627|

结果目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-V3-20260924-B3-COMPARISON`。
候选权重：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-V3-20260924-B3-8K/checkpoints/step_000004000.pt`。
CPU ONNX相对PyTorch最大绝对误差：3.1590462e-06。
模块采样与整体计时分开，含事件与调用影响，不强求相加等于总时长。

### 16通道/4倍打包/2主干块

|划分|类别|原尺寸PSNR|SSIM|亮度偏差|裁块PSNR|裁块SSIM|
|---|---|---:|---:|---:|---:|---:|
|val|weather_light|30.2373|0.958030|2.8293|27.7458|0.822823|
|val|weather_medium|29.9830|0.955017|1.9979|27.5181|0.813202|
|test|weather_light|30.6727|0.962402|2.9302|28.1538|0.837819|
|test|weather_medium|31.1863|0.963507|1.9276|28.4826|0.838125|

|模块|独立采样均值ms|
|---|---:|
|model.down|0.00848|
|model.head.0|0.03817|
|model.body.0.norm|0.00036|
|model.body.0.conv1.rep_conv|0.03458|
|model.body.0.act|0.00438|
|model.body.1.norm|0.00154|
|model.body.1.conv1.rep_conv|0.03672|
|model.body.1.act|0.00420|
|thumbnail_pool.0|0.00519|
|thumbnail_pool.1|0.00917|
|model.global_reference.encoder.0|0.01640|
|model.global_reference.encoder.1|0.00972|
|model.global_reference.encoder.2|0.01861|
|model.global_reference.encoder.3|0.00913|
|reference_pool.0|0.01103|
|reference_pool.1|0.00907|
|model.global_reference.project|0.01617|
|reference_resize|0.01435|
|model.tail.0|0.00567|
|model.tail.1.rep_conv|0.03799|
|model.upsample.0|0.08472|
|model.upsample.1|0.02605|
|model.upsample.2|0.02627|
|model.upsample.3|0.02637|

结果目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-V3-20260924-B2-COMPARISON`。
候选权重：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-V3-20260924-B2-8K/checkpoints/step_000008000.pt`。
CPU ONNX相对PyTorch最大绝对误差：2.1457672e-06。
模块采样与整体计时分开，含事件与调用影响，不强求相加等于总时长。

### 8通道/2倍打包/1主干块

|划分|类别|原尺寸PSNR|SSIM|亮度偏差|裁块PSNR|裁块SSIM|
|---|---|---:|---:|---:|---:|---:|
|val|weather_light|28.5031|0.957034|4.7608|28.1418|0.906967|
|val|weather_medium|28.4068|0.955550|3.4506|27.9648|0.903552|
|test|weather_light|28.4376|0.958783|5.0871|28.1634|0.914581|
|test|weather_medium|28.8070|0.960515|3.9889|28.4874|0.916033|

|模块|独立采样均值ms|
|---|---:|
|model.down|0.00744|
|model.head.0|0.04456|
|model.body.0.norm|0.00046|
|model.body.0.conv1.rep_conv|0.05162|
|model.body.0.act|0.00708|
|thumbnail_pool.0|0.00576|
|thumbnail_pool.1|0.00456|
|model.global_reference.encoder.0|0.00704|
|model.global_reference.encoder.1|0.00816|
|model.global_reference.encoder.2|0.01841|
|model.global_reference.encoder.3|0.00905|
|reference_pool.0|0.01100|
|reference_pool.1|0.00878|
|model.global_reference.project|0.01559|
|reference_resize|0.02063|
|model.tail.0|0.00200|
|model.tail.1.rep_conv|0.05869|
|model.upsample.0|0.08769|
|model.upsample.1|0.02812|
|model.upsample.2|0.02627|

结果目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-V4-20260924-COMPARISON`。
候选权重：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-V4-20260924-LIGHT-MEDIUM-10K/checkpoints/step_000008000.pt`。
CPU ONNX相对PyTorch最大绝对误差：2.3841858e-06。
模块采样与整体计时分开，含事件与调用影响，不强求相加等于总时长。

## 状态与交接

本轮只训练轻/中度天气；其他三组未验证新结构。未进行新模型的板端转换、量化和实测。预处理仍是逐帧/分段离线统计；模型速度目标不代表全流程实时达标。
逐帧计时、模块耗时、对照图、视频、3倍细节诊断均保存在服务器。视频已解码检查，未完整人工观看。媒体不下载，不生成网页。
用户允许画质小幅下降但未给出具体阈值，本轮只报告实际代价，不自行宣布画质验收。原数据、GT、基线、部署包保持可复现。
