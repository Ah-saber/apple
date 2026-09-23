# S03完整训练与测试结果

已从8000步续训至200000步。验证最佳权重为第126000步，最后权重为第200000步。选择依据始终为验证集裁块PSNR，整图指标和测试集没有参与选权重。训练进程退出码为0。

## 结果

|模型|步数|验证裁块PSNR/SSIM|验证整图PSNR/SSIM|测试裁块PSNR/SSIM|测试整图PSNR/SSIM|
|---|---:|---|---|---|---|
|Original reference 8k|8000|31.1255 / 0.944776|30.5926 / 0.962062|31.7931 / 0.952711|31.5091 / 0.969518|
|S02 8k|8000|31.8536 / 0.950595|30.0750 / 0.960128|32.7344 / 0.958089|31.1050 / 0.967083|
|S03 8k|8000|30.6077 / 0.938804|29.9919 / 0.957638|31.4436 / 0.948012|30.9903 / 0.964782|
|S03 best 126000|126000|32.6890 / 0.959446|28.7991 / 0.956688|33.9563 / 0.966947|30.2562 / 0.965109|
|S03 last 200000|200000|32.5633 / 0.959535|28.8659 / 0.955991|33.7928 / 0.967092|30.3111 / 0.964583|

PSNR单位dB。SSIM为结构相似程度。两场景等权；每个划分每场景12帧。整图流程为原尺寸RAW→网络3倍输出→裁剪[0,1]→面积回缩3倍→同一区域GT比较。没有真实3倍高分辨率GT，因此另存真实3倍局部图，GT放大仅作风格参考。

相对S03 8000步，验证最佳权重在val的整图PSNR变化为-1.1928 dB、SSIM变化-0.000950；裁块PSNR变化+2.0813 dB。
相对S03 8000步，验证最佳权重在test的整图PSNR变化为-0.7341 dB、SSIM变化+0.000327；裁块PSNR变化+2.5127 dB。

不能预先认定延长训练会改善整图效果；以表中实际结果及图片、视频判断。这里比较训练预算不同的完整方案，不是同预算结构消融。测试集此前已用于开发分析，不能称为全新盲测。

## 各场景

|划分|场景|模型|整图PSNR|整图SSIM|亮度偏差/灰阶|
|---|---|---|---:|---:|---:|
|val|weather_light|baseline|30.7828|0.964122|1.3417|
|val|weather_light|s02|30.3827|0.962612|-1.3052|
|val|weather_light|s03_8k|30.2269|0.959964|0.4501|
|val|weather_light|s03_best|29.2213|0.959491|-3.4865|
|val|weather_light|s03_last|29.3010|0.958839|-3.1442|
|val|weather_medium|baseline|30.4023|0.960001|0.4976|
|val|weather_medium|s02|29.7673|0.957644|-2.1162|
|val|weather_medium|s03_8k|29.7570|0.955312|-0.3263|
|val|weather_medium|s03_best|28.3768|0.953885|-4.5187|
|val|weather_medium|s03_last|28.4308|0.953143|-4.2145|
|test|weather_light|baseline|31.3629|0.969200|1.3928|
|test|weather_light|s02|31.1243|0.967296|-1.0473|
|test|weather_light|s03_8k|30.9499|0.964973|0.3902|
|test|weather_light|s03_best|30.3709|0.965701|-2.7954|
|test|weather_light|s03_last|30.4389|0.965196|-2.6718|
|test|weather_medium|baseline|31.6553|0.969836|0.3606|
|test|weather_medium|s02|31.0857|0.966869|-1.9127|
|test|weather_medium|s03_8k|31.0306|0.964591|-0.6501|
|test|weather_medium|s03_best|30.1414|0.964517|-3.6911|
|test|weather_medium|s03_last|30.1834|0.963969|-3.6423|

## 推理耗时

|版本|均值ms|P50 ms|P95 ms|
|---|---:|---:|---:|
|s03_8k|0.9946|0.9943|1.0006|
|s03_best|0.9981|0.9981|1.0030|
|s03_last|0.9963|0.9963|1.0025|
|s03_8k_fastpool|0.9091|0.9087|0.9146|
|s03_best_fastpool|0.9143|0.9143|0.9192|
|s03_last_fastpool|0.9095|0.9087|0.9149|

5090共享环境，FP32、关闭TF32，1×1×1024×1280归一化RAW输入，3倍输出；包含整图参考。fastpool为等价分级均值池化。每模型预热50次后3×200次同步计时，另存逐模块耗时。排除归一化统计、CPU/GPU传输、8bit转换和文件读写。SS928尚未实测。

## 连续视频

视频按RAW、GT、S03 8000步、验证最佳、最后200000步排列。每个验证/测试序列使用全部连续帧；另存真实3倍局部视频。统一[0,1]映射8bit，没有逐图拉伸。播放12fps仅用于查看，不代表已确认的采集帧率。

|划分/场景|模型|平均亮度偏差|亮度偏差时间标准差|连续残差变化MAE|
|---|---|---:|---:|---:|
|val/weather_light|s03_8k|0.48439|0.02910|2.68884|
|val/weather_light|s03_best|-3.47704|0.03960|2.69718|
|val/weather_light|s03_last|-3.13022|0.04101|2.71325|
|val/weather_medium|s03_8k|-0.30031|0.02950|2.98404|
|val/weather_medium|s03_best|-4.51629|0.03961|2.99431|
|val/weather_medium|s03_last|-4.20730|0.04218|3.01231|
|test/weather_light|s03_8k|0.39310|0.03449|2.19454|
|test/weather_light|s03_best|-2.81649|0.04717|2.19844|
|test/weather_light|s03_last|-2.68847|0.04941|2.21253|
|test/weather_medium|s03_8k|-0.63754|0.04388|2.17473|
|test/weather_medium|s03_best|-3.70045|0.05530|2.18023|
|test/weather_medium|s03_last|-3.64894|0.05731|2.19434|

连续残差变化为相邻两帧(prediction−GT)差值的平均绝对值，跳过冻结分段边界。没有做运动对齐，指标仍受运动、纹理和GT变化影响，不能单独证明无闪烁；实际视频需人工查看。

## 训练与选择记录

训练目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-LIGHT_MEDIUM-200K`。权重和Adam状态、随机状态、采样epoch与next_batch均继承原8000步检查点，原学习率20万步余弦曲线不变。模型16通道4块，RAW几何增强与0.1辅助RAW损失保持。每2000步验证与保存，测试仅在200000步及最终自动评估进行。原父目录保持不变。

归一化仍使用完整序列的离线统计，未验证实时因果部署。辅助RAW头仅训练使用，整图参考在推理保留。ONNX输入归一化RAW，推理无GT。

## 导出与文件

- `/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/onnx_best`：step126000，CPU ONNX一致性通过，最大绝对误差5.7220459e-06；权重SHA256 `1c8c07024af6855c7a988ed00d6ab52fa5e9beec9390b00ac8f1cf4e0ab7e541`。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/onnx_last`：step200000，CPU ONNX一致性通过，最大绝对误差5.24520874e-06；权重SHA256 `9543ca5401a6b8b04ff89acfd4f34bc92ea50bc29e3a6ceb35777628ea6f1cbf`。
- 指标及图片：`/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/quality/`；连续视频及逐帧诊断：`/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/videos/`。
- 计时：`/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/timing/`；训练曲线：`/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/training_curves.png`。
- 模型清单与校验：`/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/models.json`；全过程状态：`/data/zhangbenzhuang/huawei_sr/runs/SS928-S03-FULL-20260923-ANALYSIS/status.json`。

最佳和最后权重均保留；不自动替换其他场景模型。
