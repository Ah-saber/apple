# 提速结构跨场景验证结果

三组均为16通道、4倍入口打包、2主干块、1×1末端卷积，保留完整RAW整图参考和3倍输出。每组从随机权重训练12000步，前8000步原裁块，后4000步交替原裁块和均匀原尺寸训练。对照为同训练顺序的S03混合训练版，同时保留S03原裁块版作参考。

## 计时

GPU统一FP32、关闭TF32、batch1，1024×1280输入、3072×3840输出，包含整图参考；50次预热、600次测量。历史比例换算=GPU均值×41/0.914313，目标≤16.7ms。归一化、传输、8bit转换不计入模型耗时；SS928由部署端实测。

|组|GPU均值 ms|GPU P95 ms|理论换算 ms|同轮S03混合对照 ms|按同轮比例换算 ms|
|---|---:|---:|---:|---:|---:|
|day|0.3394|0.3417|15.22|0.9515|14.62|
|heavy|0.3452|0.3486|15.48|0.9454|14.97|
|night|0.3372|0.3399|15.12|0.9440|14.65|

## 分类别测试结果（各自验证最优权重）

单位dB。S03混合版与快速版采用相同训练顺序及12000步总预算，权重均从8000/10000/12000按验证原尺寸PSNR选取。夜间两类共同选择一份权重，不能分别挑测试最好的步数。

|场景|S03原版整图|S03混合整图|快速版整图|相对混合变化|S03混合裁块|快速版裁块|相对混合变化|
|---|---:|---:|---:|---:|---:|---:|---:|
|白天|21.923|23.010|21.581|-1.429|21.235|19.448|-1.788|
|重度天气|25.979|26.630|26.544|-0.087|27.182|26.452|-0.730|
|重度天气 C32|21.467|21.877|22.312|+0.434|22.400|22.588|+0.187|
|普通夜间|23.522|25.434|25.632|+0.198|26.738|25.678|-1.059|
|特殊夜间|22.922|21.779|22.327|+0.547|23.206|22.703|-0.503|

## 相同步数对照：双方均12000步

|场景|S03混合整图|快速版整图|变化|S03混合裁块|快速版裁块|变化|
|---|---:|---:|---:|---:|---:|---:|
|白天|22.486|21.288|-1.197|20.911|19.406|-1.505|
|重度天气|26.630|26.544|-0.087|27.182|26.452|-0.730|
|重度天气 C32|21.877|22.312|+0.434|22.400|22.588|+0.187|
|普通夜间|25.434|25.730|+0.296|26.738|26.498|-0.240|
|特殊夜间|21.779|22.328|+0.549|23.206|23.119|-0.086|

## 补充：此前原裁块模型的两万步预算结果

下表原模型在两万步训练范围内按验证选择；预算高于本轮，只作已有结果参考。

|场景|原模型选定步数|原模型整图|快速版整图|原模型裁块|快速版裁块|
|---|---:|---:|---:|---:|---:|
|白天|14000|22.152|21.581|21.274|19.448|
|重度天气|16000|25.919|26.544|26.839|26.452|
|重度天气 C32|16000|20.994|22.312|21.853|22.588|
|普通夜间|12000|23.522|25.632|26.777|25.678|
|特殊夜间|12000|22.922|22.327|24.547|22.703|

## 验证、结构相似度与亮度偏差

亮度偏差单位为8bit灰度，正数表示预测更亮。

|划分|场景|混合整图PSNR/SSIM|快速整图PSNR/SSIM|混合亮度偏差|快速亮度偏差|混合裁块SSIM|快速裁块SSIM|
|---|---|---:|---:|---:|---:|---:|---:|
|val|白天|21.965/0.71837|20.705/0.63388|-0.81|-2.65|0.59888|0.44535|
|test|白天|23.010/0.72970|21.581/0.64291|-1.02|-2.70|0.60105|0.45149|
|val|重度天气|27.326/0.95293|27.258/0.95282|+2.93|+2.48|0.93314|0.84844|
|val|重度天气 C32|18.854/0.89109|18.851/0.89889|-25.79|-26.05|0.87596|0.80648|
|test|重度天气|26.630/0.94529|26.544/0.94629|+4.10|+3.80|0.92933|0.84934|
|test|重度天气 C32|21.877/0.92179|22.312/0.92947|+16.50|+15.10|0.90858|0.85572|
|val|普通夜间|24.657/0.92016|24.900/0.90463|+5.53|+5.65|0.91099|0.82311|
|val|特殊夜间|20.052/0.83209|20.163/0.80244|-14.21|-14.99|0.81890|0.73683|
|test|普通夜间|25.434/0.92567|25.632/0.90729|+2.63|+2.62|0.91374|0.82312|
|test|特殊夜间|21.779/0.81364|22.327/0.79658|+1.41|+0.98|0.81171|0.75129|

## 连续视频的帧间误差变化

统计预测减GT后的残差，在相邻帧之间的平均绝对变化，单位为8bit灰度；跳过归一化分段边界。该量含运动和预测误差变化，不能单独证明闪烁程度。

|视频|S03原裁块|S03混合|快速版|
|---|---:|---:|---:|
|val_day_normal_0_consecutive|6.0270|5.8085|6.4133|
|test_day_normal_1_consecutive|8.0293|7.6223|8.6174|
|val_weather_heavy_0_consecutive|3.7548|3.6740|3.5725|
|val_weather_heavy_c32_1_consecutive|4.0686|3.9437|3.8068|
|val_weather_heavy_c32_2_consecutive|4.3685|4.2349|4.0746|
|test_weather_heavy_3_consecutive|4.1524|4.0535|3.9411|
|test_weather_heavy_c32_4_consecutive|4.2490|4.0707|3.9717|
|test_weather_heavy_c32_5_consecutive|4.0211|3.8410|3.7460|
|val_night_ordinary_0_consecutive|4.0746|4.2319|4.2560|
|val_night_special_1_consecutive|6.2372|7.1073|7.0690|
|test_night_ordinary_2_consecutive|3.8550|4.0928|4.0635|
|test_night_special_3_consecutive|8.2283|8.9107|9.0481|

## 权重、结果与连续视频

### day

选定10000步：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-DAY-B2-12K/checkpoints/step_000010000.pt`。
对照目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-DAY-B2-12K-COMPARISON`。
图片/视频从左至右：RAW输入、GT、S03原裁块版、S03混合版、快速B2。静态native图的RAW列是裁块输入放大预览；视频RAW列是真正原尺寸归一化RAW区域。native预测均来自整帧或允许ROI推理。

- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-DAY-B2-12K-COMPARISON/val_day_normal_0_consecutive.mp4`：120帧，capture_group_development，区域[0, 0, 1024, 1280]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-DAY-B2-12K-COMPARISON/test_day_normal_1_consecutive.mp4`：120帧，capture_group_development，区域[0, 0, 1024, 1280]。
### heavy

选定12000步：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K/checkpoints/step_000012000.pt`。
对照目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K-COMPARISON`。
图片/视频从左至右：RAW输入、GT、S03原裁块版、S03混合版、快速B2。静态native图的RAW列是裁块输入放大预览；视频RAW列是真正原尺寸归一化RAW区域。native预测均来自整帧或允许ROI推理。

- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K-COMPARISON/val_weather_heavy_0_consecutive.mp4`：120帧，capture_group_development，区域[0, 0, 1024, 1280]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K-COMPARISON/val_weather_heavy_c32_1_consecutive.mp4`：120帧，spatial_development，区域[2, 0, 1020, 192]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K-COMPARISON/val_weather_heavy_c32_2_consecutive.mp4`：120帧，spatial_development，区域[2, 0, 1020, 192]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K-COMPARISON/test_weather_heavy_3_consecutive.mp4`：120帧，capture_group_development，区域[0, 0, 1024, 1280]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K-COMPARISON/test_weather_heavy_c32_4_consecutive.mp4`：120帧，spatial_development，区域[2, 1088, 1020, 192]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-HEAVY-B2-12K-COMPARISON/test_weather_heavy_c32_5_consecutive.mp4`：120帧，spatial_development，区域[2, 1088, 1020, 192]。
### night

选定10000步：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-NIGHT-B2-12K/checkpoints/step_000010000.pt`。
对照目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-NIGHT-B2-12K-COMPARISON`。
图片/视频从左至右：RAW输入、GT、S03原裁块版、S03混合版、快速B2。静态native图的RAW列是裁块输入放大预览；视频RAW列是真正原尺寸归一化RAW区域。native预测均来自整帧或允许ROI推理。

- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-NIGHT-B2-12K-COMPARISON/val_night_ordinary_0_consecutive.mp4`：60帧，capture_group_development，区域[0, 0, 1024, 1280]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-NIGHT-B2-12K-COMPARISON/val_night_special_1_consecutive.mp4`：120帧，spatial_development，区域[2, 0, 1020, 192]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-NIGHT-B2-12K-COMPARISON/test_night_ordinary_2_consecutive.mp4`：60帧，capture_group_development，区域[0, 0, 1024, 1280]。
- `/data/zhangbenzhuang/huawei_sr/runs/SS928-SPEED-CROSS-20260924-NIGHT-B2-12K-COMPARISON/test_night_special_3_consecutive.mp4`：120帧，spatial_development，区域[2, 1088, 1020, 192]。

## 解释范围与校验

- 整图指标：3倍输出截断到[0,1]后3倍平均回缩，与原尺寸GT比较；无真实3倍GT，需结合裁块和3倍局部图看细节。
- 数据冻结：白天训练3689帧；重度240+240帧；夜间180+120帧；每类验证/测试各12帧。C32及特殊夜间是同序列的空间留出，不能当独立新序列泛化。归一化和参考严格限定允许ROI；当前归一化为离线统计。
- 本轮12000步为固定预算验证，没有证明完全收敛。新结构从随机初始化，同一随机种子不代表相同初始函数。仅各组单次训练，没有多种子置信区间。
- 连续视频全帧生成并解码检查；预测减GT的帧间变化含运动、噪声等影响，无运动对齐，不能单独判定闪烁；未完整人工观看所有视频。
- 全部媒体保留服务器。未覆盖旧模型、GT和部署包；未执行板端转换/量化/实测。用户尚未给出允许下降的数值阈值，不自行宣布画质可接受。
- 所有选定权重SHA、归一化及中间GT索引SHA一致；独立审计与对比程序的选定权重指标一致；预检查覆盖各训练序列/ROI及8000步切换边界。

## 本轮判断

三组理论换算速度均达到16.7ms目标，暂不建议全场景统一替换。白天退化最明显；重度天气整图指标较接近原结构，裁块SSIM仍下降；夜间对混合训练对照的整图PSNR改善，但结构相似度下降，特殊夜间仍不及保留的原裁块训练版。

夜间12000步时，普通/特殊裁块PSNR差距缩到0.24/0.09dB，裁块SSIM仍下降0.073/0.048。验证两类整图平均PSNR在10000步为22.531dB，12000步为22.375dB，因此按预定规则保留10000步作为正式选择；12000步结果独立列出，不依据测试结果更改选择。
