# SS928 本轮审查：全模型提速与小目标拖影

七类主模型、四条天气四分之一网格路径、日间当前帧备选均已完成服务器实验。保留原权重和浮点输出接口的布局改动，在七类主模型上测得完整GPU前向均值下降12.4%～26.2%；日间备选下降13.6%。SS928新图尚未板测，16.7毫秒目标尚未确认达成。

拖影已有实质代码修改和部分收益，仍存在目标弱化及质量、耗时取舍。以下候选保持独立，原版本和权重保留。本轮没有制作交付包、提交或推送，先由用户审查。

## 保留原权重的提速

把原生相位的重排与三倍复制改成Reshape、Transpose、Expand，减少中间展开和逐行拼接；分别保留Float16输出和先截断到UInt8再展开的接口。输出的三倍展开沿用既有最近邻复制，学习预测仍在原生分辨率。

下表保持原浮点输出类型，RTX5090，完整输入到3072×3840输出，预热30次、计时300次。RAW读取、输入准备、主机传输不在计时内。

| 模型 | 原均值 ms | 新均值 ms | 新P95 ms | 均值下降 |
|---|---:|---:|---:|---:|
| 日间九帧 | 0.3854 | 0.3275 | 0.3333 | 15.0% |
| 轻天气半网格 | 0.3382 | 0.2962 | 0.3045 | 12.4% |
| 中天气半网格 | 0.3403 | 0.2955 | 0.3029 | 13.2% |
| 重天气半网格 | 0.3230 | 0.2763 | 0.2875 | 14.4% |
| 重天气C32半网格 | 0.3279 | 0.2784 | 0.2875 | 15.1% |
| 普通夜间 | 0.4093 | 0.3022 | 0.3052 | 26.2% |
| 特殊夜间 | 0.3640 | 0.3179 | 0.3216 | 12.7% |
| 日间当前帧备选 | 0.3682 | 0.3183 | 0.3217 | 13.6% |

四条天气四分之一网格的浮点布局均值下降19.6%～21.2%；主卷积乘加约0.811G，相对于半网格约2.463G减少67%。低网格的质量取舍另行保留，未统一替换半网格。相同字节输出条件下，七类主模型均值下降约12%～27%。[完整均值/P95及逐次计时](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/RESULTS.md)。

普通/特殊夜间线性尾部合并及边界补偿，使完整卷积乘加减少约8.6%/8.2%；真实60/120帧相对于原字节输出最大差1灰度，平均差约0.036/0.045灰度。GPU较慢，保留为板测备选。半精度参考采样、卷积实现双线性采样等变体也已比较，未直接替换默认路径。

## 拖影改动与取舍

天气尝试一帧、三帧、九帧主体约束训练，以及局部历史混合。局部历史混合按当前RAW与过去RAW之差控制当前位置特征的比例；一帧及强切换能减轻拖影，但部分静态变化、弱结构误差增加约四成。三帧半网格在现有天气序列上更平衡：整图PSNR基本保持，静态变化和弱结构误差降低，小目标位置及残留改善。参考和既有预处理的历史范围保留，外部输入仍为九帧。

日间新增冻结原主体的残差分支，即只学习相对原输出的补偿量。夜间保留主体权重，按RAW差异进行局部特征混合，并把混合强度从1降至0.25。推理分支使用当前与历史RAW；GT用于训练和离线分析。

下表只统计GT辅助提取的4～64像素亮暗目标。历史旧位置超额灰度和局部重心误差越低越好，目标对比比例接近1较好。日间采用已修正的实际直线行车道。

| 场景及待审候选 | 旧位置灰度：原→新 | 重心偏差px：原→新 | 对比比例：原→新 | 道路静态变化 |
|---|---:|---:|---:|---:|
| 轻天气三帧 | 2.735→2.169 | 3.261→2.753 | 0.805→0.871 | -6.6% |
| 中天气三帧 | 4.101→3.556 | 2.224→1.895 | 0.910→0.984 | -6.6% |
| 重天气三帧 | 3.349→2.940 | 3.238→3.022 | 0.840→0.903 | -5.6% |
| 日间残差 | 4.254→3.878 | 1.896→1.662 | 0.659→0.779 | +2.5% |
| 普通夜间弱混合 | 1.495→1.452 | 4.535→2.708 | 0.242→0.315 | +2.3% |
| 特殊夜间弱混合 | 1.805→1.760 | 6.738→4.092 | 0.300→0.415 | +13.0% |

重天气C32的三帧候选完整PSNR为23.8342→23.8296分贝，静态及弱结构误差降低；其有效标注区域不覆盖本轮所选天气道路，未给道路小目标改善结论。特殊夜间只在右侧192像素有效标注内统计。

日间残差整图PSNR为25.4843→25.7924分贝，弱结构误差约增加3.6%。它的浮点完整GPU均值约0.5147毫秒，原图约0.3854毫秒，增加约34%；特殊夜间弱混合也增加GPU开销。拖影候选没有在所有模型上同时满足速度与质量要求。原权重布局提速可单独使用；日间残差和特殊夜间混合保留为用户审查、板测候选。

## 先查看的连续图片

以下均为GT、原九帧、候选的同场景同帧八帧序列；视频为12帧/秒展示速度，不能用来判断实际推理速度。

- [轻天气三帧：第080～087帧](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/targets_tiny_half3_light/weather_light/target_083_eight_frames.png)
- [日间残差：实际行车道第035～042帧](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/targets_tiny_lane_day_residual/day_normal/target_038_eight_frames.png)
- [特殊夜间弱混合：第052～059帧](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/targets_tiny_night_weak/night_special/target_055_eight_frames.png)
- [天气完整连续对比视频](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/half3_eval_light/weather_light/full120_history_intervention.mp4)
- [日间完整连续对比视频](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/residual_nonzero_eval_day/day_normal/full120_history_intervention.mp4)
- [普通夜间60帧对比视频](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/night_motion_weak/night_ordinary/full60_history_intervention.mp4)
- [特殊夜间120帧对比视频](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/night_motion_weak/night_special/full120_history_intervention.mp4)

已人工检查上述代表八帧图片及部分整图，未逐帧人工观看全部28段视频。GT阈值连通区域不是逐个目标的人工标注，仍可能受噪声、边缘和处理后GT变化影响；小幅数值差异不视为显著结论。当前结论限于这些开发序列，反复用测试片段检查候选，尚无独立盲测。用户截图表达小目标拖影，未绑定这些实验的具体场景与帧号。

## 核验和作废记录

七类主路径合计780帧，加四条天气低网格480帧及日间备选120帧，共1380个完整输入。两种字节布局及保持Float16的布局，在相同直接运行后端逐帧完整输出一致。普通夜间只有60帧，特殊夜间120帧。

布局一致性不能推广为编译器或NPU逐位一致。GPU编译后首帧浮点数值差已另记，最大约0.25～0.375灰度。七类拖影候选各12份输入核对：浮点结果截断后与相同权重的字节接口输出一致，共84份；已补齐对应浮点图。

导出图经过ONNX检查，141张图有记录；28段视频完整解码数量正确；本地审查清单520个文件全部SHA-256匹配。原v13本地既有10个模型文件与服务器副本一致。两个审查工具脚本副本有差异，已保留各自校验值，未据此声称所有工具副本均相同。

本轮早期天气低网格未恢复权重保存的参考注入选项，相关结果作废；已先恢复选项，再与旧v13独立FP32图核对，最大差约0.00023灰度后重跑。日间零阈值校准也未采用，最终用训练RAW的全区域信号分位数；该量包含运动，不解释为噪声方差。日间旧区域混入建筑和停放车辆，其目标统计作废，已收紧到同场景实际行车道并重算。

- [作废天气配置及有效范围](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/INVALID_EARLY_WEATHER_CONFIG.json)
- [作废日间目标区域及替换记录](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/INVALID_EARLY_DAY_TARGET_ROI.json)
- [最终文件、图及视频核验](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/review_file_verification.json)
- [本地逐文件核验](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/local_file_verification.json)
- [原模型与工具副本对照](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/source_v13_local_comparison.json)

## 复现与下一步

服务器独立目录：`/data/zhangbenzhuang/huawei_sr/runs/SS928-ALL-MODELS-ROUND2-20260930`。使用`/data/zhangbenzhuang/miniconda3/envs/test/bin/python`、GPU1，Torch 2.7；TF32关闭，保留原混合精度规则。实验源码保存在本目录runtime，训练记录包含原权重/缓存校验值、种子、损失、验证选择和步骤；大型浮点缓存留在服务器。本地test环境已通过全部源码语法检查，模型计算在服务器完成。

七类拖影候选的浮点图在[候选图及输入对照](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/candidate_float_exports/candidate_float_exports.json)；原权重浮点布局图在`results/exact_layout_five_r2/`及`results/exact_layout_night_r2/`，日间备选在`results/day_backup_float/`。完整实验数值、源路径和逐次耗时见[RESULTS.md](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/RESULTS.md)。

下一步需要用户审查连续目标及建筑结构，并取得SS928的编译、逐算子耗时、量化连续画质及完整纯NPU均值/P95。[板测步骤与计时边界](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/BOARD-VALIDATION.md)已保存。当前不宣称全局最优，也不宣称拖影已消除或NPU达到16.7毫秒；后续按实际板测瓶颈决定改动。
