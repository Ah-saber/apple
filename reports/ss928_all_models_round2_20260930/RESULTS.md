# 完整实验数值

表内 GPU 时间为预先就绪输入到完整3072×3840输出，RTX5090，预热30次、计时300次、均值与P95；不包含RAW读取、输入准备、主机传输。NPU未实测。

## 保留原浮点输出接口的完整前向

| 路径 | 原均值/P95 ms | 新均值/P95 ms | 均值下降 | 卷积乘加 G |
|---|---:|---:|---:|---:|
| day_normal_half | 0.385375/0.392451 | 0.327504/0.333312 | 15.0% | 6.520 |
| weather_light_half | 0.338198/0.349392 | 0.296228/0.304533 | 12.4% | 2.463 |
| weather_light_quarter | 0.349568/0.360323 | 0.276649/0.283875 | 20.9% | 0.811 |
| weather_medium_half | 0.340323/0.350290 | 0.295504/0.302947 | 13.2% | 2.463 |
| weather_medium_quarter | 0.347410/0.356405 | 0.279332/0.289382 | 19.6% | 0.811 |
| weather_heavy_half | 0.322975/0.334050 | 0.276310/0.287490 | 14.4% | 2.462 |
| weather_heavy_quarter | 0.329600/0.339237 | 0.259814/0.269510 | 21.2% | 0.811 |
| weather_heavy_c32_half | 0.327932/0.334502 | 0.278370/0.287499 | 15.1% | 2.462 |
| weather_heavy_c32_quarter | 0.332697/0.341862 | 0.264542/0.273571 | 20.5% | 0.811 |
| night_ordinary | 0.409309/0.416088 | 0.302214/0.305226 | 26.2% | 4.860 |
| night_special | 0.364025/0.369346 | 0.317892/0.321570 | 12.7% | 5.133 |
| day_current_backup | 0.368168/0.375205 | 0.318273/0.321696 | 13.6% | — |

[float_speed_profile/float_speed_profile.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/float_speed_profile/float_speed_profile.json)；[day_backup_float/day_backup.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/day_backup_float/day_backup.json)。卷积量包含固定卷积，填充位置按常规乘加口径计数，不含布局和采样的内存开销，不代表NPU时间或内存峰值。

编译后相对于直接运行的首帧浮点误差已单独记录，布局同精度逐帧一致性是在直接运行后端核对；不能将其推广为编译器或NPU逐位一致。

## 相同完整字节输出的前向

| 路径 | 原均值/P95 ms | 新均值/P95 ms | 均值下降 |
|---|---:|---:|---:|
| day_normal_half | 0.388907/0.386371 | 0.330944/0.328834 | 14.9% |
| weather_light_half | 0.329460/0.336672 | 0.286081/0.291107 | 13.2% |
| weather_medium_half | 0.329046/0.336869 | 0.288388/0.293154 | 12.4% |
| weather_heavy_half | 0.312632/0.319522 | 0.273234/0.280642 | 12.6% |
| weather_heavy_c32_half | 0.311545/0.321406 | 0.271018/0.277301 | 13.0% |
| weather_light_quarter | 0.332639/0.347440 | 0.270991/0.279971 | 18.5% |
| weather_medium_quarter | 0.339573/0.349670 | 0.265869/0.274307 | 21.7% |
| weather_heavy_quarter | 0.321013/0.330501 | 0.247765/0.255464 | 22.8% |
| weather_heavy_c32_quarter | 0.321135/0.330851 | 0.249689/0.259650 | 22.2% |
| night_ordinary | 0.400578/0.398720 | 0.291283/0.297989 | 27.3% |
| night_special | 0.354010/0.359264 | 0.298328/0.306147 | 15.7% |
| day_current_backup | 0.360564/0.369920 | 0.311990/0.318325 | 13.5% |

字节输出在小网格截断后再展开；与原图末端截断的字节输出核对。Float16接口与UInt8接口分别比较，表间时间不直接合并。

## 完整连续画质

质量数值按各场景真实GT有效区域、去掉3像素边界计算。PSNR单位为分贝；静态变化和弱结构误差以灰度计，越低越好；平均运动响应不能代替小目标定位。

| 实验 | 场景 | 方法 | PSNR | 静态变化 | 弱结构误差 | 运动响应 |
|---|---|---|---:|---:|---:|---:|
| adaptive_correct_light | weather_light | 原半网格九帧 | 34.3061 | 0.8917 | 1.2824 | 0.4685 |
| adaptive_correct_light | weather_light | 原四分之一九帧 | 33.6013 | 1.0546 | 1.4108 | 0.5176 |
| adaptive_correct_light | weather_light | 局部当前半网格 | 34.3185 | 0.9370 | 1.5559 | 0.6296 |
| adaptive_correct_light | weather_light | 局部当前四分之一 | 33.5908 | 1.0895 | 1.5529 | 0.5839 |
| adaptive_correct_light | weather_medium | 原半网格九帧 | 33.7781 | 0.8836 | 1.2832 | 0.4751 |
| adaptive_correct_light | weather_medium | 原四分之一九帧 | 32.8541 | 1.0410 | 1.4067 | 0.5234 |
| adaptive_correct_light | weather_medium | 局部当前半网格 | 33.7917 | 0.9283 | 1.5563 | 0.6380 |
| adaptive_correct_light | weather_medium | 局部当前四分之一 | 32.8450 | 1.0762 | 1.5506 | 0.5908 |
| adaptive_correct_heavy | weather_heavy | 原半网格九帧 | 30.2750 | 1.5056 | 1.7535 | 0.5324 |
| adaptive_correct_heavy | weather_heavy | 原四分之一九帧 | 30.1082 | 1.7562 | 2.0263 | 0.6514 |
| adaptive_correct_heavy | weather_heavy | 局部当前半网格 | 30.2600 | 1.7458 | 2.4217 | 0.7975 |
| adaptive_correct_heavy | weather_heavy | 局部当前四分之一 | 30.0519 | 2.0334 | 2.6033 | 0.8239 |
| adaptive_correct_heavy | weather_heavy_c32 | 原半网格九帧 | 23.8342 | 1.6241 | 1.9835 | 0.7028 |
| adaptive_correct_heavy | weather_heavy_c32 | 原四分之一九帧 | 23.9027 | 1.8107 | 2.1584 | 0.8153 |
| adaptive_correct_heavy | weather_heavy_c32 | 局部当前半网格 | 23.8222 | 1.8369 | 2.6415 | 0.9978 |
| adaptive_correct_heavy | weather_heavy_c32 | 局部当前四分之一 | 23.8895 | 2.0619 | 2.7139 | 1.0212 |
| adaptive_nonzero_day | day_normal | 原半网格九帧 | 25.4843 | 1.7430 | 3.1135 | 0.3335 |
| adaptive_nonzero_day | day_normal | 局部当前半网格 | 25.5912 | 1.7669 | 3.1725 | 0.3607 |
| residual_eval_light | weather_light | 原半网格九帧 | 34.3061 | 0.8917 | 1.2824 | 0.4685 |
| residual_eval_light | weather_light | 原四分之一九帧 | 33.6013 | 1.0546 | 1.4108 | 0.5176 |
| residual_eval_light | weather_light | 目标残差半网格 | 34.3192 | 0.8792 | 1.2642 | 0.4816 |
| residual_eval_light | weather_light | 目标残差四分之一 | 33.7480 | 1.0000 | 1.3268 | 0.5015 |
| residual_eval_light | weather_medium | 原半网格九帧 | 33.7781 | 0.8836 | 1.2832 | 0.4751 |
| residual_eval_light | weather_medium | 原四分之一九帧 | 32.8541 | 1.0410 | 1.4067 | 0.5234 |
| residual_eval_light | weather_medium | 目标残差半网格 | 33.7925 | 0.8713 | 1.2642 | 0.4881 |
| residual_eval_light | weather_medium | 目标残差四分之一 | 32.9746 | 0.9873 | 1.3219 | 0.5049 |
| residual_eval_heavy | weather_heavy | 原半网格九帧 | 30.2750 | 1.5056 | 1.7535 | 0.5324 |
| residual_eval_heavy | weather_heavy | 原四分之一九帧 | 30.1082 | 1.7562 | 2.0263 | 0.6514 |
| residual_eval_heavy | weather_heavy | 目标残差半网格 | 30.2431 | 1.4999 | 1.7693 | 0.5588 |
| residual_eval_heavy | weather_heavy | 目标残差四分之一 | 30.2238 | 1.6518 | 1.8736 | 0.6200 |
| residual_eval_heavy | weather_heavy_c32 | 原半网格九帧 | 23.8342 | 1.6241 | 1.9835 | 0.7028 |
| residual_eval_heavy | weather_heavy_c32 | 原四分之一九帧 | 23.9027 | 1.8107 | 2.1584 | 0.8153 |
| residual_eval_heavy | weather_heavy_c32 | 目标残差半网格 | 23.7701 | 1.6195 | 2.0030 | 0.7262 |
| residual_eval_heavy | weather_heavy_c32 | 目标残差四分之一 | 23.9043 | 1.7145 | 2.0170 | 0.7684 |
| residual_nonzero_eval_day | day_normal | 原半网格九帧 | 25.4843 | 1.7430 | 3.1135 | 0.3335 |
| residual_nonzero_eval_day | day_normal | 目标残差半网格 | 25.7924 | 1.7787 | 3.2249 | 0.4356 |
| correct_fp32_eval_light | weather_light | 原半网格九帧 | 34.3061 | 0.8917 | 1.2824 | 0.4685 |
| correct_fp32_eval_light | weather_light | 原四分之一九帧 | 33.6013 | 1.0546 | 1.4108 | 0.5176 |
| correct_fp32_eval_light | weather_light | 最近三帧训练 | 33.6928 | 0.9174 | 1.2670 | 0.4398 |
| correct_fp32_eval_light | weather_light | 九帧目标约束训练 | 33.6775 | 0.9228 | 1.2679 | 0.4484 |
| correct_fp32_eval_light | weather_medium | 原半网格九帧 | 33.7781 | 0.8836 | 1.2832 | 0.4751 |
| correct_fp32_eval_light | weather_medium | 原四分之一九帧 | 32.8541 | 1.0410 | 1.4067 | 0.5234 |
| correct_fp32_eval_light | weather_medium | 最近三帧训练 | 32.9391 | 0.9060 | 1.2635 | 0.4448 |
| correct_fp32_eval_light | weather_medium | 九帧目标约束训练 | 32.9092 | 0.9105 | 1.2640 | 0.4534 |
| correct_fp32_eval_heavy | weather_heavy | 原半网格九帧 | 30.2750 | 1.5056 | 1.7535 | 0.5324 |
| correct_fp32_eval_heavy | weather_heavy | 原四分之一九帧 | 30.1082 | 1.7562 | 2.0263 | 0.6514 |
| correct_fp32_eval_heavy | weather_heavy | 最近三帧训练 | 30.0395 | 1.6553 | 1.8895 | 0.5955 |
| correct_fp32_eval_heavy | weather_heavy | 九帧目标约束训练 | 30.1070 | 1.6140 | 1.8342 | 0.5856 |
| correct_fp32_eval_heavy | weather_heavy_c32 | 原半网格九帧 | 23.8342 | 1.6241 | 1.9835 | 0.7028 |
| correct_fp32_eval_heavy | weather_heavy_c32 | 原四分之一九帧 | 23.9027 | 1.8107 | 2.1584 | 0.8153 |
| correct_fp32_eval_heavy | weather_heavy_c32 | 最近三帧训练 | 23.7816 | 1.7136 | 2.0116 | 0.7418 |
| correct_fp32_eval_heavy | weather_heavy_c32 | 九帧目标约束训练 | 23.8654 | 1.6747 | 1.9657 | 0.7334 |
| half3_eval_light | weather_light | 原半网格九帧 | 34.3061 | 0.8917 | 1.2824 | 0.4685 |
| half3_eval_light | weather_light | 最近三帧训练 | 34.3157 | 0.8387 | 1.2348 | 0.4308 |
| half3_eval_light | weather_medium | 原半网格九帧 | 33.7781 | 0.8836 | 1.2832 | 0.4751 |
| half3_eval_light | weather_medium | 最近三帧训练 | 33.7882 | 0.8305 | 1.2354 | 0.4359 |
| half3_eval_heavy | weather_heavy | 原半网格九帧 | 30.2750 | 1.5056 | 1.7535 | 0.5324 |
| half3_eval_heavy | weather_heavy | 最近三帧训练 | 30.2808 | 1.4284 | 1.6704 | 0.4822 |
| half3_eval_heavy | weather_heavy_c32 | 原半网格九帧 | 23.8342 | 1.6241 | 1.9835 | 0.7028 |
| half3_eval_heavy | weather_heavy_c32 | 最近三帧训练 | 23.8296 | 1.5460 | 1.8793 | 0.6305 |
| half3_eval_day | day_normal | 原半网格九帧 | 25.4843 | 1.7430 | 3.1135 | 0.3335 |
| half3_eval_day | day_normal | 最近三帧训练 | 25.4128 | 1.7340 | 3.0830 | 0.3361 |
| half1_eval_light | weather_light | 原半网格九帧 | 34.3061 | 0.8917 | 1.2824 | 0.4685 |
| half1_eval_light | weather_light | 当前帧训练 | 34.2215 | 1.2577 | 1.8284 | 0.7227 |
| half1_eval_light | weather_medium | 原半网格九帧 | 33.7781 | 0.8836 | 1.2832 | 0.4751 |
| half1_eval_light | weather_medium | 当前帧训练 | 33.7066 | 1.2483 | 1.8294 | 0.7315 |
| half1_eval_heavy | weather_heavy | 原半网格九帧 | 30.2750 | 1.5056 | 1.7535 | 0.5324 |
| half1_eval_heavy | weather_heavy | 当前帧训练 | 30.4200 | 2.1466 | 2.7215 | 0.8604 |
| half1_eval_heavy | weather_heavy_c32 | 原半网格九帧 | 23.8342 | 1.6241 | 1.9835 | 0.7028 |
| half1_eval_heavy | weather_heavy_c32 | 当前帧训练 | 24.1081 | 2.2106 | 2.9346 | 1.0834 |
| half1_eval_day | day_normal | 原半网格九帧 | 25.4843 | 1.7430 | 3.1135 | 0.3335 |
| half1_eval_day | day_normal | 当前帧训练 | 25.6261 | 1.8573 | 3.2573 | 0.3771 |
| night_motion_r2 | night_ordinary | 原夜间九帧 | 27.8682 | 0.9998 | 1.3191 | 0.2528 |
| night_motion_r2 | night_ordinary | 局部当前夜间 | 27.8732 | 1.1590 | 1.8227 | 0.7257 |
| night_motion_r2 | night_special | 原夜间九帧 | 25.9193 | 1.1065 | 1.2592 | 0.3793 |
| night_motion_r2 | night_special | 局部当前夜间 | 25.7959 | 1.7973 | 2.7133 | 1.1837 |
| night_motion_weak | night_ordinary | 原夜间九帧 | 27.8682 | 0.9998 | 1.3191 | 0.2528 |
| night_motion_weak | night_ordinary | 局部当前夜间 | 27.8740 | 1.0241 | 1.3453 | 0.3593 |
| night_motion_weak | night_special | 原夜间九帧 | 25.9193 | 1.1065 | 1.2592 | 0.3793 |
| night_motion_weak | night_special | 局部当前夜间 | 25.9101 | 1.2347 | 1.5044 | 0.5668 |

正确FP32九帧实验初始化保留原权重，没有预设偏重当前系数；早期视频沿用“偏重当前”标签，正确含义为九帧目标约束训练。原标签保留，避免修改既有证据。

## 4～600像素道路目标

GT辅助的亮暗连通区域统计；背景为各方法完整序列中位数，扣除每帧道路整体漂移。历史残留和位置偏差越低越好，对比比例接近1较好。静态道路变化包含噪声与真实变化，不能单独称为噪声。

| 实验 | 场景 | 方法 | 旧位置超额灰度 | 重心误差 px | 目标对比比例 | 道路静态变化 |
|---|---|---|---:|---:|---:|---:|
| targets_adaptive_light | weather_light | 原半网格九帧 | 6.5362 | 5.0263 | 0.4677 | 0.9317 |
| targets_adaptive_light | weather_light | 原四分之一九帧 | 4.9456 | 3.6077 | 0.5635 | 1.0708 |
| targets_adaptive_light | weather_light | 局部当前半网格 | 5.2882 | 3.5315 | 0.5987 | 1.0106 |
| targets_adaptive_light | weather_light | 局部当前四分之一 | 4.9801 | 3.5309 | 0.5616 | 1.1254 |
| targets_adaptive_light | weather_medium | 原半网格九帧 | 6.9019 | 3.9264 | 0.6351 | 0.9220 |
| targets_adaptive_light | weather_medium | 原四分之一九帧 | 5.6485 | 2.8820 | 0.6964 | 1.0602 |
| targets_adaptive_light | weather_medium | 局部当前半网格 | 6.1612 | 2.8593 | 0.7201 | 1.0029 |
| targets_adaptive_light | weather_medium | 局部当前四分之一 | 5.7046 | 2.8470 | 0.6964 | 1.1170 |
| targets_adaptive_heavy | weather_heavy | 原半网格九帧 | 7.6708 | 5.5245 | 0.4220 | 1.5172 |
| targets_adaptive_heavy | weather_heavy | 原四分之一九帧 | 6.2622 | 3.7850 | 0.5725 | 1.7579 |
| targets_adaptive_heavy | weather_heavy | 局部当前半网格 | 6.3017 | 3.4923 | 0.5633 | 1.7863 |
| targets_adaptive_heavy | weather_heavy | 局部当前四分之一 | 6.3822 | 3.5282 | 0.5811 | 2.0647 |
| targets_residual_light | weather_light | 原半网格九帧 | 6.5362 | 5.0263 | 0.4677 | 0.9317 |
| targets_residual_light | weather_light | 原四分之一九帧 | 4.9456 | 3.6077 | 0.5635 | 1.0708 |
| targets_residual_light | weather_light | 目标残差半网格 | 6.5709 | 5.0495 | 0.4606 | 0.9203 |
| targets_residual_light | weather_light | 目标残差四分之一 | 5.5172 | 4.2104 | 0.5091 | 1.0208 |
| targets_residual_light | weather_medium | 原半网格九帧 | 6.9019 | 3.9264 | 0.6351 | 0.9220 |
| targets_residual_light | weather_medium | 原四分之一九帧 | 5.6485 | 2.8820 | 0.6964 | 1.0602 |
| targets_residual_light | weather_medium | 目标残差半网格 | 6.9588 | 3.9776 | 0.6321 | 0.9111 |
| targets_residual_light | weather_medium | 目标残差四分之一 | 6.0414 | 3.3495 | 0.6612 | 1.0122 |
| targets_residual_heavy | weather_heavy | 原半网格九帧 | 7.6708 | 5.5245 | 0.4220 | 1.5172 |
| targets_residual_heavy | weather_heavy | 原四分之一九帧 | 6.2622 | 3.7850 | 0.5725 | 1.7579 |
| targets_residual_heavy | weather_heavy | 目标残差半网格 | 7.4599 | 5.1730 | 0.4384 | 1.5156 |
| targets_residual_heavy | weather_heavy | 目标残差四分之一 | 6.5699 | 4.1080 | 0.5442 | 1.6593 |
| targets_fp32_light | weather_light | 原半网格九帧 | 6.5362 | 5.0263 | 0.4677 | 0.9317 |
| targets_fp32_light | weather_light | 原四分之一九帧 | 4.9456 | 3.6077 | 0.5635 | 1.0708 |
| targets_fp32_light | weather_light | 最近三帧训练 | 5.0742 | 3.7512 | 0.5373 | 0.9277 |
| targets_fp32_light | weather_light | 九帧目标约束训练 | 5.2056 | 3.9096 | 0.5325 | 0.9348 |
| targets_fp32_light | weather_medium | 原半网格九帧 | 6.9019 | 3.9264 | 0.6351 | 0.9220 |
| targets_fp32_light | weather_medium | 原四分之一九帧 | 5.6485 | 2.8820 | 0.6964 | 1.0602 |
| targets_fp32_light | weather_medium | 最近三帧训练 | 5.7744 | 3.0971 | 0.6800 | 0.9207 |
| targets_fp32_light | weather_medium | 九帧目标约束训练 | 5.8082 | 3.1527 | 0.6754 | 0.9255 |
| targets_fp32_heavy | weather_heavy | 原半网格九帧 | 7.6708 | 5.5245 | 0.4220 | 1.5172 |
| targets_fp32_heavy | weather_heavy | 原四分之一九帧 | 6.2622 | 3.7850 | 0.5725 | 1.7579 |
| targets_fp32_heavy | weather_heavy | 最近三帧训练 | 6.3794 | 3.8704 | 0.5635 | 1.6551 |
| targets_fp32_heavy | weather_heavy | 九帧目标约束训练 | 6.4468 | 4.0449 | 0.5511 | 1.6129 |
| targets_half3_light | weather_light | 原半网格九帧 | 6.5362 | 5.0263 | 0.4677 | 0.9317 |
| targets_half3_light | weather_light | 最近三帧训练 | 5.3537 | 3.9458 | 0.5458 | 0.8720 |
| targets_half3_light | weather_medium | 原半网格九帧 | 6.9019 | 3.9264 | 0.6351 | 0.9220 |
| targets_half3_light | weather_medium | 最近三帧训练 | 6.1141 | 3.2453 | 0.6928 | 0.8617 |
| targets_half3_heavy | weather_heavy | 原半网格九帧 | 7.6708 | 5.5245 | 0.4220 | 1.5172 |
| targets_half3_heavy | weather_heavy | 最近三帧训练 | 6.4061 | 4.3772 | 0.5182 | 1.4346 |
| targets_night_strong | night_ordinary | 原夜间九帧 | 1.4946 | 4.5349 | 0.2422 | 1.0015 |
| targets_night_strong | night_ordinary | 局部当前夜间 | 1.3717 | 1.6669 | 0.5975 | 1.1527 |
| targets_night_strong | night_special | 原夜间九帧 | 1.8051 | 6.7384 | 0.2996 | 1.1608 |
| targets_night_strong | night_special | 局部当前夜间 | 1.7814 | 3.0931 | 0.7957 | 1.9686 |
| targets_night_weak | night_ordinary | 原夜间九帧 | 1.4946 | 4.5349 | 0.2422 | 1.0015 |
| targets_night_weak | night_ordinary | 局部当前夜间 | 1.4516 | 2.7077 | 0.3147 | 1.0241 |
| targets_night_weak | night_special | 原夜间九帧 | 1.8051 | 6.7384 | 0.2996 | 1.1608 |
| targets_night_weak | night_special | 局部当前夜间 | 1.7598 | 4.0918 | 0.4153 | 1.3121 |
| targets_half1_light | weather_light | 原半网格九帧 | 6.5362 | 5.0263 | 0.4677 | 0.9317 |
| targets_half1_light | weather_light | 当前帧训练 | 5.1541 | 3.5046 | 0.5986 | 1.3403 |
| targets_half1_light | weather_medium | 原半网格九帧 | 6.9019 | 3.9264 | 0.6351 | 0.9220 |
| targets_half1_light | weather_medium | 当前帧训练 | 6.0490 | 2.8159 | 0.7191 | 1.3266 |
| targets_half1_heavy | weather_heavy | 原半网格九帧 | 7.6708 | 5.5245 | 0.4220 | 1.5172 |
| targets_half1_heavy | weather_heavy | 当前帧训练 | 6.2935 | 3.4723 | 0.5619 | 2.1839 |
| targets_lane_day_adaptive | day_normal | 原半网格九帧 | 3.6986 | 1.9164 | 0.6601 | 2.0710 |
| targets_lane_day_adaptive | day_normal | 局部当前半网格 | 3.6605 | 1.8151 | 0.6823 | 2.1003 |
| targets_lane_day_residual | day_normal | 原半网格九帧 | 3.6986 | 1.9164 | 0.6601 | 2.0710 |
| targets_lane_day_residual | day_normal | 目标残差半网格 | 3.3031 | 1.6784 | 0.7594 | 2.1175 |
| targets_lane_day_three | day_normal | 原半网格九帧 | 3.6986 | 1.9164 | 0.6601 | 2.0710 |
| targets_lane_day_three | day_normal | 最近三帧训练 | 3.6710 | 1.8642 | 0.6757 | 2.0518 |
| targets_lane_day_one | day_normal | 原半网格九帧 | 3.6986 | 1.9164 | 0.6601 | 2.0710 |
| targets_lane_day_one | day_normal | 当前帧训练 | 3.5966 | 1.7525 | 0.6878 | 2.1895 |

目标区域由GT阈值辅助提取，未有人工作目标逐个标注；小幅数值改变不视为显著结果。日间采用修正后的实际直线行车道，旧日间区域混入建筑和停放车辆，其指标作废。C32不覆盖所选道路，未统计该道路目标；特殊夜间只在右侧192像素有效标注内统计。用户截图未对应此处具体场景和帧号。

## 4～64像素微小目标

GT辅助的亮暗连通区域统计；背景为各方法完整序列中位数，扣除每帧道路整体漂移。历史残留和位置偏差越低越好，对比比例接近1较好。静态道路变化包含噪声与真实变化，不能单独称为噪声。

| 实验 | 场景 | 方法 | 旧位置超额灰度 | 重心误差 px | 目标对比比例 | 道路静态变化 |
|---|---|---|---:|---:|---:|---:|
| targets_tiny_adaptive_light | weather_light | 原半网格九帧 | 2.7352 | 3.2610 | 0.8054 | 0.9516 |
| targets_tiny_adaptive_light | weather_light | 原四分之一九帧 | 1.8990 | 2.8034 | 0.8121 | 1.0885 |
| targets_tiny_adaptive_light | weather_light | 局部当前半网格 | 2.0465 | 2.6039 | 0.9349 | 1.0305 |
| targets_tiny_adaptive_light | weather_light | 局部当前四分之一 | 1.8672 | 2.6772 | 0.8341 | 1.1428 |
| targets_tiny_adaptive_light | weather_medium | 原半网格九帧 | 4.1011 | 2.2240 | 0.9099 | 0.9459 |
| targets_tiny_adaptive_light | weather_medium | 原四分之一九帧 | 3.1489 | 1.7492 | 0.9472 | 1.0826 |
| targets_tiny_adaptive_light | weather_medium | 局部当前半网格 | 3.3744 | 1.6437 | 1.0347 | 1.0269 |
| targets_tiny_adaptive_light | weather_medium | 局部当前四分之一 | 3.1455 | 1.7478 | 0.9621 | 1.1390 |
| targets_tiny_adaptive_heavy | weather_heavy | 原半网格九帧 | 3.3491 | 3.2377 | 0.8401 | 1.5484 |
| targets_tiny_adaptive_heavy | weather_heavy | 原四分之一九帧 | 2.6891 | 2.8117 | 0.8791 | 1.7858 |
| targets_tiny_adaptive_heavy | weather_heavy | 局部当前半网格 | 2.7479 | 2.6089 | 0.9538 | 1.8190 |
| targets_tiny_adaptive_heavy | weather_heavy | 局部当前四分之一 | 2.6127 | 2.7053 | 0.9191 | 2.0931 |
| targets_tiny_half3_light | weather_light | 原半网格九帧 | 2.7352 | 3.2610 | 0.8054 | 0.9516 |
| targets_tiny_half3_light | weather_light | 最近三帧训练 | 2.1692 | 2.7527 | 0.8714 | 0.8889 |
| targets_tiny_half3_light | weather_medium | 原半网格九帧 | 4.1011 | 2.2240 | 0.9099 | 0.9459 |
| targets_tiny_half3_light | weather_medium | 最近三帧训练 | 3.5560 | 1.8949 | 0.9835 | 0.8831 |
| targets_tiny_half3_heavy | weather_heavy | 原半网格九帧 | 3.3491 | 3.2377 | 0.8401 | 1.5484 |
| targets_tiny_half3_heavy | weather_heavy | 最近三帧训练 | 2.9400 | 3.0217 | 0.9027 | 1.4615 |
| targets_tiny_night_strong | night_ordinary | 原夜间九帧 | 1.4946 | 4.5349 | 0.2422 | 1.0015 |
| targets_tiny_night_strong | night_ordinary | 局部当前夜间 | 1.3717 | 1.6669 | 0.5975 | 1.1527 |
| targets_tiny_night_strong | night_special | 原夜间九帧 | 1.8051 | 6.7384 | 0.2996 | 1.1608 |
| targets_tiny_night_strong | night_special | 局部当前夜间 | 1.7814 | 3.0931 | 0.7957 | 1.9686 |
| targets_tiny_night_weak | night_ordinary | 原夜间九帧 | 1.4946 | 4.5349 | 0.2422 | 1.0015 |
| targets_tiny_night_weak | night_ordinary | 局部当前夜间 | 1.4516 | 2.7077 | 0.3147 | 1.0241 |
| targets_tiny_night_weak | night_special | 原夜间九帧 | 1.8051 | 6.7384 | 0.2996 | 1.1608 |
| targets_tiny_night_weak | night_special | 局部当前夜间 | 1.7598 | 4.0918 | 0.4153 | 1.3121 |
| targets_tiny_half1_light | weather_light | 原半网格九帧 | 2.7352 | 3.2610 | 0.8054 | 0.9516 |
| targets_tiny_half1_light | weather_light | 当前帧训练 | 1.9164 | 2.5976 | 0.9386 | 1.3568 |
| targets_tiny_half1_light | weather_medium | 原半网格九帧 | 4.1011 | 2.2240 | 0.9099 | 0.9459 |
| targets_tiny_half1_light | weather_medium | 当前帧训练 | 3.2952 | 1.6329 | 1.0361 | 1.3459 |
| targets_tiny_half1_heavy | weather_heavy | 原半网格九帧 | 3.3491 | 3.2377 | 0.8401 | 1.5484 |
| targets_tiny_half1_heavy | weather_heavy | 当前帧训练 | 2.6510 | 2.6219 | 0.9252 | 2.2095 |
| targets_tiny_lane_day_adaptive | day_normal | 原半网格九帧 | 4.2545 | 1.8957 | 0.6588 | 2.1090 |
| targets_tiny_lane_day_adaptive | day_normal | 局部当前半网格 | 4.1951 | 1.7939 | 0.6881 | 2.1390 |
| targets_tiny_lane_day_residual | day_normal | 原半网格九帧 | 4.2545 | 1.8957 | 0.6588 | 2.1090 |
| targets_tiny_lane_day_residual | day_normal | 目标残差半网格 | 3.8785 | 1.6616 | 0.7792 | 2.1626 |
| targets_tiny_lane_day_three | day_normal | 原半网格九帧 | 4.2545 | 1.8957 | 0.6588 | 2.1090 |
| targets_tiny_lane_day_three | day_normal | 最近三帧训练 | 4.2109 | 1.8433 | 0.6762 | 2.0869 |
| targets_tiny_lane_day_one | day_normal | 原半网格九帧 | 4.2545 | 1.8957 | 0.6588 | 2.1090 |
| targets_tiny_lane_day_one | day_normal | 当前帧训练 | 4.1351 | 1.7298 | 0.6952 | 2.2261 |

目标区域由GT阈值辅助提取，未有人工作目标逐个标注；小幅数值改变不视为显著结果。日间采用修正后的实际直线行车道，旧日间区域混入建筑和停放车辆，其指标作废。C32不覆盖所选道路，未统计该道路目标；特殊夜间只在右侧192像素有效标注内统计。用户截图未对应此处具体场景和帧号。

## 文件和数值核验

- [exact_layout_five_r2/all_sequence_verification.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/exact_layout_five_r2/all_sequence_verification.json)
- [exact_layout_night_r2/all_sequence_verification.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/exact_layout_night_r2/all_sequence_verification.json)
- [day_backup_float/day_backup.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/day_backup_float/day_backup.json)
- [night_tail_full_sequence/night_tail_full_sequence.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/night_tail_full_sequence/night_tail_full_sequence.json)
- [review_file_verification.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/review_file_verification.json)
- [local_file_verification.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/local_file_verification.json)
- [INVALID_EARLY_WEATHER_CONFIG.json](D:/work_project/rawIR/reports/ss928_all_models_round2_20260930/results/INVALID_EARLY_WEATHER_CONFIG.json)
