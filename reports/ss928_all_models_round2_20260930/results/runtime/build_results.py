"""Build local review tables from explicitly selected, valid experiment records."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];R=BASE/'results'
def read(path):return json.loads((R/path).read_text(encoding='utf-8'))
def link(path):return f'[{path}]({(R/path).as_posix()})'
lines=['# 完整实验数值','', '表内 GPU 时间为预先就绪输入到完整3072×3840输出，RTX5090，预热30次、计时300次、均值与P95；不包含RAW读取、输入准备、主机传输。NPU未实测。','', '## 保留原浮点输出接口的完整前向','', '| 路径 | 原均值/P95 ms | 新均值/P95 ms | 均值下降 | 卷积乘加 G |','|---|---:|---:|---:|---:|']
j=read('float_speed_profile/float_speed_profile.json')
for scene,v in j['scenes'].items():
    a,b=[v['cases'][k]['GPU'] for k in ['original_float','repeat_float']]
    lines.append(f"| {scene} | {a['mean_ms']:.6f}/{a['p95_ms']:.6f} | {b['mean_ms']:.6f}/{b['p95_ms']:.6f} | {100*(1-b['mean_ms']/a['mean_ms']):.1f}% | {v['profile']['convolution_MAC']/1e9:.3f} |")
j=read('day_backup_float/day_backup.json');a,b=[j['cases'][k]['GPU'] for k in ['original_float','float_repeat']]
lines.append(f"| day_current_backup | {a['mean_ms']:.6f}/{a['p95_ms']:.6f} | {b['mean_ms']:.6f}/{b['p95_ms']:.6f} | {100*(1-b['mean_ms']/a['mean_ms']):.1f}% | — |")
lines+=['',link('float_speed_profile/float_speed_profile.json')+'；'+link('day_backup_float/day_backup.json')+'。卷积量包含固定卷积，填充位置按常规乘加口径计数，不含布局和采样的内存开销，不代表NPU时间或内存峰值。','', '编译后相对于直接运行的首帧浮点误差已单独记录，布局同精度逐帧一致性是在直接运行后端核对；不能将其推广为编译器或NPU逐位一致。','', '## 相同完整字节输出的前向','', '| 路径 | 原均值/P95 ms | 新均值/P95 ms | 均值下降 |','|---|---:|---:|---:|']
for folder,case in [('speed_five_r2','repeat_byte'),('speed_quarter_correct','repeat_byte'),('speed_night_actual','legacy_repeat')]:
    j=read(folder+'/all_speed.json')
    for scene,v in j['scenes'].items():
        if folder=='speed_five_r2' and scene.endswith('quarter'):continue
        a,b=[v['cases'][k]['GPU'] for k in ['original_byte',case]]
        lines.append(f"| {scene} | {a['mean_ms']:.6f}/{a['p95_ms']:.6f} | {b['mean_ms']:.6f}/{b['p95_ms']:.6f} | {100*(1-b['mean_ms']/a['mean_ms']):.1f}% |")
j=read('day_backup_float/day_backup.json');a,b=[j['cases'][k]['GPU'] for k in ['original_byte','repeat']]
lines.append(f"| day_current_backup | {a['mean_ms']:.6f}/{a['p95_ms']:.6f} | {b['mean_ms']:.6f}/{b['p95_ms']:.6f} | {100*(1-b['mean_ms']/a['mean_ms']):.1f}% |")
lines+=['','字节输出在小网格截断后再展开；与原图末端截断的字节输出核对。Float16接口与UInt8接口分别比较，表间时间不直接合并。','', '## 完整连续画质','', '质量数值按各场景真实GT有效区域、去掉3像素边界计算。PSNR单位为分贝；静态变化和弱结构误差以灰度计，越低越好；平均运动响应不能代替小目标定位。','', '| 实验 | 场景 | 方法 | PSNR | 静态变化 | 弱结构误差 | 运动响应 |','|---|---|---|---:|---:|---:|---:|']
groups=['adaptive_correct_light','adaptive_correct_heavy','adaptive_nonzero_day','residual_eval_light','residual_eval_heavy','residual_nonzero_eval_day','correct_fp32_eval_light','correct_fp32_eval_heavy','half3_eval_light','half3_eval_heavy','half3_eval_day','half1_eval_light','half1_eval_heavy','half1_eval_day','night_motion_r2','night_motion_weak']
for folder in groups:
    name='night_motion.json' if folder.startswith('night_motion') else 'evaluation.json';j=read(folder+'/'+name)
    for scene,v in j['scenes'].items():
        for method,m in v['summary'].items():
            label=method.replace('九帧偏重当前训练','九帧目标约束训练')
            lines.append(f"| {folder} | {scene} | {label} | {m['psnr_db']:.4f} | {m['static_residual_change_gray']:.4f} | {m['weak_structure_error_gray']:.4f} | {m['motion_response_ratio']:.4f} |")
lines+=['','正确FP32九帧实验初始化保留原权重，没有预设偏重当前系数；早期视频沿用“偏重当前”标签，正确含义为九帧目标约束训练。原标签保留，避免修改既有证据。','']
for tiny in [False,True]:
    lines+=['## '+('4～64像素微小目标' if tiny else '4～600像素道路目标'),'', 'GT辅助的亮暗连通区域统计；背景为各方法完整序列中位数，扣除每帧道路整体漂移。历史残留和位置偏差越低越好，对比比例接近1较好。静态道路变化包含噪声与真实变化，不能单独称为噪声。','', '| 实验 | 场景 | 方法 | 旧位置超额灰度 | 重心误差 px | 目标对比比例 | 道路静态变化 |','|---|---|---|---:|---:|---:|---:|']
    dirs=(['targets_tiny_adaptive_light','targets_tiny_adaptive_heavy','targets_tiny_adaptive_day','targets_tiny_residual_day','targets_tiny_half3_light','targets_tiny_half3_heavy','targets_tiny_half3_day','targets_tiny_night_strong','targets_tiny_night_weak'] if tiny else ['targets_adaptive_light','targets_adaptive_heavy','targets_adaptive_nonzero_day','targets_residual_light','targets_residual_heavy','targets_residual_day','targets_fp32_light','targets_fp32_heavy','targets_half3_light','targets_half3_heavy','targets_half3_day','targets_night_strong','targets_night_weak'])
    dirs+=['targets_'+('tiny_' if tiny else '')+'half1_'+g for g in ['day','light','heavy']]
    dirs=[d for d in dirs if not (d.endswith('_day') or d.endswith('_nonzero_day'))]
    dirs+=['targets_'+('tiny_' if tiny else '')+'lane_day_'+label for label in ['adaptive','residual','three','one']]
    for folder in dirs:
        j=read(folder+'/signed_target_summary.json')
        for scene,v in j['scenes'].items():
            for method,m in v['summary'].items():
                label=method.replace('九帧偏重当前训练','九帧目标约束训练')
                def value(k):return f'{m[k]:.4f}' if k in m else '无有效样本'
                lines.append(f"| {folder} | {scene} | {label} | {value('historical_only_signed_excess_gray')} | {value('local_target_centroid_error_pixels')} | {value('current_target_signed_contrast_ratio')} | {value('road_static_residual_change_gray')} |")
    lines+=['','目标区域由GT阈值辅助提取，未有人工作目标逐个标注；小幅数值改变不视为显著结果。日间采用修正后的实际直线行车道，旧日间区域混入建筑和停放车辆，其指标作废。C32不覆盖所选道路，未统计该道路目标；特殊夜间只在右侧192像素有效标注内统计。用户截图未对应此处具体场景和帧号。','']
lines+=['## 文件和数值核验','']
for p in ['exact_layout_five_r2/all_sequence_verification.json','exact_layout_night_r2/all_sequence_verification.json','day_backup_float/day_backup.json','night_tail_full_sequence/night_tail_full_sequence.json','review_file_verification.json','local_file_verification.json','INVALID_EARLY_WEATHER_CONFIG.json']:
    lines.append('- '+link(p))
(BASE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('RESULT_TABLES_WRITTEN')
