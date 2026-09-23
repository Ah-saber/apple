"""Write measured full-training results without treating test data as checkpoint selection."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();A=a.output
q=json.loads((A/'quality/report.json').read_text());t=json.loads((A/'timing/report.json').read_text());v=json.loads((A/'videos/report.json').read_text());completed=json.loads((a.run/'completed.json').read_text());receipt=json.loads((a.run/'exit.json').read_text())
assert q['status']==v['status']=='complete' and completed['steps']==200000 and receipt['exit_code']==0
means={sp:{k:{mode:{metric:sum(d[k+'_'+mode][metric] for d in q['summary'][sp].values())/2 for metric in ('psnr','ssim','mean_bias')} for mode in ('reduced','native')} for k in q['identities']} for sp in ('val','test')}
lines=['# S03完整训练与测试结果','',f"已从8000步续训至200000步。验证最佳权重为第{completed['best']['step']}步，最后权重为第200000步。选择依据始终为验证集裁块PSNR，整图指标和测试集没有参与选权重。训练进程退出码为0。",'', '## 结果','', '|模型|步数|验证裁块PSNR/SSIM|验证整图PSNR/SSIM|测试裁块PSNR/SSIM|测试整图PSNR/SSIM|','|---|---:|---|---|---|---|']
for k,identity in q['identities'].items():
 cells=[f"{means[sp][k][mode]['psnr']:.4f} / {means[sp][k][mode]['ssim']:.6f}" for sp,mode in [('val','reduced'),('val','native'),('test','reduced'),('test','native')]]
 lines.append(f"|{identity['label']}|{identity['step']}|"+'|'.join(cells)+'|')
lines+=['','PSNR单位dB。SSIM为结构相似程度。两场景等权；每个划分每场景12帧。整图流程为原尺寸RAW→网络3倍输出→裁剪[0,1]→面积回缩3倍→同一区域GT比较。没有真实3倍高分辨率GT，因此另存真实3倍局部图，GT放大仅作风格参考。','']
for sp in ('val','test'):
 b,n=means[sp]['s03_8k'],means[sp]['s03_best'];lines.append(f"相对S03 8000步，验证最佳权重在{sp}的整图PSNR变化为{n['native']['psnr']-b['native']['psnr']:+.4f} dB、SSIM变化{n['native']['ssim']-b['native']['ssim']:+.6f}；裁块PSNR变化{n['reduced']['psnr']-b['reduced']['psnr']:+.4f} dB。")
lines+=['','不能预先认定延长训练会改善整图效果；以表中实际结果及图片、视频判断。这里比较训练预算不同的完整方案，不是同预算结构消融。测试集此前已用于开发分析，不能称为全新盲测。','', '## 各场景','', '|划分|场景|模型|整图PSNR|整图SSIM|亮度偏差/灰阶|','|---|---|---|---:|---:|---:|']
for sp,scenes in q['summary'].items():
 for scene,d in scenes.items():
  for k in q['identities']:
   x=d[k+'_native'];lines.append(f"|{sp}|{scene}|{k}|{x['psnr']:.4f}|{x['ssim']:.6f}|{x['mean_bias']:.4f}|")
lines+=['','## 推理耗时','', '|版本|均值ms|P50 ms|P95 ms|','|---|---:|---:|---:|']
for k,x in t['results'].items():
 z=x['total'];lines.append(f"|{k}|{z['mean_ms']:.4f}|{z['p50_ms']:.4f}|{z['p95_ms']:.4f}|")
lines+=['','5090共享环境，FP32、关闭TF32，1×1×1024×1280归一化RAW输入，3倍输出；包含整图参考。fastpool为等价分级均值池化。每模型预热50次后3×200次同步计时，另存逐模块耗时。排除归一化统计、CPU/GPU传输、8bit转换和文件读写。SS928尚未实测。','', '## 连续视频','', '视频按RAW、GT、S03 8000步、验证最佳、最后200000步排列。每个验证/测试序列使用全部连续帧；另存真实3倍局部视频。统一[0,1]映射8bit，没有逐图拉伸。播放12fps仅用于查看，不代表已确认的采集帧率。','', '|划分/场景|模型|平均亮度偏差|亮度偏差时间标准差|连续残差变化MAE|','|---|---|---:|---:|---:|']
for seq in v['sequences']:
 for k,x in seq['summary'].items():
  temporal=x['consecutive_residual_change_mae_gray'];ts=f'{temporal:.5f}' if temporal is not None else '无有效连续对'
  lines.append(f"|{seq['split']}/{seq['scene']}|{k}|{x['bias_mean_gray']:.5f}|{x['bias_std_over_time_gray']:.5f}|{ts}|")
lines+=['','连续残差变化为相邻两帧(prediction−GT)差值的平均绝对值，跳过冻结分段边界。没有做运动对齐，指标仍受运动、纹理和GT变化影响，不能单独证明无闪烁；实际视频需人工查看。','', '## 训练与选择记录','', f"训练目录：`{a.run}`。权重和Adam状态、随机状态、采样epoch与next_batch均继承原8000步检查点，原学习率20万步余弦曲线不变。模型16通道4块，RAW几何增强与0.1辅助RAW损失保持。每2000步验证与保存，测试仅在200000步及最终自动评估进行。原父目录保持不变。", '', '归一化仍使用完整序列的离线统计，未验证实时因果部署。辅助RAW头仅训练使用，整图参考在推理保留。ONNX输入归一化RAW，推理无GT。', '', '## 导出与文件','']
for folder in ('onnx_best','onnx_last'):
 d=json.loads((A/folder/'verification.json').read_text());lines.append(f"- `{A/folder}`：step{d['step']}，CPU ONNX一致性通过，最大绝对误差{max(x['max_abs'] for x in d['checks']):.9g}；权重SHA256 `{d['checkpoint_sha256']}`。")
lines+=[f'- 指标及图片：`{A}/quality/`；连续视频及逐帧诊断：`{A}/videos/`。',f'- 计时：`{A}/timing/`；训练曲线：`{A}/training_curves.png`。',f'- 模型清单与校验：`{A}/models.json`；全过程状态：`{A}/status.json`。', '', '最佳和最后权重均保留；不自动替换其他场景模型。']
(A/'RESULTS.md').write_text('\n'.join(lines)+'\n')
parent=Path(json.loads((a.run/'continuation_parent.json').read_text())['parent_run']);rows=[]
for source in (parent,a.run):rows += [json.loads(x) for x in (source/'metrics.jsonl').read_text().splitlines() if x.strip()]
rows=sorted((x for x in rows if x['split']=='val'),key=lambda x:x['step']);fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric in zip(axes,('psnr','ssim')):
 ax.plot([x['step'] for x in rows],[x['macro'][metric] for x in rows],label='Validation macro');ax.axvline(8000,color='gray',linestyle='--',label='Continuation starts');ax.axvline(completed['best']['step'],color='orange',linestyle=':',label='Selected best');ax.set_xlabel('Optimizer step');ax.set_ylabel(metric.upper());ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(A/'training_curves.png',dpi=150);plt.close(fig)
(A/'summary.json').write_text(json.dumps(dict(status='complete',means=means,selected_step=completed['best']['step'],last_step=completed['steps'],timing={k:v['total'] for k,v in t['results'].items()}),indent=2))
print(A/'RESULTS.md')
