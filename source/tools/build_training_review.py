"""Build an offline browser review page from downloaded run artifacts."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    completed = json.loads((args.run/'completed.json').read_text(encoding='utf-8'))
    labels = {'day_normal':'正常白天','weather_light':'天气轻档','weather_medium':'天气中档',
              'weather_heavy':'天气重档','weather_heavy_c32':'重档＋宽尺度补偿',
              'night_ordinary':'普通夜间','night_special':'特殊夜间'}
    labels = {scene: label for scene, label in labels.items() if scene in result['validation']}
    rows, options = [],[]
    for scene,label in labels.items():
        a,b=result['validation'][scene],result['test'][scene]
        scope='空间开发' if a['evaluation_scope']=='spatial_development' else '采集组留出'
        rows.append(f'<tr><td>{label}</td><td>{a["psnr"]:.3f}</td><td>{a["ssim"]:.4f}</td>'
                    f'<td>{b["psnr"]:.3f}</td><td>{b["ssim"]:.4f}</td><td>{scope}</td></tr>')
        options.append(f'<option value="{scene}">{label}</option>')
    best=result['selected_checkpoint']['step']
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>B0 红外超分首轮验收</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#101722;color:#e9eef8;max-width:1450px;margin:32px auto;padding:0 22px}
h1,h2{line-height:1.25}h2{margin-top:38px}p{max-width:1050px}.muted{color:#adb9cc}.note{background:#243048;border-left:4px solid #e6b75a;padding:15px 20px}
table{border-collapse:collapse;width:100%;background:#192434}th,td{padding:10px 14px;border-bottom:1px solid #384354;text-align:left}
img,video{max-width:100%;height:auto;background:#0b0e13;border-radius:8px}a{color:#9ec7ff}select,button{font:inherit;padding:8px 14px;background:#25344b;color:white;border:1px solid #5b7290;border-radius:5px}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{overflow-x:auto}.caption{color:#adb9cc;font-size:14px}
@media(max-width:800px){.pair{grid-template-columns:1fr}td,th{padding:6px;font-size:12px}}
</style><h1>RT4KSR B0 · RAW → 8bit ×3</h1>
<p class="muted">EXP-TASK-019-004 · D1 / S1 · GPU 0 · 首轮训练结果，等待画质验收</p>
<div class="note">完成 STEPS 步；按七类等权验证 PSNR 选择第 BEST 步。测试按预设间隔监控，不用于选权重。
GT 是 JPEG / 传统增强教师；下表属于合成低分辨率评测。完整 RAW 的三倍输出没有真实 HR 指标。两类空间开发不代表独立泛化。</div>
<h2>各场景指标</h2><div class="panel"><table><thead><tr><th>场景</th><th>验证 PSNR</th><th>验证 SSIM</th><th>测试 PSNR</th><th>测试 SSIM</th><th>范围</th></tr></thead><tbody>ROWS</tbody></table></div>
<h2>查看图像与连续帧</h2><div class="toolbar"><label>场景 <select id="scene">OPTIONS</select></label><label>图片 <select id="mode"><option value="overview">完整 RAW 推理概览</option><option value="detail">原分辨率固定 ROI</option><option value="val">合成低分辨率验证</option><option value="test">合成低分辨率测试</option></select></label></div>
<p id="caption" class="caption"></p><a id="imageLink" target="_blank"><img id="comparison" alt="场景对照"></a>
<div class="toolbar"><a id="full" target="_blank">打开完整 3072×3840 SR PNG</a><a id="reference" target="_blank">打开教师插值参考</a></div>
<p class="caption">视频左侧：原尺寸教师；右侧：完整 RAW 三倍推理后缩小供全图浏览。12 帧、6 fps 循环仅供短片检查，不能代替长序列稳定性验收。</p>
<video id="preview" controls muted loop playsinline preload="metadata"></video>
<h2>训练记录</h2><img src="training_curves.png" alt="训练损失和学习率曲线"><img src="scene_metric_curves.png" alt="各场景PSNR与SSIM曲线">
<h2>权重与证据</h2><p><a href="checkpoints/best.pt">训练 best</a> · <a href="checkpoints/last.pt">训练 last</a> · <a href="artifacts/b0_deploy.pt">融合部署权重</a> · <a href="config.json">固定配置</a> · <a href="result.json">最终指标</a> · <a href="checkpoint_index.json">权重散列索引</a> · <a href="metrics.jsonl">全部评测记录</a></p>
<p class="muted">SS928 尚未转换和上板。服务器速度不能用于证明 NPU &lt;16.7 ms。ONNX及数值检查另见 EXP-TASK-019-005。</p>
<script>
const scene=document.querySelector('#scene'),mode=document.querySelector('#mode');
function update(changeVideo){const s=scene.value,m=mode.value;let path;
if(m==='val'||m==='test'){path=`selected_best/${m}/examples/${s}.jpg`;document.querySelector('#caption').textContent='从左到右：固定 DN 显示、教师目标、B0 预测、绝对误差×4。空间两类已先裁区再推理。';}
else {path=`native_qualitative/${s}_${m}.jpg`;document.querySelector('#caption').textContent='左：教师双三次插值，仅作视觉参考；右：B0 完整 RAW 输入的三倍输出。点击图片单独查看。';}
document.querySelector('#comparison').src=path;document.querySelector('#imageLink').href=path;
document.querySelector('#full').href=`native_qualitative/${s}_sr_x3.png`;document.querySelector('#reference').href=`native_qualitative/${s}_teacher_bicubic_reference.png`;
if(changeVideo){const v=document.querySelector('#preview');v.src=`native_qualitative/${s}_12frames_preview.mp4`;v.load();}}
scene.addEventListener('change',()=>update(true));mode.addEventListener('change',()=>update(false));update(true);
</script></html>'''
    document=document.replace('STEPS',str(completed['steps'])).replace('BEST',str(best))
    document=document.replace('ROWS',''.join(rows)).replace('OPTIONS',''.join(options))
    document=document.replace('EXP-TASK-019-004',html.escape(args.run.name))
    document=document.replace('首轮验收','训练验收').replace('首轮训练结果','本轮训练结果')
    document=document.replace('七类',str(len(labels))+'类').replace('两类空间开发','空间开发子集').replace('空间两类','空间开发子集')
    document=document.replace('ONNX及数值检查另见 EXP-TASK-019-005。','ONNX及数值检查见本轮实验报告。')
    (args.run/'review.html').write_text(document,encoding='utf-8')
    print(args.run/'review.html')


if __name__ == '__main__':
    main()
