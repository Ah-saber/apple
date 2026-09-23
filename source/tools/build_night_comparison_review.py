"""Create an offline night-run comparison page without modifying frozen run files."""
import argparse
import html
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new review path')
    comparison = json.loads((args.comparison / 'comparison.json').read_text())
    completed = json.loads((args.run / 'completed.json').read_text())
    config = json.loads((args.run / 'config.json').read_text())
    names = {'night_ordinary': '普通夜间', 'night_special': '特殊夜间'}
    def link(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        return html.escape(os.path.relpath(path, args.output.parent).replace('\\', '/'), quote=True)
    rows = []
    for split, label in [('val', '验证'), ('test', '测试')]:
        for scene, name in names.items():
            row = comparison['splits'][split]['scenes'][scene]
            old, new = row['mixed'], row['night']
            rows.append(f'<tr><td>{label} · {name}</td><td>{old["psnr"]:.3f} / {old["ssim"]:.4f}</td>'
                        f'<td>{new["psnr"]:.3f} / {new["ssim"]:.4f}</td><td>{row["delta_psnr"]:+.3f} dB</td></tr>')
    sections = []
    visuals = args.comparison / 'comparison_visuals'
    for scene, name in names.items():
        overview = link(visuals / (scene + '_overview.jpg'))
        building = link(visuals / (scene + '_building.jpg'))
        top = link(visuals / (scene + '_top.jpg'))
        video = link(visuals / (scene + '_12frames_comparison.mp4'))
        native = link(args.run / 'native_qualitative' / (scene + '_sr_x3.png'))
        sections.append(f'<section><h2>{name}</h2><p>左：教师插值参考；中：上一轮夜间 P128；右：本轮夜间 P256。</p>'
                        f'<a href="{overview}" target="_blank"><img src="{overview}" alt="{name}全图对照"></a>'
                        f'<details><summary>查看建筑与顶部固定 ROI</summary><img src="{building}" alt="建筑 ROI">'
                        f'<img src="{top}" alt="顶部 ROI"></details>'
                        f'<p><a href="{native}" target="_blank">打开本轮完整 3840×3072 输出 PNG</a></p>'
                        '<p>同帧视频：左为原尺寸教师，中为上一轮SR缩小预览，右为本轮SR缩小预览。12帧、6fps，循环播放。</p>'
                        f'<video src="{video}" controls controlslist="noplaybackrate" muted autoplay loop playsinline preload="metadata"></video></section>')
    old_step = comparison['baseline_checkpoint']['step']
    new_step = comparison['night_checkpoint']['step']
    size = config['train_crop_hr']
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>夜间超分 P256 · 第二轮对照</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 24px;background:#111923;color:#edf2f7}}
h1,h2{{line-height:1.3}}section{{margin-top:36px}}a{{color:#99c7ff}}img,video{{display:block;width:100%;height:auto;margin:12px 0}}
.notice{{background:#263142;border-left:4px solid #e5b865;padding:14px 20px}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px;border-bottom:1px solid #3c4858;text-align:left}}.table{{overflow-x:auto}}summary{{cursor:pointer;padding:12px;background:#263142}}
.muted{{color:#b4c1d2}}@media(max-width:640px){{body{{padding:0 12px}}th,td{{padding:6px;font-size:13px}}}}</style>
<h1>夜间超分 · P256 / 20万步</h1>
<p>训练已完成 {completed['steps']:,} 步。训练输入 {size[0]//3}×{size[1]//3}，输出 {size[0]}×{size[1]}，batch {config['batch_size']}。</p>
<div class="notice">普通夜间测试略有改善，特殊夜间测试退化；完整 RAW 的顶部条纹和纹理问题仍未解决，未通过画质预评。</div>
<p>两轮均按两类等权验证 PSNR 选择模型：上一轮第 {old_step:,} 步，本轮第 {new_step:,} 步。测试不参与选择。</p>
<div class="table"><table><thead><tr><th>集合 / 场景</th><th>上一轮 PSNR / SSIM</th><th>本轮 PSNR / SSIM</th><th>PSNR变化</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">每类各12张验证、12张测试。特殊夜间为空间开发评测。patch面积和训练次数同时改变，不能视为单变量消融；插值教师不是三倍HR真值。</p>
{''.join(sections)}
<h2>训练曲线与文件</h2><img src="{link(args.run/'training_curves.png')}" alt="训练损失与学习率曲线">
<img src="{link(args.run/'scene_metric_curves.png')}" alt="逐场景验证和测试曲线">
<p><a href="{link(args.run/'config.json')}">固定配置</a> · <a href="{link(args.run/'result.json')}">最终指标</a> ·
<a href="{link(args.comparison/'comparison.json')}">完整对照</a> · <a href="{link(args.run/'checkpoints/best.pt')}">best权重</a> ·
<a href="{link(args.run/'checkpoints/last.pt')}">last权重</a> · <a href="{link(args.run/'artifacts/b0_deploy.pt')}">融合权重</a></p>
<p class="muted">视频已做文件解码检查；浏览器连续播放和长时序稳定性未认证。SS928算子转换、量化和板上时延尚未验证。原实验文件保持冻结。</p></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
