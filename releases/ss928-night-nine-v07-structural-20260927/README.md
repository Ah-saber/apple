# v0.7 后续：统计布局、九相位输出与参考激活

目标：完整SS928模型推理≤16.7 ms，保留九帧去噪、连续帧亮度与局部对比度、弱目标。当前板端基线仍为普通81.228／特殊80.351 ms；本包尚无新NPU结果。

本轮已实现统计空间打包、两种轴向完整36相位重排、四种输出相位数量、原生像素均值约束、输出层量化搜索，以及保留多尺度的ReLU参考分支训练。所有候选和未采用方案均保留。

新组合实测GPU：普通0.377052→0.349534 ms（下降7.30%），特殊0.382930→0.349969 ms（下降8.61%）。对应同轮RTX5090、完整输入输出、30预热／600逐次事件计时，无CUDA图。新组合180帧RAW尺寸PSNR普通+0.041／特殊+0.052 dB；两处普通弱目标响应约下降1.64%／1.54%。完整3倍纹理改变，不能宣称无损。

详细数字、逐模块状态、失败方案及NPU边界见 [RESULTS.md](RESULTS.md)。最新接手状态见 [HANDOFF.md](HANDOFF.md)。

## 文件

- `source_onnx/`：完整1024×1280输入与64×96小图；`_rewritten` 为在冻结原图上执行受约束改写的候选。全图输出均为3072×3840。`_small`仅用于数值控制，不交付为完整画质视频。
- `onnx/`：完整尺寸的统计、轴向36相位、九相位输出独立算子微图，F16／F32两套。
- `runtime/`、`models/`：可重现Torch源码和检查点；`tools/`：供部署端直接改写已有native_r2的工具。
- `evidence/`：逐次计时、180帧画质、弱目标窗口、训练记录、量化假设和独立ONNX数值检查。
- `videos/`：两个完整场景输入／GT／v0.7源码／新组合四栏对照，以及完整3倍中间帧PNG。视频各栏显示完整RAW视野，3倍结果平均回RAW尺寸用于同视野比较。单独PNG保留真实3倍分辨率。
- `MANIFEST.json`、`SHA256SUMS`：文件职责、未验证状态和全部交付文件校验值。

## 部署测哪些已有版本

先以相同编译、量化、校准输入和同步计时协议复测native_r2基线。已有完整候选可分别测：

1. `space_pack`：保留36相位，只换统计布局。
2. `temporal_space_pack`：保留36相位，先九帧到三组统计再空间打包。
3. `phase3_quantsearch_nearest`：量化搜索后的9相位，3倍重排再2倍最近邻扩展。
4. `phase3_quantsearch_fixed`：相同9相位权重，一个固定6×6反卷积展开；对应`_rewritten`图是真正单反卷积。
5. `reference_relu_output_stable`：保留多尺度、重新训练的参考ReLU。
6. `space_pack_reference_relu_output_stable_phase3_quantsearch_nearest`：有完整180帧证据与最新GPU实测的组合。
7. `space_pack_reference_relu_output_stable_phase3_quantsearch_fixed_rewritten`：同组合的单反卷积部署路径；数值控制以最近邻源码为参照。

解析均值约束9相位（`preserve`）、保留原36相位的候选仍可使用。1×1输出、简单4／1相位已发生画质问题，供研究回溯，不推荐直接部署。

所有源ONNX仅完成ONNX结构与数值检查，未经过SS928 SDK转换。部署端实际native_r2可能已经改写参考缩放与输出重排，优先使用下面的受约束工具保留其兼容处理。若native图没有唯一CRD排列的DepthToSpace6，工具会拒绝；不要手动绕过保护，请回报实际输出子图结构。

## 在部署端已有native_r2上组合改写

以普通场景为例，在本包目录执行；Python环境需`onnx`、`numpy`：

```bash
python tools/build_native_candidate.py \
  --native /path/to/ordinary_native_r2.onnx \
  --old-source source_onnx/ordinary_baseline.onnx \
  --ref-source source_onnx/ordinary_reference_relu_output_stable.onnx \
  --projection-source source_onnx/ordinary_phase3_quantsearch_nearest.onnx \
  --statistics space_pack \
  --phase-mode preserve_nearest \
  --output /path/to/new_unique/ordinary_space_relu_phase9.onnx
```

特殊场景将`ordinary`替换为`special`。单反卷积输出改为`--phase-mode preserve_fixed`。只测前端可省略三个source参数及phase-mode；只测9相位可省略statistics与ref-source。已存在输出拒绝覆盖。工具核对原生输出卷积权重必须与冻结基线相符，参考分支逐层匹配原权重，保留部署图已有参考缩放兼容路径；会输出SHA与未验证元数据。

源36相位输出若已被native_r2完全展开成六路反卷积且移除DepthToSpace6，应在部署侧参考兼容修复之前调用本工具，然后沿用部署侧修复流程。实际native_r2未传回，本包没有冒充实际R2等价验证。

## Torch重现依赖

使用`test`环境。本包默认查找同级 `ss928-night-nine-system-speed-20260926/`，该冻结版本还依赖同级 `ss928-night-nine-packed-front-20260926/`。二者均在apple原提交保存，保留完整目录。可用`RAWIR_SYSTEM_RELEASE_DIR`显式指定原system-speed目录。独立ONNX推理与部署图改写不依赖这两个Torch目录。

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('runtime').resolve()))
from structural_candidates import load_structural
model = load_structural('ordinary',
    'space_pack_reference_relu_output_stable_phase3_quantsearch_nearest',
    Path('models'), device='cuda')
# model(nine_raw_float32, reference_thumb_float32)
```

研究训练、测速与完整序列脚本保留当时远程数据路径；迁移数据需调整记录路径，不能把它们当作通用无数据演示。`export_searched.py`保留为备用Torch导出脚本，本轮新权重ONNX实际由`build_searched_local.py`基于此前已验证的导出结构生成，独立检查记录注明执行方式。

## 接下来需要的板端信息

请返回每个候选的转换成功／失败信息、AICPU情况、完整同步耗时、任务分解，重点核对统计SpaceToDepth、Resize×2与单反卷积展开是否被拆成多任务，并在相同输入上返回精度对照。新图未编译时，GPU不能确定SS928采用哪一个输出路径。

目标16.7 ms未达到。输入／Report格式转换和主体保留项没有本轮已证实的10倍收益；本包不把名义乘加数变化、单层量化压力检查或GPU下降当作NPU结果。

## 本轮交付数值验证范围

204个ONNX结构检查通过；此前统计／参考分支完整小图控制两场景各30项及固定参考卷积保护通过，记录见`evidence/*_export_checks.json`。新增搜索权重到源码／改写图数组逐元素相等；独立ONNX参考引擎验证30项最终输出子图控制、两项原生权重不匹配拒绝，以及12项9相位微图映射。扩大到新组合完整小图的每场景15项检查因本地参考引擎过慢已中止，未标记为通过。实际native_r2与SS928 SDK仍未经验证。
