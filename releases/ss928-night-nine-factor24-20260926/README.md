# 夜间九帧 24 通道提速候选：2026-09-26

本包包含普通夜间、特殊夜间两组最新候选，供 SS928 部署端编译和实测。分支 `codex/ss928-night-nine-factor24-20260926`，目标仓库 `https://github.com/Ah-saber/apple.git`。原有四场景版本保持在其原目录。

服务器 RTX 5090 同条件整图配对测速：普通 0.8759→0.7584 毫秒，特殊夜间 0.8518→0.7302 毫秒，提速约 13%～14%。不含预处理、传输和首次编译；这些结果不代表 SS928 时间。板端目标为完整同步模型第 95 百分位耗时不超过 16.7 毫秒，目前没有达标证据。

## 模型选择

| 场景 | 半精度九帧输入 | 单精度九帧输入 |
| --- | --- | --- |
| 普通夜间 | `models/ordinary_factor24_1024x1280_float16_nozero.onnx` | `models/ordinary_factor24_1024x1280_float32_nozero.onnx` |
| 特殊夜间 | `models/special_factor24_1024x1280_float16_nozero.onnx` | `models/special_factor24_1024x1280_float32_nozero.onnx` |

四个模型均为完整静态尺寸，opset 17，已通过 ONNX 静态检查，未进行 SS928 编译或执行。单精度输入只改变接口，图内卷积仍为半精度，轨迹统计使用单精度，输出为 8 位灰度。

## 输入输出

连续布局 NCHW，依次为批次、通道、高、宽；数值文件使用小端。

| 名称 | 形状 | 类型 | 有效字节 |
| --- | --- | --- | ---: |
| `nine_raw` | `[1,9,1024,1280]` | 所选图的 float16 或 float32 | 23,592,960 或 47,185,920 |
| `reference_thumb` | `[1,1,64,64]` | float32 | 16,384 |
| `display_uint8` | `[1,1,3072,3840]` | uint8 | 11,796,480 |

九帧由旧到新，片段起点不足九帧时重复片段首帧。普通与特殊夜间的校正不同。完整规则、代码、已有校准参数与固定场在 `preprocess/PREPROCESS.md` 及同目录。新序列必须使用匹配的校准参数。先按既有单精度规则完成 RAW 校正归一化，再将九帧张量转为所选输入精度；缩略图保持单精度。

## 核对转换输出

`input_vectors/` 包含两个真实整图输入。里面旧的 `display_uint8` 字段不属于本候选，工具只读取九帧输入和缩略图。本候选参考输出仅使用 `reference_outputs/`，分别保存两种输入精度的源模型输出和服务器编译输出；每种接口两条执行路径最大差异均为 1 灰度。另用包内独立命令核对半精度输入：普通源模型与生成参考相差最多 1 灰度、涉及 240/11,796,480 个像素，特殊夜间完全一致；记录见 `evidence/package_runtime_verification.json`。

在包根目录用已配置依赖的环境运行，输出目录须不存在：

```sh
python tools/prepare_vectors.py --input-precision float16 --output work/fp16
python tools/prepare_vectors.py --input-precision float32 --output work/fp32
```

将对应文件绑定两个输入。根据设备张量描述去除填充，将有效字节输出保存为连续 NCHW 文件，再比较：

```sh
python tools/compare_uint8.py --scene ordinary --input-precision float16 --actual work/ordinary_board.u8 --output work/ordinary_compare.json
python tools/compare_uint8.py --scene special --input-precision float16 --actual work/special_board.u8 --output work/special_compare.json
```

单精度接口使用 `--input-precision float32`。默认比较源模型输出，`--reference-runtime compiled` 可比较服务器编译输出。输出平均、第 99 百分位和最大差异、超过 1 灰度的像素比例及差异位置。量化后的画质必须另查完整连续帧，单样例数值检查不能替代视频检查。

## 改动和精度

第一层 5×5 卷积近似分为 1×5、5×1 两层，中间 24 通道。通过原训练区域的输入统计拟合，未使用 GT 或梯度训练。该步骤有近似误差。前部卷积每像素乘加从上个合并版的 6000 次降至 4280 次。随后删除轨迹最大值的一次冗余零初始化，普通 60 帧、特殊夜间 120 帧完整输出逐像素一致。

相对上个合并版，普通 PSNR 26.98575→26.99637，特殊夜间 25.97273→25.97885。静态时序误差略降；普通亮度时序误差增加约 2.07%，特殊增加约 0.18%。所有指标并非同时改善。普通 16 通道候选出现更明显闪烁指标退步，已淘汰。

指标范围为固定留出区域：普通 `(y=2,x=0,h=1020,w=1278)`，特殊夜间 `(y=2,x=1088,h=1020,w=192)`。特殊夜间指标不覆盖整幅宽度；模型推理与视频覆盖整图。低幅运动指标并非人工标注的弱目标检出率。本轮验证覆盖这两个夜间序列。

完整视频四栏为 RAW、GT、上个合并版、当前候选，5120×1080、12 帧/秒，模型三倍输出按面积平均回原 RAW 网格显示：

- `evidence/ordinary_weighted24_quality.mp4`：60 帧。
- `evidence/special_weighted24_quality.mp4`：120 帧。

详细指标、测试方法、测速采样和一致性证据见 `evidence/`。其中 `RESULTS.md` 引用的是实验原目录；实际文件按本包 `evidence/`、`models/` 对应读取。

## NPU 编译与计时

使用部署端已验证工具链，依据本图两个输入的名称、形状和类型设置，输出为 uint8，关闭额外均值缩放。不能沿用旧单输入 `raw` 模型的接口参数。未附带未经验证的 OM 或编译命令。

重点核对 Resize、Erf、Round、字节输出转换、混合精度以及轨迹相关操作；检查算子分配和 CPU 回退。新分解中间特征若完整写入内存占 60 MiB，写读约产生 120 MiB 访问；实际分块融合未知，计算量下降不能保证 NPU 提速。旧部署报告中 Resize 曾回退 CPU，需重新核对当前图。

保留转换器版本、完整命令、日志、量化选项、OM 校验值与输入输出描述。固定设备负载和完整尺寸，预热 50 次后至少连续测 600 次，异步执行须等待完成。记录平均、第 95 百分位、最大耗时和逐次样本：

1. 设备输入已就绪至完整模型输出及同步完成。
2. RAW 预处理、输入传输、输出传输及布局整理分别计时。
3. 主机 RAW 就绪至完整 8 位输出在主机可用的总时间；离线序列标定和文件读写另列。

填写模板见 `tools/board_timing_template.csv`。

## 独立运行与校验

模型权重及运行源码齐全，无须原训练仓库。`*_fused.pt` 用于构建冻结结构，`*_factor24.pt` 覆盖完整候选权重；加载器校验基准权重 SHA 并严格加载，自动使用删除零初始化的轨迹实现。

在已配置的 test 环境运行，验证环境为 Python 3.10、PyTorch 2.7.0+cu128：

```sh
python runtime/infer_vector.py --model models/ordinary_factor24.pt --baseline models/ordinary_fused.pt --input input_vectors/ordinary_frame_20.npz --output work/ordinary_server.u8 --execution source --input-precision float16 --device cuda
```

特殊夜间改用 `special_factor24.pt`、`special_fused.pt` 和 `special_frame_60.npz`。`--execution compiled` 为服务器编译路径。默认显存上限 2.5 GiB，可使用 `--gpu-lock` 指定共享锁；输出文件须不存在。

`reproduce/` 保留实验入口，服务器绝对路径仅用于追溯。独立使用本包 `runtime/` 和 `tools/`。来源及许可证见 `runtime/PROVENANCE.json`、`runtime/LICENSE`。所有文件校验：

```sh
sha256sum -c MANIFEST.sha256
```
