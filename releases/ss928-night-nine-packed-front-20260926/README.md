# 夜间九帧联合去噪提速候选

目标为 SS928 整模型推理 ≤16.7 ms。当前完成 GPU 实测及完整画质核对，板端编译与耗时待验证。详细实验、局限及部署必须回传的信息见 [RESULTS.md](RESULTS.md)。

## 默认候选

| 场景 | 模型 | 完整 FP32 输出 GPU 耗时 | 相对上次 S8 |
|---|---|---:|---:|
| 普通夜间 | `onnx/joint8/ordinary_float32.onnx` | 0.438714 ms | 减少 37.47% |
| 特殊夜间 | `onnx/joint12/special_float32.onnx` | 0.429291 ms | 减少 36.19% |

以上为 RTX5090 编译执行，30 次预热、三轮各 200 次，同输入输出接口，不含读取、预处理、传输与编译。相同 FP16 输入、整图 uint8 输出接口测得 0.386187 / 0.379498 ms；这组数据不能与 FP32 接口混算收益。

普通夜间 PSNR 下降约 0.014 dB，弱变化区域误差上升约 0.54%；特殊夜间 PSNR 提高约 0.040 dB。两段完整视频的静态波动、亮度误差波动与局部对比度误差波动均降低，结构相似度略降。没有验证其他部署数据的质量保证。

## 输入输出

输入 `nine_raw`：连续 NCHW、FP32 `[1,9,1024,1280]`，历史到当前排列，47,185,920 字节。输入 `reference_thumb`：FP32 `[1,1,64,64]`，16,384 字节。规则见 [preprocess/PREPROCESS.md](preprocess/PREPROCESS.md)。

输出 `display_gray`：灰度已缩放至 `[0,255]`，连续 NCHW `[1,1,3072,3840]`，默认 FP32、47,185,920 字节。显示时直接四舍五入到 uint8，不能再次乘 255。板端若有行步幅，按有效元素提取。

同目录 `*_float16.onnx` 输出 FP16、23,592,960 字节，最大舍入 0.0625 灰度级；完整序列舍入检查见 `evidence/half_*_evaluation.json`。这是可选的板端对照，默认模型保持 FP32 输出。输入仍为 FP32。

`onnx/s8_halfshuffle/` 为只改输出重排的上次 S8 对照。`joint12/ordinary_*` 与 `joint8/special_*` 为两种宽度对照；特殊八通道存在更大的闪烁误差及结构相似度退步，默认选择十二通道。

## 独立复现

在 test 环境、以本目录为当前目录执行：

```bash
sha256sum -c MANIFEST.sha256
python tools/infer_vector.py --scene ordinary --input input_vectors/ordinary_frame_20.npz --output ordinary.f32
python tools/compare_output.py --reference reference_outputs/ordinary_frame_20.npz --output ordinary.f32 --dtype float32
python tools/infer_vector.py --scene special --input input_vectors/special_frame_60.npz --output special.f32
python tools/compare_output.py --reference reference_outputs/special_frame_60.npz --output special.f32 --dtype float32
```

默认自动选择普通八通道、特殊十二通道。可选 `--variant joint8|joint12|s8`、`--dtype float16`、`--execution compiled`。GPU 执行支持 `--lock-file` 和 `--gpu-memory-gib`。不安装依赖。独立包运行与参考的平均误差分别约 0.000036 / 0.000005 灰度级；cuDNN 算法选择可能使少量像素差一档，见 `evidence/*_compare.json`。

完整对照视频在 `videos/`，四幅均显示完整传感器范围，60 / 120 帧；模型输出按 3×面积缩小以便与 RAW、GT 对齐，不裁局部。

## SS928 编译及后续优化

源码 ONNX 检查通过，未运行 SS928 SDK，不能把源码文件当成已经验证的 R4 兼容图。

部署端有实际 R4 图时，可先替换其前半段以保留原有主干和输出兼容处理：

```bash
python tools/splice_r4_packed.py --r4 ordinary_R4.onnx --student onnx/joint8/ordinary_float32.onnx --output ordinary_joint_R4.onnx
python tools/splice_r4_packed.py --r4 special_R4.onnx --student onnx/joint12/special_float32.onnx --output special_joint_R4.onnx
```

工具核对主干入口及输出卷积的权重、形状和参数，拒绝不匹配图。实际 R4 文件本地缺失，目前仅在冻结源码对照图验证。

随后可独立测输出改写，便于识别收益来源：

```bash
python tools/rewrite_r4_output.py --input ordinary_joint_R4.onnx --output ordinary_joint_halfshuffle_R4.onnx --dtype float32
python tools/rewrite_r4_output.py --input ordinary_joint_R4.onnx --output ordinary_joint_halfout_R4.onnx --dtype float16 --dcr
```

`--dcr` 仅允许六倍重排直接返回单通道输出；这个条件下两种重排排列等价。真实 R4 若由 FP32 产生重排前张量，则转换 FP16 会引入额外舍入，须对比板端输出。转换器若已分解掉该重排，应回到分解前源码或使用现有转换流程，工具不臆测厂家变换。

量化校准用 `calibration/*_train_*.npz` 或部署端独立训练区样本；`input_vectors/` 为质量测试向量。校准元数据记录训练区、帧号、读取模块及补齐方式，共每场景四组，属于初始小样本。

下一步需要部署端回传：完整同步模型耗时（50 次预热 / 600 次）、逐算子耗时、CPU 回落数量、编译与量化配置、输入输出类型及步幅、整图浮点输出。新卷积和反卷积的映射、六倍重排分解及格式变换成本决定后续结构修改方向。RAW 读取、NPU 交互耗时不计入目标。
