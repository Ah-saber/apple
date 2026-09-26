# 夜间九帧轻量轨迹候选

本轮针对SS928报告约129.75 ms的轨迹前端，训练轻量卷积模块拟合原四通道轨迹。优先测试 `compact8`，`wide12` 是质量变化较小的对照。目标是完整模型推理≤16.7 ms；目前仅确认GPU收益，SS928编译、板端速度和量化后画质未测。

本目录追加于apple原部署分支 `codex/ss928-night-nine-factor24-20260926`，父提交 `09f92b08ce738759989add5c782fea380c74425b`。原版本保存在相邻目录 `ss928-night-nine-factor24-20260926`，可按提交回溯。

## 已测结果

RTX5090，预热30次，三轮各200次同步GPU计时。相同输入输出接口逐组比较：

| 接口 | 场景 | 原模型 ms | compact8 ms | 提速 |
|---|---|---:|---:|---:|
| FP16九帧输入、整图uint8输出（上一版GPU接口） | 普通 | 0.760639 | 0.690921 | 9.17% |
| 同上 | 特殊夜间 | 0.733081 | 0.655272 | 10.61% |
| FP32九帧输入、整图FP32灰度输出（本轮ONNX接口） | 普通 | 0.800612 | 0.708261 | 11.54% |
| 同上 | 特殊夜间 | 0.778369 | 0.674197 | 13.38% |

不含编译、RAW读取/预处理、主机与设备拷贝。GPU收益不能按比例换算NPU收益。

完整普通60帧、特殊120帧回归：compact8的PSNR相对原模型分别−0.0017/−0.0059 dB；静止区域输出波动分别约+0.93%/+0.73%；弱运动区域误差约−0.05%/+0.07%。局部亮度及对比度波动有小幅变化，详见[全部指标和限制](evidence/RESULTS.md)。特殊夜间质量指标覆盖右侧192像素宽的留出区；完整视频展示整张画面。弱运动指标是GT时间变化代理，没有人工目标标注。

## 查看完整视频

- [普通夜间 compact8，60帧](videos/ordinary_compact8_full.mp4)
- [特殊夜间 compact8，120帧](videos/special_compact8_full.mp4)
- [普通夜间 wide12](videos/ordinary_wide12_full.mp4)
- [特殊夜间 wide12](videos/special_wide12_full.mp4)

四栏：RAW、同帧GT、当前模型、候选。每栏保留1024×1280完整传感器画面；3倍模型输出用面积平均回缩展示。视频为GPU模型输出，不能当板端视频。5120×1080，12fps为展示速度。

## 模型和接口

`models/{ordinary,special}_trajectory_student_float32_compact8.onnx` 为优先候选；对应 `wide12` 文件作对照。四图都通过ONNX检查，尚未应用部署端实际R4兼容改写。所有轨迹计算、去噪、色调映射和输出放大都在模型图内。

| 名称 | 类型、形状 | 有效字节数 |
|---|---|---:|
| nine_raw | FP32 `[1,9,1024,1280]` | 47,185,920 |
| reference_thumb | FP32 `[1,1,64,64]` | 16,384 |
| display_gray | FP32 `[1,1,3072,3840]`，0～255灰度 | 47,185,920 |

输出没有舍入为uint8；显示时舍入、裁剪到0～255。预处理与原九帧版本一致，代码及固定参数在 `preprocess/`；所附样例沿用原输入。`models/` 含原冻结权重、两种新模块权重；`runtime/` 可独立加载，无需训练工程。训练仅更新新模块，普通/特殊分别2000步，验证集分别选第2000/1500步。训练输入仅来自原训练区域，未读取GT。新的学习模块无法保证与原速度搜索逐项等价。

## 接入部署端R4

优先在实际已验证的R4 ONNX上替换前端，保留后续兼容改写及输出接口。工具要求原门控卷积与候选的冻结权重、偏置、补零和步长约定吻合；不吻合会拒绝处理。

```bash
python tools/splice_r4_front.py \
  --r4 /path/to/native_r4_float32.onnx \
  --student models/special_trajectory_student_float32_compact8.onnx \
  --output /path/to/special_r4_student8.onnx
```

该工具保留实际R4原输入输出名称、类型和尺寸；若R4仍用 `display_uint8` 命名FP32输出，名称保留。新增前端使用半精度卷积参数，在原门控卷积输入处匹配其类型。必须确认本SDK支持新 `ConvTranspose`（核8、步长4）、类型转换及时间均值；编译后检查是否出现CPU回退。若不支持，继续调整前端表达并重新核对结果。

我们没有收到实际R4图，只在冻结源图控制上测试了替换。独立ONNX参考执行器的小尺寸数值检查最多差1灰度，见 `evidence/compact8/splice_source_control_verification.json`。该检查不能确认实际R4编译或板端速度。也可使用完整源ONNX并应用部署端既有兼容改写。

校准新模型后，按原50次预热、600次同步完整模型调用记录均值及P95，并保存独立NPU任务分解。RAW读取、预处理、拷贝、缓存维护及显示转换不纳入本轮速度目标。原报告的约216 ms是对照，不能用缩小图边界后的时间替代完整模型时间。所附两幅输入用于数值检查，不能视为充分独立的量化校准集合。

## 源模型复现与板端输出检查

在有Torch/CUDA的test环境、此目录下运行；ONNX前端替换工具还需要onnx。

```bash
python tools/infer_vector.py --scene ordinary --variant compact8 \
  --input input_vectors/ordinary_frame_20.npz --output /tmp/ordinary_student.f32
python tools/compare_output.py \
  --reference reference_outputs/ordinary_frame_20_student.npz \
  --output /tmp/ordinary_student.f32
```

特殊夜间用 `special_frame_60.npz`。板端结果可由相同比较工具读取；有行补齐时传 `--row-stride-bytes`。源模型独立加载验证中，普通逐值一致，特殊最大约0.1245灰度、舍入后74个像素最多差1灰度，受卷积自动选算法影响。该比较针对同版本源输出，GT画质指标另见回归证据。

`reproduce/` 保留训练、验证、导出和三轮计时脚本，依赖记录中的服务器数据、划分与原检查点；训练数据不在此目录。全部文件校验见 `MANIFEST.sha256`，推理权重来源见 `runtime/PROVENANCE.json`。
