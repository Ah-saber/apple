# 归一化、权重与ONNX约定

## 输入归一化

逐帧使用 `x=(RAW.astype(float32)-offset_t)/scale_segment`，不裁剪输入。RAW是已解码、无文件头、小端uint16灰度图，1024×1280。不要直接把含左右拼接或设备头的原始文件交给模型。

算法相同，数值参数随序列、分段和帧变化。`normalization/light_medium.json` 用于轻中度；`normalization/cross_scene.json` 用于其余三组。校验值见MODEL_MANIFEST.json。样例 `metadata.json` 的参数仅用于该样例，不能复制到其他视频。参数由RAW统计得到，不使用GT；推理无需GT。

新视频采用 `source/src/ir_sr/sequence_normalization.py::calibrate_sequence` 的同一离线算法：按既有规则切段；每帧间隔8采样求中位数；减去帧中心和段参考后的残差限制在[-4,4]内，估计公共偏移；根据段平均RAW的0.1%/99.9%分位数确定范围，最小跨度20。最终逐帧offset和段内scale以源码运算为准。

```bash
python tools/deploy_io.py calibrate-sequence --sequence complete_raw_sequence.npy --output work/sequence_parameters.json
python tools/deploy_io.py normalize --raw frame_0037.u16 --metadata work/sequence_parameters.json --frame 37 --output work/frame37.f32
```

序列数组必须是uint16 `[T,1024,1280]`。当前算法依赖未来帧，属于离线方案；实时因果归一化未实现。直接改成单帧min/max、固定范围或实时滑动统计会改变输入分布，需要重新验证。

## 模型与权重

四组均为16通道、2块主干、4倍入口打包、1×1末端卷积、分级2→2→3像素重排，净放大3倍。主干使用ReLU且无LayerNorm，参考小网络仍含GELU；不能将整图描述成仅有ReLU。RAW缩略图及参考分支均在ONNX内。

每组 `training_checkpoint.pt` 包含训练参数及状态，`static_fused_weights.pt` 为部署结构融合权重，两者的加载结构不同。部署端优先使用已核验ONNX，身份及步数见MODEL_MANIFEST.json。白天辅助RAW权重为0，其余0.1；辅助头仅参与训练，推理图中已删除。

## ONNX要求

opset17；静态batch1；唯一输入raw `[1,1,1024,1280]`，唯一输出display `[1,1,3072,3840]`。图内采用分级均值池化及完整参考融合，无GridSample、LayerNormalization和训练辅助头。算子/形状见每组 `operator_inventory.json`；卷积量及参数量见 `complexity.json`。

重点核对SpaceToDepth、三次DepthToSpace、Resize、AveragePool和GELU对应Erf的转换。禁止重复置换已经重排好的输出卷积通道。若编译器要求图改写，保存新ONNX身份并重新做CPU、板端数值核验。

独立重导出（无需数据集或GT）：

```bash
python tools/reexport.py --group heavy --output work/reexport_heavy --input work/inputs/samples/weather_heavy.f32
```

需PyTorch、ONNX、ONNX Runtime及NumPy。服务器已核验环境为原restormer，PyTorch2.7.0+cu128、ONNX1.17.0、ONNX Runtime1.20.1；ONNX依赖由既有目录提供，没有新安装。完整版本以 `reports/environment.json` 为准。工具验证训练检查点SHA，再导出、检查ONNX和CPU误差。导出器版本不同可能改变图字节及SHA，不应通过复制旧校验值掩盖差异。

## 输出及指标

模型输出为未截断FP32；显示时执行 `np.rint(np.clip(y,0,1)*255).astype(np.uint8)`。数值对比应在这一步之前进行。

训练报告中的整图指标使用三倍输出截断后平均回缩，与原尺寸GT比较；当前没有真实三倍分辨率GT。C32/特殊夜间的验证和测试来自空间留出区域；归一化和参考未跨区域取样。四组固定整图ONNX可供板端测速，但这两类的全帧画质、量化和长时稳定性仍需部署端独立核实。
