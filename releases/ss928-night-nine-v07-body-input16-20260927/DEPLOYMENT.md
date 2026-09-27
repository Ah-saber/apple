# 部署对照与输入约定

所有文件都是源码图；尚未得到SS928 SDK转换成功、任务划分、精度或测速结果。已有native R2图无法传回，工具允许部署端保留其已验证处理并仅替换主体。

## 对照顺序

1. 对照原设备版本及此次源父版 `ordinary/special_combo_input9_half_aligned16_nearest.onnx`。
2. 保留四层的 `*_input16_nhwc.onnx` 和 `*_input16_native5.onnx`：独立测试声明和格式。九真实帧＋七零槽、Half输入，参考Float32，完整Half输出。SDK不支持五维时记录错误并用四维/原接口。
3. `*_body3_quantsearch_npu.onnx`、`*_body2_quantsearch_npu.onnx`：原九槽Float32接口、32通道正负统计、单固定反卷积、完整Half输出；主体已校准。三层优先核对画质，两层关注低幅运动细节。
4. `*_input16_nhwc_body3.onnx`、`*_input16_native5_body3.onnx`：组合。直接比较同输入/输出约定，不能把模型外反卷积或参考计算排除计时。

每场景上述9类还包含同父接口三/两层，均提供 `_small` 对照。图内标有实际输入/输出shape和dtype，以图声明为准，不沿用旧输入字节解释。输出Half灰度范围0～255，输出Float32方案同范围；读出时先按声明类型解释，转换需纳入设备完整模型时间。

## 生成16槽输入

在test环境numpy可用时：

```sh
python build_input_vectors.py --source /absolute/path/ordinary_frame_20.npz --output /absolute/path/new-vector --layout nhwc16
# 五维试验使用 --layout native5；输出目录须不存在
```

NPZ字段沿用`nine_raw`与`reference_thumb`，normalized输入按原九帧順序，不重新归一化。输入f16数据为连续NHWC16（五维增加单长度轴）、参考f32连续数组。manifest包含shape、字节数与SHA256。此工具仅改变输入表示，未预计算统计或参考分支。

## 保留设备原图的主体移植

需部署端实际原R2图，并与本交付父版主体系数一致；工具按Half系数匹配四个16→16、3×3、ReLU串联层，须唯一匹配、串联独占，匹配失败明确报错。

```sh
python tools/transplant_body.py --native /absolute/path/native_r2.onnx --reference source_onnx/ordinary_combo_input9_half_aligned16_nearest.onnx --candidate source_onnx/ordinary_body3_quantsearch.onnx --output /absolute/path/native_r2_body3.onnx
```

特殊场景更换special文件，两层更换body2。工具保留原主体输入/输出及所有其他节点；保留native系数dtype，数值与SDK激活量化仍需检查。已用实际权重的隔离主体图验证Half/Float32、三种尺寸全部像素一致及修改原系数拒绝例。真实native_r2尚未验证。

请回传：SDK兼容错误或成功记录、输出shape/dtype、单帧完整推理和各任务耗时、旧/新同帧输出校验与完整视频。宿主读取/传输不纳入考核；模型统计、参考、完整输出展开、NPU格式转换及Report转换需纳入。没有新板端结果前，16.7ms仍未验证。
