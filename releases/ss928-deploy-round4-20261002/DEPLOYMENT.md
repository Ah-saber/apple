# 部署试测步骤

## 1. 核对交付与真实数据

先运行 `runtime/verify_delivery.py`，校验模型、脚本和清单。`MODEL_INDEX.json` 的路径相对于本部署目录，`control` 是第三轮同权重控制，`candidate` 是第四轮八块参考采样与普通输出复制候选。两张图都尚未通过SDK编译或板测；可运行的既有板端控制也继续保留。

复用原真实数据包解压目录，记为 `data_v13`，其下面应直接包含 `deployment_corrected` 和 `night_calibration_final`。五类在 `deployment_corrected/<scene>/`，夜间在 `night_calibration_final/<scene>/`。每类 `train_00`～`train_03.npz` 是四份训练校准；`test_00`～`test_11.npz` 是十二份准备测试输入。

上一v0.13实际使用了旧诊断输入，本次须纠正。用四份训练校准生成编译器校准输入；准备测试输入不作为训练校准。当前输入工具会拒绝非清单源NPZ、错误SHA、非因果历史或重复校准历史。

所有图有两个输入，顺序是 `nine_raw`、`reference_thumb`，布局NCHW、字节序小端。**两输入精度逐图读取，不统一强制FP16：特殊夜间RAW可能为FP32，参考为FP16。** 图声明、BIN字节数及哈希都会写入输入metadata。

## 2. 为确切模型生成BIN和PC参考

以下命令从仓库根目录运行，使用原test环境。示例为轻天气候选；按索引换场景和角色，不修改脚本。输出目录必须不存在，避免覆盖。

```bash
python releases/ss928-deploy-round4-20261002/runtime/prepare_sample.py \
  --data-root data_v13 --scene weather_light --role candidate \
  --sample train_00.npz --out board_inputs/weather_light/candidate/train_00

python releases/ss928-deploy-round4-20261002/runtime/prepare_sample.py \
  --data-root data_v13 --scene weather_light --role candidate \
  --sample test_00.npz --out board_inputs/weather_light/candidate/test_00

python releases/ss928-deploy-round4-20261002/runtime/regenerate_reference.py \
  --model releases/ss928-deploy-round4-20261002/models/weather_light/candidate.onnx \
  --sample data_v13/deployment_corrected/weather_light/test_00.npz \
  --out pc/weather_light_candidate_test_00.npz
```

校准依次生成 `train_00`～`train_03`；测试准备 `test_00`～`test_11`。控制图也按 `--role control` 和 `control.onnx` 独立生成、记录图与输入身份。PC参考必须由确切新图生成，不能直接把NPZ中旧PC输出作为本轮图的参考。

原归一化、参考内容和九帧历史已在源NPZ中准备好，输入工具只校验和按模型精度转换，不重新估计归一化或重复当前帧。

## 3. 编译与单次完整输出核对

使用部署端既有SDK编译入口，以四份真实训练输入分别编译同场景控制和候选。保留原编译器版本、量化设置和原已可运行模型，输出新的独立编译目录。

记录各图编译日志、编译后模型SHA、输入类型/字节数、量化配置、算子实际执行位置及CPU回退。新图最多五维重排，避开旧大参考Gather、Expand及Resize；静态检查不保证SDK支持。编译失败先保留日志及原图，不自行近似坐标或改输出接口后继续使用原哈希。

先以一份真实测试输入运行一次，确认输出为完整 `1×1×3072×3840` Float16，即**23,592,960字节**原始输出，不含SDK头。此阶段保持模型浮点接口；图外转为8位单独评估。

将板端原始浮点输出记为 `board/weather_light_candidate_test_00.bin`，再核对同图、同输入的PC结果：

```bash
python releases/ss928-deploy-round4-20261002/runtime/compare_output.py \
  --input-metadata board_inputs/weather_light/candidate/test_00/metadata.json \
  --reference pc/weather_light_candidate_test_00.npz \
  --board-output board/weather_light_candidate_test_00.bin \
  --out comparisons/weather_light_candidate_test_00.json
```

比对工具校验PC与输入记录的图/源样本哈希、BIN哈希和完整输出字节数，报告原始灰度平均绝对差、最大差、亮度偏移，并另列裁到0～255再截断的字节差。量化误差须实报；本轮直接运行与PC后端最大0.625灰度不能作为量化必过阈值。数值通过也不能代替GT画质或耗时验证。

## 4. 同条件完整计时

优先先测日间和轻天气各一对控制/候选，确认SDK支持和计时口径，再继续其他五类。每图至少50次预热、1000次正式完整调用；两图使用同输入、精度、编译设置和输出接口，保存每次耗时并报告均值/P95。

分开记录完整纯NPU时间、SDK同步时间、输入读取/传输、图外输出转换。目标只用完整纯NPU均值/P95均≤16.7毫秒判断；没有纯NPU读数时明确标为SDK同步时间，不能换算。

同时记录逐算子时间、CPU回退、调用失败、故障前后硬件错误计数。上一普通夜间拖影图曾触发500004；本次普通夜间是保留原权重的结构图，仍需同条件验证。若再次出现实际驱动失败或硬件错误计数变化，停止该组，保留输入、图、调用位置和日志，由板卡侧处理；此文档不规定重启或固件操作。

## 5. 连续量化画质

十二份准备输入用于数值和稳定性核对，重复输入无法证明拖影改善。画质需沿原数据包manifest的预处理和归一化规则重放连续RAW，使用同帧GT：普通夜间60帧，其他各120帧。保持因果历史，不引入未来帧。

逐帧保存候选和控制的完整浮点输出/可视图，比较小目标当前位置、旧位置残留、对比、建筑轮廓、弱结构、静态变化和亮度偏移。特殊夜间GT仅右侧192像素有效；C32有效GT不覆盖已选天气道路，不能对有效区外给道路小目标改善结论。

本次两张同权重结构图在服务器直接运行输出一致，预期任务是验证板端计算布局及量化响应，不能提前宣称拖影改善。

## 6. 回传问题和结果

每组给出场景、角色、源ONNX/编译图/校准/测试输入哈希、SDK版本与配置、执行位置、预热/正式次数、逐次完整耗时、均值/P95、错误计数、PC比对JSON，以及同场景同帧GT连续输出。失败项保留完整日志和输入，避免只报告成功均值。

本轮Git推送只提供模型和工具；训练服务器上的待审新增实验不属于这次部署目录。
