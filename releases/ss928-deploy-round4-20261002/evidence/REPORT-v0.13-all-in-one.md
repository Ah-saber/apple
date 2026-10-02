# SS928 v0.13 新版部署测试完整合并报告

实验：EXP-20261001-014；测试日期：2026-10-01。状态：**部分完成，驱动故障后停止；不是生产验收通过**。

本文件合并本轮分析、全部已测时间、独立NPU逐任务、85次转换兼容记录和证据核验结论。正文和附表可单独阅读，无需另开分散时间文档。原始日志、模型及CSV仍保留在项目中，不嵌入庞大的二进制数据。合并没有追加模型测试或改变原结论。

- [结果分析与限制](#结果分析与限制)
- [完整时间统计](#完整时间统计)
- [独立NPU模块与逐任务](#独立npu模块与逐任务)
- [全部转换及兼容记录](#全部转换及兼容记录)
- [核验与实验收口](#核验与实验收口)

## 结果分析与限制

2026-10-01；EXP-20261001-014；研究快照，**未完成全部验收，未替换生产模型**。

### 结论

固定公开分支 codex/ss928-night-nine-factor24-20260926 提交 `946305bbfda9c832ddc95a5ac789567845040abe`（包含44dd595）。两份部署清单共951项载荷SHA通过，优先21图。本轮完成18组50＋1000同步（17修复候选＋1日间源图），18000成功正式调用；另有短诊断和失败前103次，不混入完成组数。旧模型未重测。

天气运动版比本轮修复控制快约12%，但仍慢于上一版冻结控制。日间运动残差补偿版慢约63%，不能认为全场景进步。普通夜间布局回退版约24.683ms，与历史24.632ms基本持平；特殊夜间正式测试未完成。所有这些仍未达到16.7ms目标。

**普通夜间运动版发生新驱动/硬件错误，已停止所有后续NPU推理。** 无重启、不停业务、不改固件/系统库/频率；故障族隔离，不自动重试、不把安全门限0/1/1改成0/2/1继续运行。

### 速度及历史对比

单位ms，同步execute（含SDK同步边界）不是独立纯NPU。每组50预热＋1000正式。历史五场景为v0.12不同F32接口，当前F16；只作实际版本结果参照，不宣称单因素结构净收益。

|场景|本轮控制mean/P95|布局修复mean|运动mean/P95|运动较本轮控制耗时|历史控制mean|运动较历史耗时|
|---|---:|---:|---:|---:|---:|---:|
|日间|64.144/66.185|64.130|104.606/106.431|+63.08%|59.876|+74.70%|
|轻天气|49.564/51.443|49.547|43.729/45.705|-11.77%|40.210|+8.75%|
|中天气|49.541/51.481|49.441|43.512/45.398|-12.17%|40.253|+8.10%|
|重天气|49.431/51.306|49.362|43.551/45.578|-11.89%|40.168|+8.42%|
|重天气C32|49.470/51.437|49.497|43.601/45.566|-11.86%|39.097|+11.52%|

普通夜间c02_r3：24.682718ms/P95 25.841391ms，已准备输入→8bit内存87.461125ms。历史同步24.632ms，约+0.21%（无显著收益结论）。特殊夜间只有本轮低维Expand短诊断，不提供本轮正式均值。

日间未改源图完整50＋1000：396.909913ms/P95 398.687694ms，内存E2E460.246595ms。R3控制64.144ms相对源图大幅下降来自本地固定Gather/布局适配，不是源布局在板上直接获得GPU同幅提速。

### 时间边界和模块瓶颈

[完整时间统计](#完整时间统计)，[独立NPU模块与逐任务](#独立npu模块与逐任务)。

同步execute不含本轮独立统计的输入拷贝、输入flush、输出flush、打包与8bit转换。模型内部入口处理、参考采样、卷积、输出布局和Report包含在execute中。外围已准备输入→8bit还不包含RAW归一化等，因此不能称RAW端到端；本轮没有新RAW预处理探针。原始输出完整23,592,960字节，打包和8bit转换不能省略后仍宣称同一完整输出接口。

独立NPU仅日间未改源图1＋2短profile：398.838575ms，两个横向参考Gather合计341.996645ms（85.75%）；每个约120MiB读和120MiB写。此为大张量索引访存瓶颈证据，不足证明DDR已饱和。主卷积联合任务约17.196ms、参考加尾部约5.565ms、Report约4.490ms，其他精确融合边界见逐任务表。没有给R3每模块虚构分解，也不能把它的64ms减去源图341ms。

低维Expand短诊断：日间约215.963ms，普通/特殊夜间约174.835/171.774ms；短测不能作为正式P95。R3采用rows32完整输出回退，两类日间/天气控制及布局修复结果几乎相同，说明本轮源GPU布局收益未保留到板端。天气三帧body确有同步时间收益，日间新增残差路径则明显更贵。

### 算子兼容与每次本地修复

[全部转换及兼容记录](#全部转换及兼容记录)。共85次转换记录，失败均保留。

- 源七张布局及四张天气运动图：八维Reshape/Transpose/Expand超过当前最多七维限制；不是所有源算子都直接支持。日间/夜间运动首个失败是常量bias别名被在线编译当动态，修复后才检查后续限制。
- 夜间原图Resize编译回退CPU（普通3处，特殊1处）；五场景原图Gather支持NPU但可能极慢。没有凭500004认定CPU回退：失败候选编译CPU列表为空，runtime AICPU任务数0。
- R1：八维降六维及夜间Resize固定相位卷积；R2：两个固定横向Gather→Transpose＋one-hot 1×1 Conv；R3：完整rows32输出＋常量别名实化。五场景当前PC回归逐位一致，夜间Resize运算顺序变化最大0.125灰度、字节最多差1，不称完全无损。
- R4：十个五场景原/运动全参考轴加权Conv候选，已转换无CPU；PC最大0.125灰度/字节≤1，**未板测**。R5精确周期映射证明失败（C32严重不满足），十项拒绝记录，不生成/部署近似图。
- 本地控制问题：R1泛化匹配误把夜间原rows32改成Expand，R3 c01沿用了它。普通c01_r3虽完成175.031ms/P95 176.797ms，却不是有效原控制，不能据此说夜间运动版提速72%或源原模型慢175ms。R6两个夜间原始rows32控制已纠正、PC回归并编译无CPU，但故障后**未板测**。历史失败和误改结果不覆盖。

### 驱动故障与稳定性

普通夜间c03_r3 OM SHA `a659ff22cf7025ae1a86b28979b9edb52ca0d1d5b8536b8c847d5be6d5320e52`；50预热＋103正式成功后execute报500004（SDK定义SVP_ACL_ERROR_DRV_FAILURE）。成功部分仅失败前描述，非1000次完成结果；无最终输出，不能报告该候选板端PSNR。

驱动累计timeout/hw/AICPU由0/1/1变成0/2/1。内核uptime590345.881980记录SVP_NPU error并dump四行task header；尚无足以定位源算子的符号证据。SDK日志出现AICPU task获取失败，但该模型AICPU=0，不能据日志措辞推断是某个CPU算子不支持。last_task_node_id220也不能直接对应ONNX第220层。

已回收部分CSV、SDK日志、前后快照、kernel log；故障板端临时模型/输入/参考/CSV均在主机SHA确认归档后清理，可由主机制品恢复。板端boot未变，业务PID13547/13833存在，own runner退出，NPU模型/stream资源释放、send=finish，MMZ恢复1590548KiB。hw_status0仅表示观察时空闲，不构成硬件故障后可安全再跑的证明。所有后续推理/profile/真实序列停测。

证据：`artifacts/EXP-20261001-014/night_ordinary/board_c03_r3_m6_steady/`；[故障后只读快照](../../records/iterations/v0.13/board-final-after-failure.json)、[kernel log](../../records/iterations/v0.13/kernel-log-after-motion-failure.json)。

### 对PC的一致性，不是GT画质

本轮同版本ONNX/ORT PC参考，对声明的同一历史诊断输入、完整F16输出统计。每场景固定一份输入，不能把重复计时当质量数据；数值偏差包含量化/执行精度和兼容改写影响。

|场景/候选|PSNR-to-PC dB|MAE灰度|bias灰度|最大误差|SSIM-to-PC|
|---|---:|---:|---:|---:|---:|
|日间/c01_r3|36.229603|3.549213|-3.516228|17.750000|0.99594379|
|日间/c02_r3|36.229603|3.549213|-3.516228|17.750000|0.99594379|
|日间/c03_r3|37.421390|3.028689|-2.965533|28.875000|0.99379180|
|普通夜间/c01_r3|42.313256|1.606508|0.306685|10.000000|0.98046971|
|普通夜间/c02_r3|42.325221|1.603669|0.298894|10.000000|0.98047115|
|重天气/c01_r3|34.426794|4.388925|-4.388652|11.375000|0.97869632|
|重天气/c02_r3|34.426794|4.388925|-4.388652|11.375000|0.97869632|
|重天气/c03_r3|35.703498|3.743469|-3.741830|10.250000|0.97898708|
|重天气C32/c01_r3|35.512660|3.653300|-3.604274|11.312500|0.97377406|
|重天气C32/c02_r3|35.512660|3.653300|-3.604274|11.312500|0.97377406|
|重天气C32/c03_r3|35.609249|3.596633|-3.515555|11.406250|0.97403022|
|轻天气/c01_r3|30.533394|6.910516|6.682784|17.625000|0.88325948|
|轻天气/c02_r3|30.533394|6.910516|6.682784|17.625000|0.88325948|
|轻天气/c03_r3|30.329776|7.063691|6.840768|18.500000|0.87912781|
|中天气/c01_r3|30.078283|7.276390|7.007095|18.375000|0.87182211|
|中天气/c02_r3|30.078283|7.276390|7.007095|18.375000|0.87182211|
|中天气/c03_r3|29.883367|7.430839|7.164843|19.375000|0.86754632|

五场景c01/c02板端输出相同。轻/中天气明显偏亮（约+6.7/+7.0灰度），日间/重天气/C32偏暗，仍需真实校准和逐层误差定位。对各自PC的PSNR变动不能证明新网络对GT画质提升；旧新PC dtype/图不同，禁止直接当GT质量排名。

上游服务器报告原→motion GT PSNR：日间25.4843→25.7924、轻34.3061→34.3157、中33.7781→33.7882、重30.2750→30.2808、C32 23.8342→23.8296、普通夜间27.8682→27.8740、特殊25.9193→25.9101dB。**仅交付文档引用，非本轮板端实测**；特殊/C32有退步，不能只看快。C32和特殊GT仅右192px有效；目标GT辅助连通域不是独立人工逐目标盲测。

### 缺失数据、未完成项与建议

真实校准/GT包 `SS928-V13-REAL-CALIBRATION-AND-GT-20260930.tar`，8,420,392,960字节，SHA `d748b9582cca5aab1cd85760c866e351de8925ff916590515cf7b9be74483a59`，当前只给服务器/data路径，无已找到的公开下载/本地副本。已询问用户路径或URL，尚未获得。当前校准/输入来自冻结旧交付，不能冒充新版真实九帧或GT验收。

未完成：普通运动1000次、全部特殊夜间R3正式、两张纠正R6控制、R4板测、R3独立模块profile、真实120/60帧时序及GT、RAW全链与长期业务共存验收。本轮无新生产发布。

后续优先级：先定位/评审500004硬件任务错误及安全恢复条件（不自动重启或试错），拿到新版真实数据后重做校准/数值验证；天气保留三帧body研究但对旧基线仍未更快，日间残差补偿不作为速度首选；固定采样和平台输出布局比仅看MAC/GPU更关键。精确固定加权采样R4仅是待测候选，R5不满足等价证明应拒绝。完整输出若改成native后CPU放大需要另外定义接口和实测全链，不能通过把耗时挪到预处理口径宣称真正提速。

历史报告哈希复验；本地v0.13独立留档。未创建或提交GitHub Issue。

## 完整时间统计

每项50预热＋1000正式；1280×1024九帧及参考已预处理，完整3072×3840 FLOAT16输出。重复固定输入不是1000质量样本。单位ms。

|场景/候选|同步均值|P50|P95|最大|已准备输入→8bit内存|
|---|---:|---:|---:|---:|---:|
|日间/c01_r3|64.143504|64.084021|66.184731|69.922333|127.835241|
|日间/c01_source|396.909913|396.926854|398.687694|401.558167|460.246595|
|日间/c02_r3|64.130272|64.147562|66.059631|68.050750|127.142267|
|日间/c03_r3|104.606099|104.523812|106.430908|113.713334|167.638222|
|普通夜间/c01_r3（本地误改Expand，非原始控制）|175.031393|174.955730|176.796657|180.756417|239.396323|
|普通夜间/c02_r3|24.682718|24.645604|25.841391|31.958917|87.461125|
|重天气/c01_r3|49.430930|49.290500|51.306021|53.827625|113.183100|
|重天气/c02_r3|49.362466|49.281708|51.232823|54.456000|113.767878|
|重天气/c03_r3|43.551402|43.436771|45.578117|49.381958|107.298073|
|重天气C32/c01_r3|49.470401|49.385646|51.437269|56.827458|112.590898|
|重天气C32/c02_r3|49.497400|49.374063|51.444960|62.948958|113.669054|
|重天气C32/c03_r3|43.600960|43.500625|45.566194|55.266000|107.178847|
|轻天气/c01_r3|49.564034|49.431459|51.443477|58.420583|112.051569|
|轻天气/c02_r3|49.546868|49.407145|51.419557|57.732916|114.011948|
|轻天气/c03_r3|43.729227|43.612437|45.704865|49.017334|106.280814|
|中天气/c01_r3|49.541314|49.405500|51.481079|55.556500|113.696264|
|中天气/c02_r3|49.441350|49.325520|51.385367|55.123125|113.426578|
|中天气/c03_r3|43.511621|43.447333|45.398188|46.654292|106.945339|

### 同次正式运行外围分项均值

不含RAW归一化、动态校正、缩略图制作、真实历史环形缓冲、主机传输、文件保存。

|场景/候选|输入拷贝|输入flush|同步execute|输出flush|输出打包拷贝|8bit转换|内存E2E|
|---|---:|---:|---:|---:|---:|---:|---:|
|日间/c01_r3|10.870750|1.853629|64.143504|1.722906|11.222424|38.016253|127.835241|
|日间/c01_source|10.918414|1.859356|396.909913|1.692522|11.164347|37.697339|460.246595|
|日间/c02_r3|10.815102|1.875702|64.130272|1.698264|11.149950|37.468141|127.142267|
|日间/c03_r3|10.805644|1.862315|104.606099|1.714559|11.126600|37.517599|167.638222|
|普通夜间/c01_r3（本地误改Expand，非原始控制）|11.125213|1.917942|175.031393|1.723590|11.276687|38.315794|239.396323|
|普通夜间/c02_r3|10.792803|1.882202|24.682718|1.691928|10.942669|37.463690|87.461125|
|重天气/c01_r3|10.956784|1.867595|49.430930|1.704139|11.211638|38.007059|113.183100|
|重天气/c02_r3|11.072159|1.903081|49.362466|1.743800|11.291273|38.389929|113.767878|
|重天气/c03_r3|10.953921|1.860566|43.551402|1.719565|11.248329|37.959292|107.298073|
|重天气C32/c01_r3|10.862681|1.859745|49.470401|1.708533|11.129391|37.555409|112.590898|
|重天气C32/c02_r3|11.063472|1.925645|49.497400|1.721788|11.286713|38.168856|113.669054|
|重天气C32/c03_r3|10.964789|1.872500|43.600960|1.752687|11.222113|37.760934|107.178847|
|轻天气/c01_r3|10.723026|1.820011|49.564034|1.686835|11.017463|37.235127|112.051569|
|轻天气/c02_r3|11.126701|1.892998|49.546868|1.731877|11.310918|38.396990|114.011948|
|轻天气/c03_r3|10.768242|1.860663|43.729227|1.690212|11.058233|37.169561|106.280814|
|中天气/c01_r3|11.095943|1.905894|49.541314|1.723012|11.276622|38.147398|113.696264|
|中天气/c02_r3|11.008183|1.873131|49.441350|1.734525|11.285703|38.078787|113.426578|
|中天气/c03_r3|10.933364|1.848479|43.511621|1.739307|11.178404|37.727999|106.945339|

### 单次启动与窗口

启动/读盘/加载/分配/首次调用/保存不加入上述稳态E2E；五段各200次仅为同次重复输入窗口。

|场景/候选|init|输入读盘校验|load|分配|first|save|五段同步均值|
|---|---:|---:|---:|---:|---:|---:|---|
|日间/c01_r3|2.546667|97.058833|6.395167|31.309250|63.077250|130.671292|64.118200, 64.105876, 64.129815, 64.095987, 64.267640|
|日间/c01_source|1.729750|122.234083|6.390167|39.121417|398.543126|133.451791|396.939438, 396.920142, 396.913330, 396.918807, 396.857845|
|日间/c02_r3|4.462375|123.711666|9.495709|41.912166|66.670250|133.075501|64.081489, 64.145708, 64.078070, 64.168069, 64.178026|
|日间/c03_r3|1.912500|102.729791|8.236333|32.944708|104.132916|132.848959|104.656469, 104.441582, 104.494005, 104.727764, 104.710675|
|普通夜间/c01_r3（本地误改Expand，非原始控制）|3.702417|93.962292|5.495750|45.785875|174.076083|128.869041|174.972751, 174.946810, 174.937104, 174.978427, 175.321872|
|普通夜间/c02_r3|2.741875|94.320916|5.552000|31.742042|24.813209|130.511709|24.729610, 24.646675, 24.715831, 24.624761, 24.696716|
|重天气/c01_r3|3.455333|94.708250|6.276292|28.463459|50.045000|167.946709|49.435105, 49.318233, 49.371876, 49.547446, 49.481993|
|重天气/c02_r3|3.639166|93.209541|6.291417|29.127708|47.865917|143.504375|49.281309, 49.368384, 49.378350, 49.354604, 49.429685|
|重天气/c03_r3|2.585625|120.497834|7.290208|39.111167|44.858000|130.507917|43.555768, 43.589411, 43.614982, 43.488465, 43.508381|
|重天气C32/c01_r3|3.592959|96.048750|6.276250|28.944916|50.772333|129.920000|49.568405, 49.428897, 49.407761, 49.497682, 49.449262|
|重天气C32/c02_r3|3.777125|95.277125|6.819625|28.506209|47.947750|168.610625|49.467371, 49.420139, 49.554876, 49.507057, 49.537557|
|重天气C32/c03_r3|1.194625|99.547333|6.235250|32.194875|42.311416|130.850375|43.617611, 43.724654, 43.699207, 43.504076, 43.459253|
|轻天气/c01_r3|3.588083|96.188042|7.274000|29.082042|51.195458|132.348251|49.530007, 49.551011, 49.545289, 49.533901, 49.659961|
|轻天气/c02_r3|4.395917|121.331125|9.892625|36.408459|50.672458|131.345833|49.488404, 49.628464, 49.496045, 49.573767, 49.547660|
|轻天气/c03_r3|3.511542|94.599625|7.450208|31.826500|44.208459|129.711125|43.913329, 43.711113, 43.672793, 43.636080, 43.712823|
|中天气/c01_r3|3.372458|120.267625|8.609625|37.247708|49.333292|133.252125|49.458491, 49.609065, 49.524174, 49.502702, 49.612138|
|中天气/c02_r3|3.931334|99.030000|6.928542|28.505500|47.829958|134.846791|49.493781, 49.397072, 49.516553, 49.443846, 49.355495|
|中天气/c03_r3|4.623792|119.846459|7.973292|37.573541|43.214250|168.674083|43.530484, 43.577520, 43.504642, 43.488091, 43.457370|

### 失败运行：普通夜间 c03_r3

50预热＋103成功正式调用后下一次execute报500004；以下只描述失败前已记录样本，不是1000次完成结果。
成功部分同步均值 48.638488ms。无最终output.bin/u8，不能提供该运行质量分数。

## 独立NPU模块与逐任务

本轮仅日间未改源图 c01_source 完成独立1预热＋2正式profile。不得视为1000次稳态，不将同步execute冒充纯NPU。其他候选独立profile因硬件故障未完成。

纯NPU mean/P95=398.838575/399.0449315 ms。

|融合模块|均值ms|占比|
|---|---:|---:|
|图内入口Preprocess|4.045360|1.01%|
|前端/主体卷积（含融合）|17.195785|4.31%|
|其余（逐任务表可追溯）|2.763925|0.69%|
|参考分支/插值|9.326290|2.34%|
|Gather索引采样/形状（实际任务）|347.228145|87.06%|
|参考+融合尾部|5.565425|1.40%|
|完整输出布局/Report（融合边界）|12.713645|3.19%|

|task|节点/融合边界|类型|mean ms|P95 ms|DDR读KiB|DDR写KiB|
|---|---|---|---:|---:|---:|---:|
|0|`nine_raw`|Preprocess|4.038215|4.072735|23045.810|23040.000|
|1|`/front/first/Conv,/body/body.0/Conv,/body/body.2/Conv`|Convolution,Convolution,Convolution|17.195785|17.218604|23099.890|15360.250|
|2|`/Slice`|Crop|0.441430|0.446830|2561.980|2560.250|
|3|`/Conv`|Convolution|1.922140|1.922527|2567.420|2560.250|
|4|`reference_thumb`|Preprocess|0.007145|0.009975|9.360|8.250|
|5|`/reference/encoder/encoder.0/Conv`|Convolution|0.017215|0.022295|2.610|128.250|
|6|`/reference/encoder/encoder.1/Div`|BinaryMath|0.033645|0.045025|129.280|128.250|
|7|`/reference/encoder/encoder.1/Erf`|Erf|0.044575|0.061031|129.050|128.250|
|8|`/reference/encoder/encoder.1/Add`|BinaryMath|0.038215|0.055383|129.160|128.250|
|9|`/reference/encoder/encoder.1/Mul`|BinaryMath|0.056575|0.079718|257.520|128.250|
|10|`/reference/encoder/encoder.2/Conv`|Convolution|0.037215|0.052582|131.480|128.250|
|11|`/reference/encoder/encoder.3/Div`|BinaryMath|0.036285|0.050302|129.280|128.250|
|12|`/reference/encoder/encoder.3/Erf`|Erf|0.034075|0.042369|128.920|128.250|
|13|`/reference/encoder/encoder.3/Add`|BinaryMath|0.035500|0.049576|129.280|128.250|
|14|`/reference/encoder/encoder.3/Mul`|BinaryMath|0.070500|0.104250|257.390|96.250|
|15|`/reference/pyramid.0/pyramid.0.0/AveragePool,/reference/pyramid.0/pyramid.0.1/Conv`|PoolingAve,Convolution|0.014070|0.016446|15.530|32.250|
|16|`/reference/pyramid.0/pyramid.0.2/Div`|BinaryMath|0.007210|0.008560|33.160|32.250|
|17|`/reference/pyramid.0/pyramid.0.2/Erf`|Erf|0.009500|0.010211|33.050|32.250|
|18|`/reference/pyramid.0/pyramid.0.2/Add`|BinaryMath|0.008925|0.011432|33.160|32.250|
|19|`/reference/pyramid.0/pyramid.0.2/Mul`|BinaryMath|0.013645|0.017052|65.520|32.250|
|20|`/reference/pyramid.0/pyramid.0.3/Conv`|Convolution|0.014570|0.020870|35.480|24.250|
|21|`/reference/resizes.0/Gather`|Gather|0.059355|0.079474|57.980|48.250|
|22|`/reference/resizes.0/Mul`|BinaryMath|0.019215|0.019408|52.250|48.250|
|23|`/reference/resizes.0/Gather_1`|Gather|0.061860|0.078447|66.110|48.250|
|24|`/reference/resizes.0/Mul_1`|BinaryMath|0.018785|0.018978|52.380|48.250|
|25|`/reference/resizes.0/Add`|Eltwise|0.015930|0.016254|97.170|48.250|
|26|`/reference/resizes.0/Gather_2`|Gather|0.597785|0.600418|771.230|768.250|
|27|`/reference/resizes.0/Mul_2`|BinaryMath|0.020000|0.020126|97.750|96.250|
|28|`/reference/resizes.0/Gather_3`|Gather|0.776355|0.941375|770.360|768.250|
|29|`/reference/resizes.0/Mul_3`|BinaryMath|0.025430|0.029930|97.750|96.250|
|30|`/reference/resizes.0/Add_1`|Eltwise|0.033000|0.040074|193.170|96.250|
|31|`/reference/Add`|Eltwise|0.028430|0.031004|193.050|96.250|
|32|`/reference/pyramid.1/pyramid.1.0/AveragePool,/reference/pyramid.1/pyramid.1.1/Conv`|PoolingAve,Convolution|0.010715|0.011745|15.030|8.250|
|33|`/reference/pyramid.1/pyramid.1.2/Div`|BinaryMath|0.003145|0.003402|9.280|8.250|
|34|`/reference/pyramid.1/pyramid.1.2/Erf`|Erf|0.004575|0.005219|8.920|8.250|
|35|`/reference/pyramid.1/pyramid.1.2/Add`|BinaryMath|0.003930|0.004767|9.280|8.250|
|36|`/reference/pyramid.1/pyramid.1.2/Mul`|BinaryMath|0.005000|0.005387|17.390|8.250|
|37|`/reference/pyramid.1/pyramid.1.3/Conv`|Convolution|0.003425|0.003681|11.480|6.250|
|38|`/reference/resizes.1/Gather`|Gather|0.046860|0.057147|26.860|24.250|
|39|`/reference/resizes.1/Mul`|BinaryMath|0.013215|0.013282|28.380|24.250|
|40|`/reference/resizes.1/Gather_1`|Gather|0.044140|0.051340|26.860|24.250|
|41|`/reference/resizes.1/Mul_1`|BinaryMath|0.013500|0.013824|28.500|24.250|
|42|`/reference/resizes.1/Add`|Eltwise|0.012930|0.014280|49.170|24.250|
|43|`/reference/resizes.1/Gather_2`|Gather|0.651570|0.793644|769.980|768.250|
|44|`/reference/resizes.1/Mul_2`|BinaryMath|0.020430|0.022356|97.750|96.250|
|45|`/reference/resizes.1/Gather_3`|Gather|0.450575|0.466519|770.730|768.250|
|46|`/reference/resizes.1/Mul_3`|BinaryMath|0.017715|0.017971|97.750|96.250|
|47|`/reference/resizes.1/Add_1`|Eltwise|0.022070|0.022907|193.050|96.250|
|48|`/reference/Add_1`|Eltwise|0.023140|0.024040|193.170|96.250|
|49|`/reference/pyramid.2/pyramid.2.0/AveragePool,/reference/pyramid.2/pyramid.2.1/Conv`|PoolingAve,Convolution|0.009215|0.009282|14.030|2.250|
|50|`/reference/pyramid.2/pyramid.2.2/Div`|BinaryMath|0.001785|0.001979|3.160|2.250|
|51|`/reference/pyramid.2/pyramid.2.2/Erf`|Erf|0.002215|0.002282|3.050|2.250|
|52|`/reference/pyramid.2/pyramid.2.2/Add`|BinaryMath|0.001570|0.001696|3.160|2.250|
|53|`/reference/pyramid.2/pyramid.2.2/Mul`|BinaryMath|0.002140|0.002140|5.520|2.250|
|54|`/reference/pyramid.2/pyramid.2.3/Conv`|Convolution|0.002430|0.002556|5.480|1.750|
|55|`/reference/resizes.2/Gather`|Gather|0.036570|0.037344|14.860|12.250|
|56|`/reference/resizes.2/Mul`|BinaryMath|0.011355|0.011548|14.750|12.250|
|57|`/reference/resizes.2/Gather_1`|Gather|0.037360|0.038323|21.860|12.250|
|58|`/reference/resizes.2/Mul_1`|BinaryMath|0.011570|0.011696|14.880|12.250|
|59|`/reference/resizes.2/Add`|Eltwise|0.010715|0.010845|25.170|12.250|
|60|`/reference/resizes.2/Gather_2`|Gather|0.458855|0.469524|769.980|768.250|
|61|`/reference/resizes.2/Mul_2`|BinaryMath|0.018715|0.018971|97.750|96.250|
|62|`/reference/resizes.2/Gather_3`|Gather|0.471715|0.498972|769.860|768.250|
|63|`/reference/resizes.2/Mul_3`|BinaryMath|0.018070|0.018520|97.750|96.250|
|64|`/reference/resizes.2/Add_1`|Eltwise|0.022070|0.022520|193.170|96.250|
|65|`/reference/Add_2`|Eltwise|0.022430|0.022817|193.050|96.250|
|66|`/reference/ReduceMean`|Reduction|0.022500|0.023850|99.230|0.440|
|67|`/reference/Add_3`|BinaryMath|0.035855|0.036498|100.500|96.250|
|68|`/reference/project/Conv`|Convolution|0.025785|0.025853|98.480|192.250|
|69|`/reference/sample/Gather`|Gather|0.780285|0.807029|1538.860|1536.250|
|70|`/reference/sample/Mul`|BinaryMath|1.496210|1.502960|1734.120|1536.250|
|71|`/reference/sample/Gather_1`|Gather|0.758215|0.759435|1539.480|1536.250|
|72|`/reference/sample/Mul_1`|BinaryMath|1.502430|1.503456|1734.000|1536.250|
|73|`/reference/sample/Add`|Eltwise|0.331855|0.343299|3074.800|1536.250|
|74|`/reference/sample/Gather_2`|Gather|171.072715|171.100233|122883.610|122880.250|
|75|`/reference/sample/Mul_2`|BinaryMath|2.444715|2.625997|15366.120|15360.250|
|76|`/reference/sample/Gather_3`|Gather|170.923930|171.102067|122884.110|122880.250|
|77|`/reference/sample/Mul_3`|BinaryMath|2.546360|2.580623|15365.620|15360.250|
|78|`/reference/sample/Add_1,/Add,/tail/tail.0/Conv,/conv/Conv,/Add_1`|Eltwise,Eltwise,Convolution,Convolution,Eltwise|5.565425|5.623282|48657.120|2560.250|
|79|`/Mul`|BinaryMath|0.400355|0.416749|2562.160|2560.250|
|80|`/rows/Reshape,/rows/Transpose_transpose_0_0`|Reshape,Permute|0.413360|0.436310|2565.720|2560.250|
|81|`/rows/Transpose_reshape_tail_0_0,/rows/Transpose_transpose_1_0`|Reshape,Permute|0.351715|0.369845|2563.640|2560.250|
|82|`/rows/Transpose_reshape_tail_1_0`|Reshape|0.357355|0.363335|2561.480|2560.250|
|83|`/rows/ConvTranspose_small_0`|Deconvolution|1.102640|1.103090|2575.110|7680.250|
|84|`/rows/ConvTranspose_small_1`|Deconvolution|1.113285|1.114568|2575.980|7680.250|
|85|`/rows/ConvTranspose_small_2`|Deconvolution|1.135430|1.158704|2574.980|7680.250|
|86|`/rows/ConvTranspose_concat,/rows/Transpose_1`|Concat,Permute|3.749500|4.086676|23044.770|23040.250|
|87|`/rows/Reshape_2,/rows/Reshape_2_report_0_0`|Reshape,Reportop|4.490360|4.525523|23041.770|23040.250|

## 全部转换及兼容记录

编译CPU算子与运行故障分别记录。编译成功并不证明稳定运行。

|场景|图/候选|返回码|CPU算子|首个错误|
|---|---|---:|---|---|
|day_normal|c01_r2|0|||
|day_normal|c01_r3|0|||
|day_normal|c01_r4|0|||
|day_normal|c01_source|0|||
|day_normal|c02_r1|0|||
|day_normal|c02_r2|0|||
|day_normal|c02_r3|0|||
|day_normal|c02_source|1||[ERROR][InitVec][170] Layer[/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|day_normal|c03_r1|1||[ERROR][ParseArgs][244] Layer[/current/Conv]'s bias is online, not support.|
|day_normal|c03_r2|1||[ERROR][ParseArgs][244] Layer[/current/Conv]'s bias is online, not support.|
|day_normal|c03_r3|0|||
|day_normal|c03_r4|0|||
|day_normal|c03_source|1||[ERROR][ParseArgs][244] Layer[/current/Conv]'s bias is online, not support.|
|night_ordinary|c01_r1|0|||
|night_ordinary|c01_r3|0|||
|night_ordinary|c01_r6|0|||
|night_ordinary|c01_source|0|/Resize:Resize, /Resize_1:Resize, /Resize_2:Resize||
|night_ordinary|c02_r1|0|||
|night_ordinary|c02_r3|0|||
|night_ordinary|c02_source|1||[ERROR][InitVec][170] Layer[/source/output/output/Reshape_3]: input num axis[8] and output num axis[4] should not be greater than 7!|
|night_ordinary|c03_r1|1||[ERROR][ParseArgs][244] Layer[/source/front/current/Conv]'s bias is online, not support.|
|night_ordinary|c03_r3|0|||
|night_ordinary|c03_source|1||[ERROR][ParseArgs][244] Layer[/source/front/current/Conv]'s bias is online, not support.|
|night_special|c01_r1|0|||
|night_special|c01_r3|0|||
|night_special|c01_r6|0|||
|night_special|c01_source|0|/Resize:Resize||
|night_special|c02_r1|0|||
|night_special|c02_r3|0|||
|night_special|c02_source|1||[ERROR][InitVec][170] Layer[/source/output/Reshape_3]: input num axis[8] and output num axis[4] should not be greater than 7!|
|night_special|c03_r1|1||[ERROR][ParseArgs][244] Layer[/source/front/current/Conv]'s bias is online, not support.|
|night_special|c03_r3|0|||
|night_special|c03_source|1||[ERROR][ParseArgs][244] Layer[/source/front/current/Conv]'s bias is online, not support.|
|weather_heavy|c01_r2|0|||
|weather_heavy|c01_r3|0|||
|weather_heavy|c01_r4|0|||
|weather_heavy|c01_source|0|||
|weather_heavy|c02_r1|0|||
|weather_heavy|c02_r2|0|||
|weather_heavy|c02_r3|0|||
|weather_heavy|c02_source|1||[ERROR][InitVec][170] Layer[/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|weather_heavy|c03_r1|0|||
|weather_heavy|c03_r2|0|||
|weather_heavy|c03_r3|0|||
|weather_heavy|c03_r4|0|||
|weather_heavy|c03_source|1||[ERROR][InitVec][170] Layer[/model/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|weather_heavy_c32|c01_r2|0|||
|weather_heavy_c32|c01_r3|0|||
|weather_heavy_c32|c01_r4|0|||
|weather_heavy_c32|c01_source|0|||
|weather_heavy_c32|c02_r1|0|||
|weather_heavy_c32|c02_r2|0|||
|weather_heavy_c32|c02_r3|0|||
|weather_heavy_c32|c02_source|1||[ERROR][InitVec][170] Layer[/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|weather_heavy_c32|c03_r1|0|||
|weather_heavy_c32|c03_r2|0|||
|weather_heavy_c32|c03_r3|0|||
|weather_heavy_c32|c03_r4|0|||
|weather_heavy_c32|c03_source|1||[ERROR][InitVec][170] Layer[/model/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|weather_light|c01_r2|0|||
|weather_light|c01_r3|0|||
|weather_light|c01_r4|0|||
|weather_light|c01_source|0|||
|weather_light|c02_r1|0|||
|weather_light|c02_r2|0|||
|weather_light|c02_r3|0|||
|weather_light|c02_source|1||[ERROR][InitVec][170] Layer[/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|weather_light|c03_r1|0|||
|weather_light|c03_r2|0|||
|weather_light|c03_r3|0|||
|weather_light|c03_r4|0|||
|weather_light|c03_source|1||[ERROR][InitVec][170] Layer[/model/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|weather_medium|c01_r2|0|||
|weather_medium|c01_r3|0|||
|weather_medium|c01_r4|0|||
|weather_medium|c01_source|0|||
|weather_medium|c02_r1|0|||
|weather_medium|c02_r2|0|||
|weather_medium|c02_r3|0|||
|weather_medium|c02_source|1||[ERROR][InitVec][170] Layer[/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|
|weather_medium|c03_r1|0|||
|weather_medium|c03_r2|0|||
|weather_medium|c03_r3|0|||
|weather_medium|c03_r4|0|||
|weather_medium|c03_source|1||[ERROR][InitVec][170] Layer[/model/output/Reshape_2]: input num axis[8] and output num axis[4] should not be greater than 7!|

## 核验与实验收口

|项目|结果|
|---|---|
|来源|951项载荷SHA复验通过；固定946305bbfda9c832ddc95a5ac789567845040abe|
|转换记录|85次，完整首错/CPU列表见上一节|
|完整同步|18组50预热＋1000正式；18000成功正式调用|
|输出一致性|17份对同版本PC诊断，不是GT验收|
|独立NPU模块|仅日间未改源图1预热＋2正式；不当1000稳态|
|失败部分|普通夜间运动50预热＋103成功正式，部分同步均值48.638488ms；随后报错，无最终输出|
|驱动累计|timeout/hw/AICPU 0/1/1→0/2/1，新增1次hw错误|
|延后只读复查|boot未变、业务13547/13833在、测试进程退出、MMZ恢复1590548KiB，错误计数未继续增加|
|安全状态|所有后续NPU推理停止，旧严格门限未放宽，不自动重试故障族|
|原始证据|成功CSV/输出/模型SHA核验；故障部分CSV及临时文件主机SHA归档后仅清理板端临时副本|
|历史|冻结旧报告SHA未变，旧模型未复测|
|验收|真实GT/时序、RAW全链、生产长稳均未通过；不是零故障发布|

业务存活和资源恢复只是观察，不保证业务图像质量未受影响，也不是故障后可以安全重启推理的授权。源算子根因尚未定位，不能将驱动错误等同于某一个ONNX算子不支持。

已更新项目STATUS、迭代台账、实验记录和本地ISSUE-016。未提交GitHub Issue、未替换生产模型、未重启或停止业务、未改固件/系统库/频率。

### 原始证据定位

- 本轮制品与CSV：`artifacts/EXP-20261001-014/`。
- 普通夜间故障：`artifacts/EXP-20261001-014/night_ordinary/board_c03_r3_m6_steady/`，包含终端日志、失败快照、`timing_partial.csv`、归档/清理核验。
- 来源/输入/兼容回归：`records/iterations/v0.13/download-audit.json`、`source-audit.json`、候选JSON及`models/handoff/apple-20261001-946305b/`。
- 板端只读证据：`records/iterations/v0.13/board-final-after-failure.json`、`board-delayed-readonly-final.json`、`kernel-log-after-motion-failure.json`。
- 报告SHA与闭环：`records/iterations/v0.13/final-verification.json`、`closure-audit.json`。

这些证据是复核材料，不是阅读本报告结论的前置条件；分享时可先发送本合并文档，只有复现实验才需要另带模型/日志/数据。
