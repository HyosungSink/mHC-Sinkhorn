# MhcSinkhorn UT

本目录包含真实 Host/Tiling 契约、参考数学、数据缓存和 Profiler 报告自检。
Host 测试使用 CANN Context Builder 调用 `code/op_host/mhc_sinkhorn.cpp` 中算子定义的回调，
检查 Host Schema 的 float16/float32 输入、输出 Shape/DType、参数接受与拒绝及资源元数据。
参考数学以题面第 3 节的 float32 运算次序作为 golden oracle，覆盖最终输出、
全部中间状态和物理对齐。两类测试均不启动 NPU Kernel。

`test_host_contract.py` 自动查找 CANN SDK；也可通过 `ASCEND_TOOLKIT_HOME` 或
`ASCEND_HOME_PATH` 指定。编译产物位于 pytest 的仓库外临时目录。
缺少 SDK/Context Builder 时跳过 Host 测试；SDK 可用时，编译或运行失败均使测试失败。

## 内容

- `host_contract.cpp`、`test_host_contract.py`：116 个真实 Host/Tiling 场景，包含
  合法形状、非法 rank/n/dtype/iterations、计数溢出及输出推导。
  非法输入可在 Shape 推导、DType 推导或 Tiling 阶段被拒绝。
  Tiling 载荷只检查长度边界；逻辑 Block 数检查 `[1, 65535]`，workspace 检查赋值与尺寸求和溢出。
  测试容器提供 2048 字节 Tiling 缓冲与 16 个 workspace 尺寸槽位。
  私有字段、对齐方案、分核算法和 workspace 字节数由实现决定；实际分块覆盖与资源使用由设备测试检验。
- `test_data_generation.py`：固定输入与存储精度、golden 舍入、缓存身份，以及
  输入/golden/描述文件被截断、等长篡改或删除后的重建。
- `test_profile_mapping.py`：报告按 Case ID 映射、重复 ID 拒绝、仅采集不完整时重试，
  以及三轮完整性与统计；使用合成报告，不启动设备采集。

- `../common/reference.py`：严格 float32 的参考实现，保存每个 `normOut` 和
  `sumOut` 步骤。
- `../common/contract_cases.py`：35 个有效用例和 11 个非法参数用例的唯一来源。
- `gen_data.py`：生成或逐字段复核 仓库外缓存中的压缩 NPZ 数据。
- `coverage_audit.py`：检查题面要求是否在测试矩阵中出现，并检查每个
  `n=4/6/8` 都覆盖 `numIters=1/2/20/100`。
- `test_mhc_sinkhorn.py`：检查递推顺序、最终索引、批间独立性、物理布局、
  确定性、非法输入及文档示例。
- 生成的 `manifest.json`：供 NPU 测试适配器读取的完整用例清单。

数据在测试运行时自动生成，默认缓存根为 `/tmp/cannjudge/mhcsinkhorn/contract-data`。
缓存按生成源码与 NumPy 版本区分，复用前检查文件哈希。
`--output-dir /绝对路径` 可导出 NPZ 和 manifest；仓库不保存这些生成结果。

生成和检查：

```bash
python3 -m tests.ut.gen_data
python3 -m tests.ut.gen_data --check
python3 -m tests.ut.coverage_audit
pytest -q tests/ut
```

## NPZ 字段

有效用例包含 `x`、`output`、`norm_out`、`sum_out`、两种物理对齐输出、两种
有效位 mask、`num_iters` 和 `eps`。物理布局为：

```text
norm_out_physical: [2*num_iters, *leading, n, 8]
sum_out_physical:  [2*num_iters, *leading, 8]
```

比较物理结果时必须使用 `norm_valid_mask` 和 `sum_valid_mask`。`sumOut[0]` 是未
定义占位，所有 padding lane 也没有规定值，测试不得比较这些位置。NPZ 中以 NaN
填充只是为了尽早暴露误用，不是 Kernel 的输出要求。

此数据集用于 CPU 数学参考与 golden 生成器的自检，manifest 描述数据字段、用例和参考契约。
设备精度验收采用 `../st/cases/` 中所选用例的配置：Coverage 与 Correctness 的
`atol=rtol=0.001`，Mock 使用各用例的显式容差。

## 覆盖结论

测试矩阵覆盖：

- 3D/4D、`n=4/6/8`，以及每个 n 下的 `numIters=1/2/20/100`；
- 单矩阵、多矩阵、4D 分解 batch 和 41 个矩阵的尾块场景；
- 全 0、全 1、负常量、非均匀、重复行、强对角、棋盘格、近均匀随机数据；
- 极大有限值、float32 边界、softmax 下溢、NaN、正无穷、负无穷及混合特殊值；
- `eps=0/1e-12/1e-6/0.1`，可检测 `eps` 是否加在规定位置；
- 输出 Shape/DType、每个中间步骤、最终索引、n 到 8 的物理对齐、批间独立性与
  重复执行确定性；
- 非法 rank、非方阵、非法 n、迭代次数上下界外、非整数迭代值，以及
  float16/float64/int32 输入。

manifest 中的 `runtime_only_checks` 作为参考契约说明，记录 ND 格式限制、四种可选输出
组合、输出元数据、重复启动、`sumOut[0]` 不比较和 padding 不参与计算等运行时事项。
设备测试范围由 `../st/cases/` 中的用例定义。

## 接口边界

- `mask` 没有有效值、广播、参与归一化或全遮蔽规则，因此不生成 mask 用例。
- 题面参考契约仅接受 float32，因此契约数据把 float16 列为非法 DType；Host/Tiling
  测试则按照 Host Schema 单独覆盖 float16/float32。
- Host Schema 未声明 `normOut/sumOut`，设备运行时不检查这两个输出；参考数据仍包含
  相应中间状态，用于数学次序和物理布局校验。

## 文档歧义

题面一处文字称输入出现 `-inf/inf/nan` 时对应输出为 NaN，但给出的稳定 softmax
公式对“有限数值中夹一个 `-inf`”会得到 0，而不是 NaN；全 `-inf` 行、`+inf` 和
NaN 则会按 IEEE 运算继续传播。golden 采用更具体的逐步公式，
`single_neginf_n6_3d` 用例用于标记该歧义。评测器若明确要求预先把所有非有限输入位置
写成 NaN，只需切换该特殊值 oracle，不能在 Kernel 中暗自猜测。
