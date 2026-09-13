# MhcSinkhorn 测试

案例按用途分为三类，共享输入生成、参考计算和 ACLNN runner。

| 类别 | 配置 | 数量 | 用途 |
|---|---|---:|---|
| Coverage | `st/cases/coverage.json` | 30 | float16/float32、3D/4D、n=4/6/8、迭代范围、batch 与尾块 |
| Correctness | `st/cases/correctness.json` | 105 | 文档示例、特殊值、eps、精度敏感性、收敛顺序、因子归一化与兜底路径压力校验 |
| Mock | `st/cases/mock.json` | 5 | 固定五组负载；数值、收敛敏感性与 MSPTI 计时 |

`full` 为 Coverage 与 Correctness 并集，共 135 项。所有设备用例均比较数值；
类别表示主要测试目的。Mock 使用固定合成输入，规格见下表；输入与参考结果均由仓库内定义独立生成。

## 目录

```text
tests/
├── common/
│   ├── cases.py              # 三类设备用例的加载与筛选
│   ├── contract_cases.py     # 题面参考自检的 35 有效、11 非法输入
│   ├── reference.py          # 严格 FP32 参考与全部中间状态
│   ├── mock_inputs.py        # 五个 Mock 的固定输入参数与生成器
│   └── fixtures.py           # 设备输入、golden 与二进制 fixture
├── st/
│   ├── cases/                # coverage/correctness/mock.json
│   ├── runner/               # ACLNN 执行、比较与计时
│   │   └── device_state/     # 独立的 UB 状态准备 Kernel
│   └── scripts/
│       ├── run_tests.py      # 选择 suite、构建工作区源码、执行与报告
│       ├── profile_mock_workloads.py # 当前源码的五例数值与性能入口
│       ├── build_ops.py      # 算子与 runner 构建、执行和身份校验
│       ├── artifact_cache.py # 通用内容寻址缓存
│       └── submission_preflight.py # 上传文件与构建检查
└── ut/
    ├── gen_data.py           # 参考契约数据生成与校验
    ├── coverage_audit.py     # 题面参考契约覆盖审计
    ├── test_case_matrix.py   # 三类隔离、筛选与设备输入检查
    └── test_mhc_sinkhorn.py  # 数学次序、布局、确定性与非法输入
```

UT 按检查对象组织：

| 文件 | 检查内容 |
|---|---|
| `ut/test_host_contract.py`、`ut/host_contract.cpp` | Host 输入接受与拒绝、输出 Shape/DType、逻辑 Block 范围及资源元数据 |
| `ut/test_mhc_sinkhorn.py` | 参考数学、迭代次序、中间状态、布局和确定性 |
| `ut/test_data_generation.py` | FP16/FP32 输入落盘、golden 舍入、矩阵排列、缓存复用及损坏重建 |
| `ut/test_profile_mapping.py` | Case/report 映射、采集重试和完整三轮 MSPTI 统计 |
| `ut/test_case_matrix.py` | 分类、筛选、输入难度、混合收敛批次及运行设施自检 |

Host 契约测试使用 CANN Context Builder，在 CPU 上调用算子定义中的 Host 回调，
无需 NPU；116 个场景覆盖两种 Schema dtype、3D/4D、n=4/6/8、迭代边界、
单矩阵/尾块/多块、非法形状和 dtype、计数溢出，以及不同 eps 参数。
合法输入须完成输出 Shape/DType 推导与 Tiling；非法输入可在任一 Host 阶段被拒绝。
Tiling 数据作为不透明字节流，只检查报告长度不超过所提供的缓冲容量。
逻辑 Block 数按 SIMD 接口的 `[1, 65535]` 范围检查；workspace 允许为零或非零，
检查所声明尺寸已赋值且求和无溢出。测试容器提供默认 2048 字节 Tiling 缓冲与 16 个 workspace 尺寸槽位。
设备测试检验实际计算、分块覆盖、尾块结果及运行时资源使用；Host UT 不规定分核算法或私有字段布局。
缺少 CANN SDK 或 Context Builder 时该项显示 skip；编译和断言失败直接报错。
Host 测试不覆盖 mask 和 normOut/sumOut 的设备 ABI。

`common/contract_cases.py` 是 CPU 参考自检的输入定义；设备用例分类以三份 JSON 为准。
设备用例通过 `ut_case` 复用契约输入生成器，并根据运行 dtype 生成参考输出。
`multi_special` 使用固定 seed 和非有限值位置生成特殊输入。
仓库保存用例定义和生成代码；NPZ、manifest 与设备二进制数据按需生成并缓存于
`/tmp/cannjudge/mhcsinkhorn`。缓存身份包含生成器、参考代码与 NumPy 版本，
缓存条目仅在文件哈希校验通过时复用。

## 用例矩阵

Coverage 为两种 dtype × n=4/6/8 × iterations=1/2/20/100 的 24 个组合，
同时安排 3D/4D、单矩阵、batch 分解及尾块；另有 6 个多块用例，各使用 4097 个矩阵。

Correctness 按数值机制选取固定输入：

| 用例组 | 数量 | 检查内容 |
|---|---:|---|
| `ut_*`、`half_*` | 30 | 文档公式、结构化输入、eps、极值与非有限值 |
| `factor_*` | 30 | 每种 dtype/n 覆盖迭代 4/5/20/99/100，eps 取下界、常规、中间值及上界 |
| `multi_factor_*` | 16 | n=6/8 的多矩阵归一化，3D/4D、奇偶迭代和尾块 |
| `multi_special_*` | 4 | 多矩阵中的 NaN/Inf 传播与独立性 |
| `path_*` | 16 | 小 batch 的极小 eps、大 batch 的较大 eps，以及 n=6/8 的三次迭代 |
| `sensitivity_*` | 3 | 固定半精度实现敏感性输入 |
| `convergence_*` | 6 | FP16/FP32 混合收敛矩阵，覆盖易在前、难在前和交错排列 |

配置每行定义一个完整用例，Case ID、seed、形状和数值参数均显式保存。
Mock 独立用于性能采样，不属于 `full`。

五个 Mock 均使用 float16、无 mask、eps=1e-6：

| Case | Shape | iterations | atol / rtol |
|---|---|---:|---:|
| `mock_case_1` | (1,2,4,4) | 30 | 0.01 / 0.01 |
| `mock_case_2` | (1,4,8,8) | 50 | 0.02 / 0.02 |
| `mock_case_3` | (1,8,8,8) | 20 | 0.01 / 0.01 |
| `mock_case_4` | (2,4,8,8) | 50 | 0.01 / 0.01 |
| `mock_case_5` | (1,16,6,6) | 100 | 0.01 / 0.01 |

`common/mock_inputs.py` 从 `[-1, 1]` 均匀分布按固定 seed 生成 logits，再舍入为 FP16。
配置中的 `matrix_order` 固定 Case 3、4 的矩阵排列，Case 5 使用独立矩阵 seed 序列。
输入按需生成，golden 由独立 FP32 数学参考计算后转为 FP16；生成代码参与缓存身份。
固定合成输入用于本地数值校验与性能比较。
数值比较使用 `abs(actual - expected) <= atol + rtol * abs(expected)`。
Coverage 与 Correctness 的 FP16/FP32 数值容差均为 atol=rtol=0.001；
Mock 使用表中显式容差。输入、形状、迭代次数、eps 和特殊值覆盖独立于容差配置。
CPU 数学参考自检验证 golden 生成器；设备候选的误差门槛由上述用例配置定义。

CPU 单元测试检查一轮归一化后仍存在超过 `atol=rtol=1e-3`
的误差，防止输入退化为过于简单的负载；输入哈希测试同时防止无意修改数据。
`profile_mock_workloads` 的 PASS 表示五个 Mock 数值校验和采样完整性检查通过，
报告提供本地 Kernel 耗时与错误元素数。

`convergence_order` 正确性用例使用独立的 Gaussian logits，明确将两个矩阵置为均匀输入，
每组覆盖 FP16/FP32 与易在前、难在前和交错三种排列。
易矩阵在前、难矩阵在后的组合专门检出仅检查批次首矩阵便退出的错误。
可用 `--suite correctness --tag convergence_order` 单独执行；它们不参与五个 Mock 的性能统计。
可选 `uniform_matrices` 指定置零 logits 的矩阵索引，`matrix_order` 必须是全部矩阵索引的排列；
先置零后排列，两项均参与输入缓存身份。

## 运行

在项目根目录执行：

```bash
python3 -m tests.st.scripts.run_tests --suite coverage --list
python3 -m tests.st.scripts.run_tests --suite correctness --list
pytest -q tests/ut
python3 -m tests.ut.gen_data --check
python3 -m tests.ut.coverage_audit

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0
python3 -m tests.st.scripts.run_tests --suite coverage
python3 -m tests.st.scripts.run_tests --suite correctness
python3 -m tests.st.scripts.run_tests --suite full
python3 -m tests.st.scripts.profile_mock_workloads
```

使用 `--case NAME` 筛选所选类别，可重复指定；跨类别选择用 `--suite custom --case NAME`。
指定了类别但案例不属于该类别时直接报错。`--tag` 可重复指定，只选择同时具有全部标签的案例；无匹配时报错。`--list` 仅列出案例，无需设备与构建。

构建默认 16 路并发，使用 `--jobs` 调整；设备测试产物放在 `/tmp/cannjudge/mhcsinkhorn/local-tests`。
缓存条目的文件清单与校验信息保存在 `.artifact-cache.json`。
`--build-dir` 指定工作区源码的构建缓存根，`--report` 指定报告路径；运行不同集合时可分别指定报告。
正确性用例预热 1 次、执行 3 次比较。全部五个 Mock 采用同一计时协议：
三个独立进程，每进程预热 4 次、测量 12 次，取三个进程均值的中位数。
报告保存实际预热次数与准备阶段用时。使用 MSPTI Kernel 时间戳，
预热阶段用时不计入被测 Kernel 耗时。每次测量前先在独立进程中执行相同的设备状态准备：
对 40 个 AIV block 的前 128 KiB UB 写入 float32 1.0，并核验各 block 的完成回执；
该进程退出后，再启动只执行被测算子的测量进程。准备 Kernel 的时间不计入被测耗时。
该准备步骤模拟非零临时缓冲状态，帮助稳定复现未初始化 lane 引起的行为；
所有 Mock 使用相同的设备准备设置。
JSON 报告记录计时参数、案例结果与构建身份。
数值错误计数取首次预热输出与最终输出中最早发现的数值失败；最终输出仍会检查，
首尾输出不一致仍报告 Wrong Answer。`*.output.tsv` 保存错误计数对应的输出，
不一致时另保存 `*.first_output.tsv` 或 `*.last_output.tsv` 供定位；无预热时检查最终输出。
可用 `--warmup` 统一覆盖所选 Case 的预热次数，`--warmup-us` 调整准备阶段的最短时间，
`--repeat` 统一覆盖所选 Case 的采样次数；未指定时采用用例配置中的 `warmup` 和 `repeat`。
`--warmup-us` 未指定时采用用例配置中的 `warmup_us`，Mock 缺省为 0。所有版本共用这些固定 Case 参数。
`--warmup-us 0` 仅按次数预热。用 `--interval-us` 设置调用间隔（不计入 Kernel 耗时），
或用 `--device-state unprepared` 关闭设备准备。比较不同版本时应保持这些参数一致。

题面仅列出 float32；Host Schema 的 float16 设备兼容性单独由 Coverage 覆盖。
CPU 参考契约仍检查 float16 非法输入。设备用例不覆盖题面未定义的 mask 和 normOut/sumOut ABI。
参考数据字段、中间状态布局及特殊值说明见 [ut/README.md](ut/README.md)。

## 实现敏感性校验

下面三例属于 `correctness.json`，标签为 `sensitivity`：

| 案例 | Shape | 类型 | iterations | seed |
|---|---|---|---:|---:|
| `sensitivity_half_n4_rows` | (1,2,4,4) | float16 | 30 | 2026091304 |
| `sensitivity_half_n8_rows` | (1,4,8,8) | float16 | 50 | 2026091308 |
| `sensitivity_half_n6_rows` | (1,16,6,6) | float16 | 100 | 2026091306 |

输入为固定 seed 的 uniform(-4,4)，eps=1e-6，rtol=atol=0.001。
输入先转为 FP16，独立参考按严格 FP32 次序计算，再按 FP16 输出比较。
三例分别覆盖不同矩阵宽度的半精度行复制、n=6 对齐、交替归一化与最终写回；
非均匀行可检出直接拷贝、初始状态写回、错误步长和错误输出等实现缺陷。

```bash
python3 -m tests.st.scripts.run_tests --suite correctness --tag sensitivity --list
python3 -m tests.st.scripts.run_tests --suite correctness --tag sensitivity \
  --report /tmp/mhc-sensitivity/run.tsv
```

包含敏感性用例时，执行入口按 `template-manifest.json` 的上传文件集构建，
本地额外头文件不会被带入构建；失败保存编译日志并返回 Compile Error。
编译成功后运行三个独立进程，各预热一次、执行三次，比较最终输出，并检查其与首次预热输出逐字节一致。
每个进程都必须通过，实际加载库和构建产物均核验哈希。
普通正确性模式的同步计时仅为诊断；敏感性校验不使用耗时作为判定条件。
设备异常、超时或测试设施错误返回 INCONCLUSIVE（退出码 2），不作为 WA/CE 证据。

`tests/` 可作为自包含测试组件，与本题接口的 `code/` 和上传清单配合运行。
执行不要求 Git 或官方服务。CE 由构建阶段检查，三份输入承担数值校验。
这组有限用例覆盖表中列出的敏感机制，不代表对所有输入的完备证明。
