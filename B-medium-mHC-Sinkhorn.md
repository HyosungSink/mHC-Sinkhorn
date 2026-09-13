# 【B组中等题】mHC-Sinkhorn 算子

---

## 1. 算子说明

MhcSinkhorn 算子基于 Sinkhorn-Knopp 迭代算法，将一个任意非负方阵变换为**双随机矩阵**（即每行元素之和、每列元素之和均为 1 的方阵）。该算子常用于深度网络中稳定信号传播、缓解梯度消失/爆炸问题，其输出可作为后续网络层的输入。

算子接收输入矩阵 `x`（形状最后两维为 n×n 方阵），对其进行 `numIters` 次交替的"行归一化—列归一化"迭代，最终输出双随机矩阵 `output`。迭代过程中还可输出归一化中间结果 `normOut` 与求和中间结果 `sumOut`，用于后续反向梯度计算。

---

## 2. 输入输出说明

### 输入规格

| 输入 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| x | 张量 | (B, S, n, n) 或 (T, n, n) | float32 | 输入张量，最后两维为待归一化的 n×n 方阵；前序维度（T 或 B、S）为批量维度，各方阵独立计算 |
| eps | 属性 | 标量 | float32 | 归一化防除零参数，建议值 1e-6 |
| numIters | 属性 | 标量 | int64 | Sinkhorn 迭代次数，控制收敛过程，取值范围 1~100，建议值 20 |

### 输出规格

| 输出类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|
| output | 与 x 一致：(B, S, n, n) 或 (T, n, n) | float32 | Sinkhorn 变换最终结果（双随机矩阵，行和、列和均为 1） |
| normOut | 逻辑形状 (2*numIters, B, S, n, n) 或 (2*numIters, T, n, n)；物理存储 size = 2*numIters·n·n_align·B·S 或 2*numIters·n·n_align·T | float32 | 迭代过程中每步的归一化中间矩阵，用于反向，可选（不输出时传空指针） |
| sumOut | 逻辑形状 (2*numIters, B, S, n) 或 (2*numIters, T, n)；物理存储 size = 2*numIters·n_align·B·S 或 2*numIters·n_align·T | float32 | 迭代过程中每步的行/列求和中间结果，用于反向，可选（不输出时传空指针） |

### 形状约束

- 输入 `x` 仅支持 3 维 `(T, n, n)` 或 4 维 `(B, S, n, n)`，其他维度数不支持。
- 矩阵维度 `n` 仅支持取值 **4、6、8**（即输入最后两维的大小）。
- `numIters` 取值范围为 1~100，超出范围报参数无效错误。
- 仅支持 `FLOAT32` 数据类型与 `ND` 数据格式（即任意多维非结构化连续格式），不支持 float16/double 等其他精度。
- `normOut` 与 `sumOut` 为可选输出，传空指针时不输出对应结果。
- 输入含 `-inf/inf/nan` 时，对应位置输出 `nan`。
- 算子默认采用确定性实现，相同输入多次调用结果一致。

### 输出形状计算公式

- `output` 形状与输入 `x` 完全一致。
- `normOut` 在 `x` 形状前增加一维 `2*numIters`：逻辑形状为 `(2*numIters, B, S, n, n)` 或 `(2*numIters, T, n, n)`，`normOut[k]` 为第 `k` 步归一化结果。
- `sumOut` 在 `x` 形状（去掉被求和的那一维）前增加一维 `2*numIters`：逻辑形状为 `(2*numIters, B, S, n)` 或 `(2*numIters, T, n)`，`sumOut[k]` 为第 `k` 步行/列求和结果。
- **物理存储对齐**：实际 Device 内存中，`n` 维按 8 对齐存储，记 `n_align = ceil(n/8)*8`（n=4/6/8 时 n_align 均为 8）。因此 `normOut` 物理元素数为 `2*numIters·n·n_align·(B·S 或 T)`，`sumOut` 物理元素数为 `2*numIters·n_align·(B·S 或 T)`（sumOut 求和后仅保留单个 n 维，按 n_align 对齐）。当 n=4 时物理元素数是逻辑形状的 2 倍，内存申请须以物理 size 为准。该对齐规则对内存分配与 tiling 切分至关重要。

---

## 3. 算子逻辑说明

### 3.1 初始化阶段（第 1 次迭代）

1. 对输入 `x` 沿最后一维（`dim=-1`，行方向）执行 softmax 归一化，使每行元素之和为 1，并加上防除零参数 `eps`，得到 `normOut[0]`。
2. 对 `normOut[0]` 沿倒数第二维（`dim=-2`，列方向）求和并 keepdim，加上 `eps`，得到列和 `sumOut[1]`。
3. 将 `normOut[0]` 按元素除以 `sumOut[1]`，完成列归一化，得到 `normOut[1]`（此时每列之和为 1）。

### 3.2 交替迭代归一化阶段（第 i 次迭代，i = 1, 2, …, numIters-1）

每次迭代包含一次行归一化和一次列归一化，交替执行使矩阵逐步逼近双随机矩阵：

1. **行归一化**：对 `normOut[2i-1]` 沿 `dim=-1`（行方向）求和并 keepdim，加 `eps` 得 `sumOut[2i]`；`normOut[2i-1] ÷ sumOut[2i]` 得 `normOut[2i]`（每行和为 1）。
2. **列归一化**：对 `normOut[2i]` 沿 `dim=-2`（列方向）求和并 keepdim，加 `eps` 得 `sumOut[2i+1]`；`normOut[2i] ÷ sumOut[2i+1]` 得 `normOut[2i+1]`（每列和为 1）。

### 3.3 最终输出与中间结果

- 最终输出取最后一次迭代的归一化结果：`output = normOut[2*numIters-1]`。
- 当 `numIters=1` 时，仅执行初始化阶段，输出 `normOut[1]`（仅满足列和为 1）。
- 可选输出 `normOut`、`sumOut` 保存全部迭代中间状态，供反向算子计算梯度使用。
- 注意：`normOut` 索引从 0 开始（normOut[0]..normOut[2*numIters-1] 均有效）；`sumOut` 索引从 **1** 开始（sumOut[1] 为首个有效值，由初始化阶段写入），`sumOut[0]` 为占位未定义（分配空间但未写入），不作为正确性中间测试点。

---

## 4. 决赛任务要求

- **精度保障**：针对不同 shape 维度（3 维/4 维）与不同 `n` 取值（4/6/8），设计算子逻辑，保证 Sinkhorn 迭代归一化精度正确，行和、列和收敛至 1。
- **性能优化**：充分发挥系统带宽能力，合理复用中间结果内存，算子性能更优。
- **切分最优**：探索输入 tensor 在 `(T, n, n)` 与 `(B, S, n, n)` 两种场景下的切分方式，找到不同输入 shape 场景下的最优解。

---

## 功能示例

```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def impl(x, eps, num_iters):
    # x: (T, n, n) 或 (B, S, n, n)，最后两维为方阵
    x = x.astype(np.float32)
    # 初始化阶段（第 1 次迭代）
    norm_out = softmax(x, axis=-1) + eps                       # normOut[0]
    sum_out = np.sum(norm_out, axis=-2, keepdims=True) + eps   # sumOut[1]
    norm_out = norm_out / sum_out                              # normOut[1]（列和为1）
    # 交替迭代阶段（i = 1 .. num_iters-1）
    for i in range(1, num_iters):
        sum_out = np.sum(norm_out, axis=-1, keepdims=True) + eps  # sumOut[2i]
        norm_out = norm_out / sum_out                            # normOut[2i]（行和为1）
        sum_out = np.sum(norm_out, axis=-2, keepdims=True) + eps # sumOut[2i+1]
        norm_out = norm_out / sum_out                            # normOut[2i+1]（列和为1）
    return norm_out  # output = normOut[2*num_iters-1]

np.set_printoptions(precision=4, suppress=True)

# 示例1：全 1 矩阵（T=1, n=4），已天然对称，1 次迭代即收敛为均匀双随机矩阵
x = np.ones((1, 4, 4), dtype=np.float32)
y = impl(x, eps=1e-6, num_iters=20)
# 输出形状：(1, 4, 4)；squeeze 第0维后矩阵每元素 0.25，行和=1、列和=1
# [[0.25 0.25 0.25 0.25]
#  [0.25 0.25 0.25 0.25]
#  [0.25 0.25 0.25 0.25]
#  [0.25 0.25 0.25 0.25]]

# 示例2：非均匀矩阵（T=1, n=4），迭代 20 次后逼近双随机矩阵
x = np.array([[[4, 3, 2, 1],
               [1, 2, 3, 4],
               [2, 2, 2, 2],
               [3, 1, 4, 2]]], dtype=np.float32)
y = impl(x, eps=1e-6, num_iters=20)
# 输出形状：(1, 4, 4)；squeeze 第0维后行和≈1、列和≈1（双随机）
# [[0.5248 0.3838 0.0618 0.0297]
#  [0.0281 0.1517 0.1804 0.6399]
#  [0.2003 0.3981 0.1742 0.2274]
#  [0.2469 0.0664 0.5836 0.1031]]

# 示例3：3 维批量输入（T=2, n=4），对每个方阵独立做 Sinkhorn 变换
x = np.stack([
    np.ones((4, 4), dtype=np.float32),
    np.array([[4,3,2,1],[1,2,3,4],[2,2,2,2],[3,1,4,2]], dtype=np.float32)
])  # shape = (2, 4, 4)
y = impl(x, eps=1e-6, num_iters=20)
# 输出形状：(2, 4, 4)；y[0] 为全 0.25 均匀双随机矩阵，y[1] 为示例2 的非均匀双随机矩阵
```

