// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcSinkhornTilingData {
    uint32_t numMat;      // 方阵总数（前序维度的乘积）
    uint32_t matPerCore;  // 每个逻辑Block处理的方阵数上限
    uint32_t iters;       // Sinkhorn迭代次数
    uint32_t order;       // 方阵阶数 n，供通用兜底路径使用
    float eps;            // 防除零参数
};
