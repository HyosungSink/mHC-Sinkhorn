// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// Each of the batch matrices is an n x n square block of the flattened input.
// The kernel holds one matrix in a padded 8x8 fp32 tile (64 lanes, one 32B data
// block per row), so the host only has to describe the batch split.
struct MhcSinkhornTilingData {
    uint32_t matrices;         // number of independent n x n matrices
    uint32_t matricesPerCore;  // matrices assigned to one core
    uint32_t n;                // matrix order: 4, 6 or 8
    uint32_t iterations;       // Sinkhorn iterations, 1..100
    float eps;                 // normalization guard
    uint32_t dtypeSize;        // input element size in bytes (2 or 4)
    uint32_t reserved0;
    uint32_t reserved1;
};
