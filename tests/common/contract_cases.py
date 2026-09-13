"""Deterministic test matrix for the documented MhcSinkhorn interface."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

import numpy as np
import numpy.typing as npt


DEFAULT_EPS = 1.0e-6
ALL_OUTPUT_MODES = ("output_only", "with_norm", "with_sum", "with_both")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    shape: tuple[int, ...]
    iterations: int
    eps: float
    pattern: str
    seed: int
    tags: tuple[str, ...]
    runtime_modes: tuple[str, ...] = ("with_both",)


@dataclass(frozen=True)
class InvalidCaseSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    iterations: object
    eps: object
    expected_exception: type[Exception]
    reason: str


VALID_CASES = (
    CaseSpec(
        "doc_ones_n4_3d_i20",
        (1, 4, 4),
        20,
        DEFAULT_EPS,
        "ones",
        1,
        ("document_example", "constant", "recommended_iterations"),
        ("output_only", "with_both"),
    ),
    CaseSpec(
        "doc_nonuniform_n4_3d_i20",
        (1, 4, 4),
        20,
        DEFAULT_EPS,
        "doc_nonuniform",
        2,
        ("document_example", "nonuniform", "recommended_iterations"),
        ALL_OUTPUT_MODES,
    ),
    CaseSpec(
        "doc_batch_n4_3d_i20",
        (2, 4, 4),
        20,
        DEFAULT_EPS,
        "doc_batch",
        3,
        ("document_example", "batch_independence", "recommended_iterations"),
    ),
    CaseSpec(
        "boundary_i1_n4_3d",
        (3, 4, 4),
        1,
        DEFAULT_EPS,
        "monotonic",
        4,
        ("iteration_boundary", "no_loop", "nonuniform"),
    ),
    CaseSpec(
        "boundary_i2_n4_4d",
        (2, 2, 4, 4),
        2,
        DEFAULT_EPS,
        "random_normal",
        5,
        ("iteration_boundary", "first_loop", "nonuniform"),
    ),
    CaseSpec(
        "boundary_i100_n4_3d",
        (2, 4, 4),
        100,
        DEFAULT_EPS,
        "random_uniform",
        6,
        ("iteration_boundary", "maximum_iterations", "nonuniform"),
    ),
    CaseSpec(
        "boundary_i1_n6_4d",
        (1, 1, 6, 6),
        1,
        DEFAULT_EPS,
        "diagonal",
        7,
        ("iteration_boundary", "no_loop", "singleton_4d"),
    ),
    CaseSpec(
        "boundary_i2_n6_3d",
        (4, 6, 6),
        2,
        DEFAULT_EPS,
        "checkerboard",
        8,
        ("iteration_boundary", "first_loop", "nonuniform"),
    ),
    CaseSpec(
        "recommended_i20_n6_4d",
        (2, 3, 6, 6),
        20,
        DEFAULT_EPS,
        "random_normal",
        9,
        ("recommended_iterations", "nonuniform"),
    ),
    CaseSpec(
        "boundary_i100_n6_3d",
        (2, 6, 6),
        100,
        DEFAULT_EPS,
        "monotonic",
        10,
        ("iteration_boundary", "maximum_iterations", "nonuniform"),
    ),
    CaseSpec(
        "boundary_i1_n8_3d",
        (1, 8, 8),
        1,
        DEFAULT_EPS,
        "random_uniform",
        11,
        ("iteration_boundary", "no_loop", "naturally_aligned"),
    ),
    CaseSpec(
        "boundary_i2_n8_4d",
        (2, 1, 8, 8),
        2,
        DEFAULT_EPS,
        "diagonal",
        12,
        ("iteration_boundary", "first_loop", "naturally_aligned"),
    ),
    CaseSpec(
        "recommended_i20_n8_3d",
        (3, 8, 8),
        20,
        DEFAULT_EPS,
        "checkerboard",
        13,
        ("recommended_iterations", "nonuniform", "naturally_aligned"),
    ),
    CaseSpec(
        "boundary_i100_n8_4d",
        (1, 2, 8, 8),
        100,
        DEFAULT_EPS,
        "random_normal",
        14,
        ("iteration_boundary", "maximum_iterations", "naturally_aligned"),
    ),
    CaseSpec(
        "constant_zero_n4_4d",
        (2, 1, 4, 4),
        2,
        DEFAULT_EPS,
        "zeros",
        15,
        ("constant", "zero", "padded_storage"),
    ),
    CaseSpec(
        "constant_negative_n6_3d",
        (2, 6, 6),
        20,
        DEFAULT_EPS,
        "constant_negative",
        16,
        ("constant", "negative_logits", "padded_storage"),
    ),
    CaseSpec(
        "repeated_rows_n8_3d",
        (2, 8, 8),
        20,
        DEFAULT_EPS,
        "repeated_rows",
        17,
        ("structured", "repeated_rows", "naturally_aligned"),
    ),
    CaseSpec(
        "strong_diagonal_n4_3d",
        (2, 4, 4),
        20,
        DEFAULT_EPS,
        "diagonal",
        18,
        ("structured", "nonuniform", "padded_storage"),
    ),
    CaseSpec(
        "checkerboard_n6_4d",
        (2, 2, 6, 6),
        20,
        DEFAULT_EPS,
        "checkerboard",
        19,
        ("structured", "nonuniform", "padded_storage"),
    ),
    CaseSpec(
        "near_uniform_n8_3d_i100",
        (2, 8, 8),
        100,
        DEFAULT_EPS,
        "near_uniform",
        20,
        ("near_uniform", "maximum_iterations", "precision"),
    ),
    CaseSpec(
        "large_finite_n4_3d",
        (2, 4, 4),
        20,
        DEFAULT_EPS,
        "large_finite",
        21,
        ("extreme_finite", "softmax_stability", "padded_storage"),
    ),
    CaseSpec(
        "float32_extremes_n6_4d",
        (1, 2, 6, 6),
        2,
        DEFAULT_EPS,
        "float32_extremes",
        22,
        ("extreme_finite", "overflow_boundary", "padded_storage"),
    ),
    CaseSpec(
        "sparse_logits_n8_3d",
        (3, 8, 8),
        20,
        DEFAULT_EPS,
        "sparse_logits",
        23,
        ("extreme_finite", "underflow", "naturally_aligned"),
    ),
    CaseSpec(
        "eps_zero_n4_3d",
        (2, 4, 4),
        20,
        0.0,
        "random_normal",
        24,
        ("eps_variant", "eps_zero"),
    ),
    CaseSpec(
        "eps_tiny_n6_4d",
        (2, 1, 6, 6),
        20,
        1.0e-12,
        "random_uniform",
        25,
        ("eps_variant", "eps_tiny"),
    ),
    CaseSpec(
        "eps_large_n8_3d",
        (2, 8, 8),
        2,
        1.0e-1,
        "monotonic",
        26,
        ("eps_variant", "eps_large", "eps_placement"),
    ),
    CaseSpec(
        "tail_batch_41_n4_3d",
        (41, 4, 4),
        2,
        DEFAULT_EPS,
        "random_normal",
        27,
        ("batch_independence", "tiling_tail", "multi_core_candidate"),
    ),
    CaseSpec(
        "factorized_batch_3x5_n6_4d",
        (3, 5, 6, 6),
        20,
        DEFAULT_EPS,
        "random_uniform",
        28,
        ("batch_independence", "factorized_4d", "tiling_tail"),
    ),
    CaseSpec(
        "single_nan_n4_3d",
        (1, 4, 4),
        1,
        DEFAULT_EPS,
        "single_nan",
        29,
        ("nonfinite", "nan", "propagation"),
    ),
    CaseSpec(
        "batched_nan_n6_4d",
        (2, 2, 6, 6),
        2,
        DEFAULT_EPS,
        "batched_nan",
        30,
        ("nonfinite", "nan", "batch_independence", "propagation"),
    ),
    CaseSpec(
        "single_posinf_n8_3d",
        (1, 8, 8),
        1,
        DEFAULT_EPS,
        "single_posinf",
        31,
        ("nonfinite", "posinf", "propagation"),
    ),
    CaseSpec(
        "multiple_posinf_n4_3d",
        (2, 4, 4),
        20,
        DEFAULT_EPS,
        "multiple_posinf",
        32,
        ("nonfinite", "posinf", "propagation"),
    ),
    CaseSpec(
        "single_neginf_n6_3d",
        (1, 6, 6),
        2,
        DEFAULT_EPS,
        "single_neginf",
        33,
        ("nonfinite", "neginf", "contract_ambiguity"),
    ),
    CaseSpec(
        "all_neginf_row_n8_4d",
        (1, 2, 8, 8),
        1,
        DEFAULT_EPS,
        "all_neginf_row",
        34,
        ("nonfinite", "neginf", "propagation"),
    ),
    CaseSpec(
        "mixed_nonfinite_n4_4d",
        (1, 2, 4, 4),
        2,
        DEFAULT_EPS,
        "mixed_nonfinite",
        35,
        ("nonfinite", "nan", "posinf", "neginf", "propagation"),
    ),
)


INVALID_CASES = (
    InvalidCaseSpec(
        "invalid_rank2", (4, 4), "float32", 20, DEFAULT_EPS, ValueError, "rank"
    ),
    InvalidCaseSpec(
        "invalid_rank5",
        (1, 1, 1, 4, 4),
        "float32",
        20,
        DEFAULT_EPS,
        ValueError,
        "rank",
    ),
    InvalidCaseSpec(
        "invalid_nonsquare_3d",
        (2, 4, 6),
        "float32",
        20,
        DEFAULT_EPS,
        ValueError,
        "square",
    ),
    InvalidCaseSpec(
        "invalid_n5", (1, 5, 5), "float32", 20, DEFAULT_EPS, ValueError, "n"
    ),
    InvalidCaseSpec(
        "invalid_n7",
        (1, 2, 7, 7),
        "float32",
        20,
        DEFAULT_EPS,
        ValueError,
        "n",
    ),
    InvalidCaseSpec(
        "invalid_iterations_0",
        (1, 4, 4),
        "float32",
        0,
        DEFAULT_EPS,
        ValueError,
        "iterations",
    ),
    InvalidCaseSpec(
        "invalid_iterations_101",
        (1, 4, 4),
        "float32",
        101,
        DEFAULT_EPS,
        ValueError,
        "iterations",
    ),
    InvalidCaseSpec(
        "invalid_iterations_float",
        (1, 4, 4),
        "float32",
        20.0,
        DEFAULT_EPS,
        TypeError,
        "iterations_type",
    ),
    InvalidCaseSpec(
        "invalid_float16",
        (1, 4, 4),
        "float16",
        20,
        DEFAULT_EPS,
        TypeError,
        "dtype",
    ),
    InvalidCaseSpec(
        "invalid_float64",
        (1, 4, 4),
        "float64",
        20,
        DEFAULT_EPS,
        TypeError,
        "dtype",
    ),
    InvalidCaseSpec(
        "invalid_int32",
        (1, 4, 4),
        "int32",
        20,
        DEFAULT_EPS,
        TypeError,
        "dtype",
    ),
)


RUNTIME_ONLY_CHECKS = (
    {
        "name": "nd_format_only",
        "expectation": "Reject non-ND format; NPZ data cannot encode device format metadata.",
    },
    {
        "name": "output_shape_and_dtype",
        "expectation": "Output must match x shape and use float32.",
    },
    {
        "name": "optional_output_modes",
        "expectation": "Exercise output only, normOut only, sumOut only, and both optional outputs.",
    },
    {
        "name": "repeat_determinism",
        "expectation": "Repeated launches with identical input must produce identical valid values.",
    },
    {
        "name": "sum_zero_untouched",
        "expectation": "sumOut[0] is allocated but undefined and is never compared.",
    },
    {
        "name": "padding_ignored",
        "expectation": "Padding lanes do not affect softmax/reductions and are never compared.",
    },
)


def matrix_count(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape[:-2], 1)


def _vary_batch(matrices: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    for index in range(matrices.shape[0]):
        matrices[index] = np.roll(matrices[index], index, axis=0)
        matrices[index] = np.roll(matrices[index], 2 * index, axis=1)
        matrices[index] += np.float32(index * 0.03125)
    return matrices


def make_input(spec: CaseSpec) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(spec.seed)
    n = spec.shape[-1]
    count = matrix_count(spec.shape)
    pattern = spec.pattern

    if pattern == "ones":
        matrices = np.ones((count, n, n), dtype=np.float32)
    elif pattern == "zeros":
        matrices = np.zeros((count, n, n), dtype=np.float32)
    elif pattern == "constant_negative":
        matrices = np.full((count, n, n), -7.25, dtype=np.float32)
    elif pattern == "doc_nonuniform":
        base = np.array(
            [[4, 3, 2, 1], [1, 2, 3, 4], [2, 2, 2, 2], [3, 1, 4, 2]],
            dtype=np.float32,
        )
        matrices = np.repeat(base[None, ...], count, axis=0)
    elif pattern == "doc_batch":
        if count != 2 or n != 4:
            raise ValueError("doc_batch is defined only for shape (2, 4, 4)")
        nonuniform = np.array(
            [[4, 3, 2, 1], [1, 2, 3, 4], [2, 2, 2, 2], [3, 1, 4, 2]],
            dtype=np.float32,
        )
        matrices = np.stack((np.ones((4, 4), dtype=np.float32), nonuniform))
    elif pattern == "random_normal":
        matrices = rng.normal(0.0, 3.0, size=(count, n, n)).astype(np.float32)
    elif pattern == "random_uniform":
        matrices = rng.uniform(-6.0, 6.0, size=(count, n, n)).astype(np.float32)
    elif pattern == "monotonic":
        base = np.linspace(-5.0, 5.0, n * n, dtype=np.float32).reshape(n, n)
        matrices = _vary_batch(np.repeat(base[None, ...], count, axis=0))
    elif pattern == "diagonal":
        base = np.full((n, n), -4.0, dtype=np.float32)
        np.fill_diagonal(base, 5.0)
        matrices = _vary_batch(np.repeat(base[None, ...], count, axis=0))
    elif pattern == "checkerboard":
        row, col = np.indices((n, n))
        base = np.where((row + col) % 2 == 0, 7.0, -7.0).astype(np.float32)
        matrices = _vary_batch(np.repeat(base[None, ...], count, axis=0))
    elif pattern == "repeated_rows":
        row = np.linspace(-3.0, 4.0, n, dtype=np.float32)
        base = np.repeat(row[None, :], n, axis=0)
        matrices = _vary_batch(np.repeat(base[None, ...], count, axis=0))
    elif pattern == "near_uniform":
        matrices = (
            np.float32(0.5)
            + rng.normal(0.0, 1.0e-6, size=(count, n, n)).astype(np.float32)
        )
    elif pattern == "large_finite":
        base = np.linspace(-1.0e4, 1.0e4, n * n, dtype=np.float32).reshape(n, n)
        matrices = _vary_batch(np.repeat(base[None, ...], count, axis=0))
    elif pattern == "float32_extremes":
        magnitude = np.float32(np.finfo(np.float32).max / 2.0)
        values = np.array(
            [magnitude, -magnitude, np.float32(1.0e30), np.float32(-1.0e30)],
            dtype=np.float32,
        )
        base = np.resize(values, n * n).reshape(n, n)
        matrices = _vary_batch(np.repeat(base[None, ...], count, axis=0))
    elif pattern == "sparse_logits":
        base = np.full((n, n), -100.0, dtype=np.float32)
        for row in range(n):
            base[row, (3 * row + 1) % n] = np.float32(row)
        matrices = _vary_batch(np.repeat(base[None, ...], count, axis=0))
    else:
        matrices = rng.normal(0.0, 2.0, size=(count, n, n)).astype(np.float32)
        if pattern == "single_nan":
            matrices[0, 0, 1] = np.nan
        elif pattern == "batched_nan":
            matrices[0, 0, 0] = np.nan
            matrices[-1, -1, -2] = np.nan
        elif pattern == "single_posinf":
            matrices[0, 2, 3] = np.inf
        elif pattern == "multiple_posinf":
            matrices[0, 1, 0] = np.inf
            matrices[0, 1, 2] = np.inf
            matrices[-1, -1, -1] = np.inf
        elif pattern == "single_neginf":
            matrices[0, 1, 4] = -np.inf
        elif pattern == "all_neginf_row":
            matrices[0, 3, :] = -np.inf
            matrices[-1, -1, 0] = -np.inf
        elif pattern == "mixed_nonfinite":
            matrices[0, 0, 0] = np.nan
            matrices[0, 1, 1] = np.inf
            matrices[-1, 2, 2] = -np.inf
        else:
            raise ValueError(f"unknown input pattern: {pattern}")

    return np.ascontiguousarray(matrices.reshape(spec.shape), dtype=np.float32)


def make_invalid_input(spec: InvalidCaseSpec) -> np.ndarray:
    return np.ones(spec.shape, dtype=np.dtype(spec.dtype))
