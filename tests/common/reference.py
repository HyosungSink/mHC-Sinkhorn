"""Float32 reference implementation for the documented MhcSinkhorn contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


SUPPORTED_N = (4, 6, 8)
N_ALIGN = 8


@dataclass(frozen=True)
class SinkhornResult:
    output: npt.NDArray[np.float32]
    norm_out: npt.NDArray[np.float32]
    sum_out: npt.NDArray[np.float32]
    norm_out_physical: npt.NDArray[np.float32]
    sum_out_physical: npt.NDArray[np.float32]
    norm_valid_mask: npt.NDArray[np.bool_]
    sum_valid_mask: npt.NDArray[np.bool_]


def validate_arguments(x: np.ndarray, num_iters: int, eps: float) -> None:
    """Validate only constraints explicitly stated by the operator document."""
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array")
    if x.dtype != np.float32:
        raise TypeError("x must have dtype float32")
    if x.ndim not in (3, 4):
        raise ValueError("x must have rank 3 or 4")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("the last two dimensions of x must form a square matrix")
    if x.shape[-1] not in SUPPORTED_N:
        raise ValueError("the matrix order must be one of 4, 6, or 8")
    if isinstance(num_iters, (bool, np.bool_)) or not isinstance(
        num_iters, (int, np.integer)
    ):
        raise TypeError("num_iters must be an integer")
    if not 1 <= int(num_iters) <= 100:
        raise ValueError("num_iters must be in [1, 100]")
    if np.asarray(eps).ndim != 0:
        raise TypeError("eps must be a scalar")


def _softmax_last_axis(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # Keep each NumPy operation in float32 to match the kernel accumulation contract.
    row_max = np.max(x, axis=-1, keepdims=True)
    shifted = np.subtract(x, row_max, dtype=np.float32)
    exponentials = np.exp(shifted, dtype=np.float32)
    denominator = np.sum(
        exponentials, axis=-1, keepdims=True, dtype=np.float32
    )
    return np.divide(exponentials, denominator, dtype=np.float32)


def mhc_sinkhorn_reference(
    x: npt.NDArray[np.float32], num_iters: int, eps: float
) -> SinkhornResult:
    """Compute final and intermediate tensors exactly in the documented order.

    ``sum_out[0]`` and all physical padding slots are filled with NaN only as
    host-side sentinels. Their validity masks are false and implementations must
    not be compared against those sentinel values.
    """
    validate_arguments(x, num_iters, eps)
    iterations = int(num_iters)
    eps32 = np.float32(eps)
    leading_shape = x.shape[:-2]
    n = x.shape[-1]
    step_count = 2 * iterations

    norm_out = np.empty((step_count,) + x.shape, dtype=np.float32)
    sum_out = np.full(
        (step_count,) + leading_shape + (n,), np.nan, dtype=np.float32
    )

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        current = np.add(_softmax_last_axis(x), eps32, dtype=np.float32)
        norm_out[0] = current

        column_sum = np.add(
            np.sum(current, axis=-2, dtype=np.float32), eps32, dtype=np.float32
        )
        sum_out[1] = column_sum
        current = np.divide(
            current, np.expand_dims(column_sum, axis=-2), dtype=np.float32
        )
        norm_out[1] = current

        for iteration in range(1, iterations):
            row_step = 2 * iteration
            row_sum = np.add(
                np.sum(current, axis=-1, dtype=np.float32), eps32, dtype=np.float32
            )
            sum_out[row_step] = row_sum
            current = np.divide(
                current, np.expand_dims(row_sum, axis=-1), dtype=np.float32
            )
            norm_out[row_step] = current

            column_step = row_step + 1
            column_sum = np.add(
                np.sum(current, axis=-2, dtype=np.float32),
                eps32,
                dtype=np.float32,
            )
            sum_out[column_step] = column_sum
            current = np.divide(
                current, np.expand_dims(column_sum, axis=-2), dtype=np.float32
            )
            norm_out[column_step] = current

    norm_out_physical = np.full(
        (step_count,) + leading_shape + (n, N_ALIGN), np.nan, dtype=np.float32
    )
    norm_out_physical[..., :n] = norm_out
    norm_valid_mask = np.zeros(norm_out_physical.shape, dtype=np.bool_)
    norm_valid_mask[..., :n] = True

    sum_out_physical = np.full(
        (step_count,) + leading_shape + (N_ALIGN,), np.nan, dtype=np.float32
    )
    sum_out_physical[..., :n] = sum_out
    sum_valid_mask = np.zeros(sum_out_physical.shape, dtype=np.bool_)
    sum_valid_mask[1:, ..., :n] = True

    return SinkhornResult(
        output=np.ascontiguousarray(norm_out[-1]),
        norm_out=np.ascontiguousarray(norm_out),
        sum_out=np.ascontiguousarray(sum_out),
        norm_out_physical=np.ascontiguousarray(norm_out_physical),
        sum_out_physical=np.ascontiguousarray(sum_out_physical),
        norm_valid_mask=np.ascontiguousarray(norm_valid_mask),
        sum_valid_mask=np.ascontiguousarray(sum_valid_mask),
    )
