from __future__ import annotations

import numpy as np
import pytest

from tests.common.contract_cases import INVALID_CASES, VALID_CASES, make_input, make_invalid_input
from tests.ut.coverage_audit import audit_coverage
from tests.ut.gen_data import ensure_data, check
from tests.common.reference import N_ALIGN, mhc_sinkhorn_reference


@pytest.mark.parametrize("case", VALID_CASES, ids=lambda case: case.name)
def test_reference_shapes_indexing_and_physical_layout(case) -> None:
    x = make_input(case)
    result = mhc_sinkhorn_reference(x, case.iterations, case.eps)
    steps = 2 * case.iterations
    n = case.shape[-1]
    leading = case.shape[:-2]

    assert result.output.shape == case.shape
    assert result.output.dtype == np.float32
    assert result.norm_out.shape == (steps,) + case.shape
    assert result.sum_out.shape == (steps,) + leading + (n,)
    assert result.norm_out_physical.shape == (steps,) + leading + (n, N_ALIGN)
    assert result.sum_out_physical.shape == (steps,) + leading + (N_ALIGN,)
    matrix_count = int(np.prod(leading))
    assert result.norm_out_physical.size == steps * matrix_count * n * N_ALIGN
    assert result.sum_out_physical.size == steps * matrix_count * N_ALIGN
    np.testing.assert_array_equal(result.output, result.norm_out[-1])

    assert np.all(result.norm_valid_mask[..., :n])
    assert not np.any(result.norm_valid_mask[..., n:])
    assert not np.any(result.sum_valid_mask[0])
    assert np.all(result.sum_valid_mask[1:, ..., :n])
    assert not np.any(result.sum_valid_mask[..., n:])
    np.testing.assert_array_equal(
        result.norm_out_physical[..., :n], result.norm_out
    )
    np.testing.assert_array_equal(result.sum_out_physical[..., :n], result.sum_out)


@pytest.mark.parametrize("case", VALID_CASES, ids=lambda case: case.name)
def test_initial_softmax_and_eps_placement(case) -> None:
    x = make_input(case)
    result = mhc_sinkhorn_reference(x, case.iterations, case.eps)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        row_max = np.max(x, axis=-1, keepdims=True)
        exponentials = np.exp(
            np.subtract(x, row_max, dtype=np.float32), dtype=np.float32
        )
        softmax = np.divide(
            exponentials,
            np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32),
            dtype=np.float32,
        )
        expected = np.add(softmax, np.float32(case.eps), dtype=np.float32)
    np.testing.assert_array_equal(result.norm_out[0], expected)


@pytest.mark.parametrize("case", VALID_CASES, ids=lambda case: case.name)
def test_every_intermediate_step_obeys_documented_recurrence(case) -> None:
    x = make_input(case)
    result = mhc_sinkhorn_reference(x, case.iterations, case.eps)
    eps32 = np.float32(case.eps)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for step in range(1, 2 * case.iterations):
            if step % 2 == 0:
                expected_sum = np.add(
                    np.sum(result.norm_out[step - 1], axis=-1, dtype=np.float32),
                    eps32,
                    dtype=np.float32,
                )
                expected_norm = np.divide(
                    result.norm_out[step - 1],
                    np.expand_dims(expected_sum, axis=-1),
                    dtype=np.float32,
                )
            else:
                expected_sum = np.add(
                    np.sum(result.norm_out[step - 1], axis=-2, dtype=np.float32),
                    eps32,
                    dtype=np.float32,
                )
                expected_norm = np.divide(
                    result.norm_out[step - 1],
                    np.expand_dims(expected_sum, axis=-2),
                    dtype=np.float32,
                )
            np.testing.assert_array_equal(result.sum_out[step], expected_sum)
            np.testing.assert_array_equal(result.norm_out[step], expected_norm)


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in VALID_CASES
        if case.iterations >= 20
        and case.eps <= 1.0e-6
        and np.isfinite(make_input(case)).all()
    ],
    ids=lambda case: case.name,
)
def test_finite_recommended_iterations_approach_doubly_stochastic(case) -> None:
    output = mhc_sinkhorn_reference(
        make_input(case), case.iterations, case.eps
    ).output
    row_error = np.max(
        np.abs(np.sum(output, axis=-1, dtype=np.float32) - np.float32(1.0))
    )
    column_error = np.max(
        np.abs(np.sum(output, axis=-2, dtype=np.float32) - np.float32(1.0))
    )

    # Every final step is a column normalization. Twenty iterations are a
    # convergence target rather than an exact row-normalization guarantee.
    assert column_error <= 3.0e-6
    assert row_error <= (5.0e-5 if case.iterations == 100 else 3.0e-2)


@pytest.mark.parametrize("case", VALID_CASES, ids=lambda case: case.name)
def test_each_batched_matrix_is_independent(case) -> None:
    x = make_input(case)
    batched = mhc_sinkhorn_reference(x, case.iterations, case.eps)
    flat_x = x.reshape((-1,) + x.shape[-2:])
    flat_output = batched.output.reshape((-1,) + x.shape[-2:])

    for index, matrix in enumerate(flat_x):
        single = mhc_sinkhorn_reference(matrix[None, ...], case.iterations, case.eps)
        np.testing.assert_array_equal(flat_output[index], single.output[0])


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda case: case.name)
def test_invalid_documented_arguments_are_rejected(case) -> None:
    x = make_invalid_input(case)
    with pytest.raises(case.expected_exception):
        mhc_sinkhorn_reference(x, case.iterations, case.eps)


def test_document_nonuniform_example() -> None:
    case = next(case for case in VALID_CASES if case.name == "doc_nonuniform_n4_3d_i20")
    result = mhc_sinkhorn_reference(make_input(case), case.iterations, case.eps)
    documented = np.array(
        [
            [0.5248, 0.3838, 0.0618, 0.0297],
            [0.0281, 0.1517, 0.1804, 0.6399],
            [0.2003, 0.3981, 0.1742, 0.2274],
            [0.2469, 0.0664, 0.5836, 0.1031],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result.output[0], documented, rtol=0.0, atol=5.0e-5)


def test_reference_is_bitwise_deterministic() -> None:
    case = next(case for case in VALID_CASES if case.name == "tail_batch_41_n4_3d")
    x = make_input(case)
    first = mhc_sinkhorn_reference(x, case.iterations, case.eps)
    second = mhc_sinkhorn_reference(x, case.iterations, case.eps)
    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_generated_files_match_reference() -> None:
    assert check(ensure_data()) == []


def test_document_coverage_is_complete() -> None:
    assert audit_coverage() == []
