"""Coverage audit for the MhcSinkhorn test matrix."""

from __future__ import annotations

from collections import Counter

from tests.common.contract_cases import (
    ALL_OUTPUT_MODES,
    INVALID_CASES,
    RUNTIME_ONLY_CHECKS,
    VALID_CASES,
    matrix_count,
)


def audit_coverage() -> list[str]:
    issues: list[str] = []
    ranks = {len(case.shape) for case in VALID_CASES}
    orders = {case.shape[-1] for case in VALID_CASES}
    iterations = {case.iterations for case in VALID_CASES}
    tags = {tag for case in VALID_CASES for tag in case.tags}
    output_modes = {mode for case in VALID_CASES for mode in case.runtime_modes}
    invalid_reasons = {case.reason for case in INVALID_CASES}
    tag_counts = Counter(tag for case in VALID_CASES for tag in case.tags)

    requirements = (
        (ranks == {3, 4}, f"expected ranks {{3, 4}}, got {sorted(ranks)}"),
        (orders == {4, 6, 8}, f"expected n {{4, 6, 8}}, got {sorted(orders)}"),
        (
            {1, 2, 20, 100}.issubset(iterations),
            f"missing iteration boundary from {sorted(iterations)}",
        ),
        (
            set(ALL_OUTPUT_MODES).issubset(output_modes),
            f"missing optional-output mode from {sorted(output_modes)}",
        ),
        (
            {"constant", "nonuniform", "extreme_finite"}.issubset(tags),
            "missing a required finite-value category",
        ),
        (
            {"nan", "posinf", "neginf"}.issubset(tags),
            "missing a required non-finite category",
        ),
        (
            all(tag_counts[tag] >= 2 for tag in ("nan", "posinf", "neginf")),
            "each non-finite category must have at least two independent cases",
        ),
        (
            {"eps_zero", "eps_tiny", "eps_large"}.issubset(tags),
            "missing an eps placement/scale category",
        ),
        (
            any(matrix_count(case.shape) == 1 for case in VALID_CASES)
            and any(matrix_count(case.shape) > 1 for case in VALID_CASES)
            and any(matrix_count(case.shape) == 41 for case in VALID_CASES),
            "missing singleton, multi-matrix, or tail batch coverage",
        ),
        (
            {"rank", "square", "n", "iterations", "iterations_type", "dtype"}
            .issubset(invalid_reasons),
            "missing a documented invalid-input category",
        ),
        (
            {case.shape[-1] for case in VALID_CASES if "padded_storage" in case.tags}
            == {4, 6}
            and any(case.shape[-1] == 8 for case in VALID_CASES),
            "missing padded or naturally aligned physical-layout coverage",
        ),
        (
            sum("document_example" in case.tags for case in VALID_CASES) == 3,
            "all three documented examples must be represented",
        ),
        (
            len(RUNTIME_ONLY_CHECKS) >= 6,
            "runtime-only contract checks are incomplete",
        ),
    )
    issues.extend(message for passed, message in requirements if not passed)

    for n in (4, 6, 8):
        seen = {case.iterations for case in VALID_CASES if case.shape[-1] == n}
        missing = {1, 2, 20, 100} - seen
        if missing:
            issues.append(f"n={n} is missing iteration values {sorted(missing)}")
        seen_ranks = {len(case.shape) for case in VALID_CASES if case.shape[-1] == n}
        if seen_ranks != {3, 4}:
            issues.append(f"n={n} is missing rank coverage; got {sorted(seen_ranks)}")

    names = [case.name for case in VALID_CASES] + [
        case.name for case in INVALID_CASES
    ]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        issues.append(f"duplicate case names: {duplicates}")
    return issues


def coverage_summary() -> dict[str, object]:
    return {
        "valid_cases": len(VALID_CASES),
        "invalid_cases": len(INVALID_CASES),
        "ranks": sorted({len(case.shape) for case in VALID_CASES}),
        "matrix_orders": sorted({case.shape[-1] for case in VALID_CASES}),
        "iterations": sorted({case.iterations for case in VALID_CASES}),
        "eps_values": sorted({case.eps for case in VALID_CASES}),
        "largest_matrix_batch": max(matrix_count(case.shape) for case in VALID_CASES),
        "runtime_only_checks": len(RUNTIME_ONLY_CHECKS),
    }


if __name__ == "__main__":
    import json

    problems = audit_coverage()
    print(json.dumps(coverage_summary(), indent=2, sort_keys=True))
    if problems:
        print("Coverage audit failed:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("Coverage audit passed.")
