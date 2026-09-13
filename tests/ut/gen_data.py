"""Generate reproducible MhcSinkhorn NPZ inputs and golden outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tests.common.contract_cases import (
    INVALID_CASES,
    RUNTIME_ONLY_CHECKS,
    VALID_CASES,
    make_input,
    make_invalid_input,
)
from tests.common.reference import mhc_sinkhorn_reference


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path("/tmp/cannjudge/mhcsinkhorn/contract-data")


def valid_payload(case) -> dict[str, np.ndarray]:
    x = make_input(case)
    result = mhc_sinkhorn_reference(x, case.iterations, case.eps)
    return {
        "x": x,
        "output": result.output,
        "norm_out": result.norm_out,
        "sum_out": result.sum_out,
        "norm_out_physical": result.norm_out_physical,
        "sum_out_physical": result.sum_out_physical,
        "norm_valid_mask": result.norm_valid_mask,
        "sum_valid_mask": result.sum_valid_mask,
        "num_iters": np.asarray(case.iterations, dtype=np.int64),
        "eps": np.asarray(case.eps, dtype=np.float32),
    }


def invalid_payload(case) -> dict[str, np.ndarray]:
    return {
        "x": make_invalid_input(case),
        "num_iters": np.asarray(case.iterations),
        "eps": np.asarray(case.eps),
        "expected_exception": np.asarray(case.expected_exception.__name__),
        "reason": np.asarray(case.reason),
    }


def build_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "oracle": "B-medium-mHC-Sinkhorn.md section 3 float32 operation order",
        "physical_layout": {
            "norm_out": "[2*num_iters, *leading, n, 8]",
            "sum_out": "[2*num_iters, *leading, 8]",
            "comparison": "Use *_valid_mask; sum_out[0] and padding are undefined.",
        },
        "comparison_policy": {
            "scope": "CPU reference self-check only; not a device acceptance suite",
            "device_acceptance": "tests/st/cases/coverage.json, correctness.json and mock.json",
        },
        "valid_cases": [
            {
                "name": case.name,
                "file": f"valid/{case.name}.npz",
                "shape": list(case.shape),
                "dtype": "float32",
                "iterations": case.iterations,
                "eps": case.eps,
                "pattern": case.pattern,
                "seed": case.seed,
                "tags": list(case.tags),
                "runtime_modes": list(case.runtime_modes),
            }
            for case in VALID_CASES
        ],
        "invalid_cases": [
            {
                "name": case.name,
                "file": f"invalid/{case.name}.npz",
                "shape": list(case.shape),
                "dtype": case.dtype,
                "iterations": case.iterations,
                "eps": case.eps,
                "expected_exception": case.expected_exception.__name__,
                "reason": case.reason,
            }
            for case in INVALID_CASES
        ],
        "runtime_only_checks": list(RUNTIME_ONLY_CHECKS),
        "deferred": [
            "mask semantics are unspecified",
            "float16 is present only in the provisional Host schema, not the formal task",
            "the final public ABI for normOut/sumOut is not confirmed",
        ],
        "contract_ambiguity": (
            "A lone -inf among finite logits evaluates to zero under the documented "
            "stable-softmax formula, while the shape-constraints prose says a non-finite "
            "input's corresponding output is NaN. Golden data follows the explicit formula."
        ),
    }


def _arrays_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        return False
    if np.issubdtype(expected.dtype, np.inexact):
        return bool(np.array_equal(expected, actual, equal_nan=True))
    return bool(np.array_equal(expected, actual))


def check_npz(path: Path, expected: dict[str, np.ndarray]) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"missing {path}"]
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(expected):
            problems.append(
                f"{path}: keys {sorted(stored.files)} != {sorted(expected)}"
            )
            return problems
        for key, expected_array in expected.items():
            if not _arrays_equal(expected_array, stored[key]):
                problems.append(f"{path}: field {key} differs from regenerated data")
    return problems


def generate(output_dir: Path) -> None:
    valid_dir = output_dir / "valid"
    invalid_dir = output_dir / "invalid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    for case in VALID_CASES:
        np.savez_compressed(valid_dir / f"{case.name}.npz", **valid_payload(case))
    for case in INVALID_CASES:
        np.savez_compressed(invalid_dir / f"{case.name}.npz", **invalid_payload(case))

    manifest = build_manifest()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def check(output_dir: Path) -> list[str]:
    problems: list[str] = []
    for case in VALID_CASES:
        path = output_dir / "valid" / f"{case.name}.npz"
        problems.extend(check_npz(path, valid_payload(case)))
    for case in INVALID_CASES:
        path = output_dir / "invalid" / f"{case.name}.npz"
        problems.extend(check_npz(path, invalid_payload(case)))

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        problems.append(f"missing {manifest_path}")
    else:
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest != build_manifest():
            problems.append(f"{manifest_path}: differs from regenerated manifest")
    return problems



def ensure_data(cache_root: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Return a validated, source-addressed dataset; generate it on a cache miss."""
    from tests.st.scripts.artifact_cache import ImmutableArtifactCache, sha256_file
    cache = ImmutableArtifactCache(cache_root)
    sources = [Path(__file__), ROOT.parent / "common/contract_cases.py",
               ROOT.parent / "common/reference.py"]
    identity = {"kind": "contract-data", "numpy": np.__version__,
                "sources": {p.name: sha256_file(p) for p in sources}}
    with cache.lock(identity):
        entry = cache.entry_path(identity)
        try:
            hit = cache.lookup(identity)
        except (ValueError, OSError, RuntimeError):
            hit = None
        if hit is not None:
            return hit[0]
        if entry.exists():
            cache.quarantine(entry)
        generate(entry)
        cache.seal(entry, identity,
                   [p.relative_to(entry) for p in entry.rglob("*") if p.is_file()],
                   {"generator": "tests.ut.gen_data"})
        return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, help="explicit export directory; default uses the external cache")
    parser.add_argument(
        "--check", action="store_true", help="verify generated data against the reference"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else ensure_data()
    if args.check:
        errors = check(output_dir)
        if errors:
            for error in errors:
                print(error)
            raise SystemExit(1)
        print(f"Verified {len(VALID_CASES)} valid and {len(INVALID_CASES)} invalid cases.")
    else:
        if args.output_dir is not None:
            generate(output_dir)
        print(f"Generated {len(VALID_CASES)} valid and {len(INVALID_CASES)} invalid cases.")
