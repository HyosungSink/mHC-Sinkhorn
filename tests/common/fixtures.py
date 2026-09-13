"""Fixture materialization for MhcSinkhorn device tests.

Each fixture is an immutable directory holding logits in the configured float16
or float32 dtype, reference weights in the same dtype (when verification is
requested), and a descriptor read by the ACLNN runner. Fixtures are content-addressed: identical case specs
always map to the same bytes, so cached entries are safe to reuse across
test runs and sessions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.common.contract_cases import VALID_CASES, make_input
from tests.common.mock_inputs import make_mock_logits

from tests.common.reference import (  # noqa: E402
    N_ALIGN,
    SUPPORTED_N,
    mhc_sinkhorn_reference,
)
from tests.st.scripts.artifact_cache import (  # noqa: E402
    ImmutableArtifactCache,
    sha256_file,
)

FIXTURE_FORMAT = 1


def _shape_text(shape) -> str:
    return ",".join(str(int(value)) for value in shape)


def validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("dtype", "float32") not in ("float32", "float16"):
        raise ValueError("fixture dtype must be registered by the Host schema")
    shape = tuple(int(value) for value in spec["shape"])
    if len(shape) not in (3, 4):
        raise ValueError(f"fixture shape must be rank 3 or 4: {shape}")
    if shape[-2] != shape[-1] or shape[-1] not in SUPPORTED_N:
        raise ValueError(f"fixture shape must end in an n×n matrix, n in {SUPPORTED_N}: {shape}")
    if int(spec["iterations"]) < 1 or int(spec["iterations"]) > 100:
        raise ValueError("fixture iterations must be within 1..100")
    if int(spec.get("seed", 0)) < 0:
        raise ValueError("fixture seed must be non-negative")
    if "matrix_order" in spec:
        order = spec["matrix_order"]
        matrices = int(np.prod(shape[:-2]))
        if (not isinstance(order, list) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(matrices))):
            raise ValueError("matrix_order must be a permutation of every matrix index")
    if "uniform_matrices" in spec:
        indices = spec["uniform_matrices"]
        matrices = int(np.prod(shape[:-2]))
        if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices)
                or any(i < 0 or i >= matrices for i in indices)):
            raise ValueError("uniform_matrices must contain distinct valid matrix indices")


def make_logits(spec: Mapping[str, Any]) -> np.ndarray:
    """Deterministic float32 logits for a fixture spec."""
    validate_spec(spec)
    shape = tuple(int(value) for value in spec["shape"])
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    pattern = spec.get("pattern", "uniform")
    if pattern == "uniform":
        x = rng.uniform(-4.0, 4.0, size=shape).astype(np.float32)
    elif pattern == "mock":
        x = make_mock_logits(spec["mock_input"], shape)
    elif pattern == "wide":
        x = rng.uniform(-80.0, 80.0, size=shape).astype(np.float32)
    elif pattern == "array":
        source = (REPO_ROOT / spec["input_array"]).resolve()
        if not source.is_relative_to(REPO_ROOT):
            raise ValueError("fixture array must be inside the repository")
        x = np.load(source, allow_pickle=False)
        if x.shape != shape:
            raise ValueError("array source shape differs from the fixture")
    elif pattern == "ones":
        x = np.ones(shape, dtype=np.float32)
    elif pattern == "mixed":
        # Finite values with a moderate spread; keeps softmax well conditioned.
        x = rng.normal(0.0, 2.0, size=shape).astype(np.float32)
    elif pattern == "multi_special":
        n = shape[-1]
        if shape != (81, n, n) or n not in (6, 8):
            raise ValueError("multi_special requires (81,6,6) or (81,8,8)")
        x = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
        x[0, 0, 0] = np.nan
        x[2, 1, 1] = np.inf
        x[-1, 2, 3] = -np.inf
    elif pattern == "ut":
        case = next(case for case in VALID_CASES if case.name == spec["ut_case"])
        x = make_input(case)
        if x.shape != shape:
            raise ValueError("UT source shape differs from the fixture")
    else:
        raise ValueError(f"unsupported fixture pattern: {pattern}")
    if "uniform_matrices" in spec:
        x = x.copy()
        x.reshape(-1, shape[-2], shape[-1])[spec["uniform_matrices"]] = 0
    if "matrix_order" in spec:
        x = x.reshape(-1, shape[-2], shape[-1])[spec["matrix_order"]].reshape(shape)
    return np.ascontiguousarray(x, dtype=np.float32)


def _fixture_descriptor(spec: Mapping[str, Any], verify: bool) -> str:
    shape = tuple(int(value) for value in spec["shape"])
    dtype = spec.get("dtype", "float32")
    tolerance = 1.0e-3 if dtype == "float16" else 1.0e-5
    lines = [
        f"fixture_format={FIXTURE_FORMAT}",
        f"case_id={spec['name']}",
        f"input.logits.shape={_shape_text(shape)}",
        f"input.logits.storage_shape={_shape_text(shape)}",
        f"input.logits.dtype={dtype}",
        "input.logits.file=logits.bin",
        f"iterations={int(spec['iterations'])}",
        f"eps={float(spec['eps']):.17g}",
        f"output.weights.shape={_shape_text(shape)}",
        f"output.weights.dtype={dtype}",
    ]
    if verify:
        lines.extend(
            (
                "expected.weights.file=expected_weights.bin",
                f"verify.atol={float(spec.get('atol', tolerance)):.17g}",
                f"verify.rtol={float(spec.get('rtol', tolerance)):.17g}",
            )
        )
    else:
        lines.append("verify.skip=1")
    return "\n".join(lines) + "\n"


def fixture_identity(spec: Mapping[str, Any], verify: bool | None = None) -> dict[str, Any]:
    if verify is None:
        verify = bool(spec.get("verify", True))
    identity = {
        "schema_version": 3,
        "kind": "mhc_sinkhorn_fixture",
        "case_id": str(spec["name"]),
        "shape": [int(value) for value in spec["shape"]],
        "iterations": int(spec["iterations"]),
        "eps": float(spec["eps"]),
        "dtype": spec.get("dtype", "float32"),
        "atol": spec.get("atol"),
        "rtol": spec.get("rtol"),
        "seed": int(spec.get("seed", 0)),
        "pattern": str(spec.get("pattern", "uniform")),
        "ut_case": spec.get("ut_case"),
        "numpy_version": np.__version__,
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name))
                          for name in ("fixtures.py", "contract_cases.py", "reference.py")},
        "verify": bool(verify),
    }
    if spec.get("pattern") == "mock":
        identity["mock_input"] = spec["mock_input"]
        identity["source_sha256"]["mock_inputs.py"] = sha256_file(Path(__file__).with_name("mock_inputs.py"))
    if spec.get("pattern") == "array":
        source = (REPO_ROOT / spec["input_array"]).resolve()
        if not source.is_relative_to(REPO_ROOT):
            raise ValueError("fixture array must be inside the repository")
        identity["input_array_sha256"] = sha256_file(source)
    if "matrix_order" in spec:
        validate_spec(spec)
        identity["matrix_order"] = list(spec["matrix_order"])
    if "uniform_matrices" in spec:
        validate_spec(spec)
        identity["uniform_matrices"] = list(spec["uniform_matrices"])
    return identity


def materialize_fixture(
    spec: Mapping[str, Any], work_root: Path, verify: bool | None = None
) -> tuple[Path, bool]:
    """Create (or reuse) the immutable fixture directory for a case spec."""
    if verify is None:
        verify = bool(spec.get("verify", True))
    identity = fixture_identity(spec, verify)
    cache = ImmutableArtifactCache(work_root / "fixture-cache")
    with cache.lock(identity):
        try:
            cached = cache.lookup(identity)
        except Exception:
            cache.quarantine(cache.entry_path(identity))
            cached = None
        if cached is not None:
            return cached[0], True
        entry = cache.entry_path(identity)
        if entry.exists():
            import shutil

            shutil.rmtree(entry)
        entry.mkdir(parents=True)

        logits = make_logits(spec).astype(spec.get("dtype", "float32"))
        (entry / "logits.bin").write_bytes(logits.tobytes(order="C"))
        relative = ["logits.bin"]
        if verify:
            result = mhc_sinkhorn_reference(logits.astype(np.float32), int(spec["iterations"]), float(spec["eps"]))
            expected = np.ascontiguousarray(result.output, dtype=logits.dtype)
            (entry / "expected_weights.bin").write_bytes(expected.tobytes(order="C"))
            relative.append("expected_weights.bin")
        (entry / "case.conf").write_text(_fixture_descriptor(spec, verify), encoding="utf-8")
        relative.append("case.conf")
        cache.seal(
            entry,
            identity,
            [Path(name) for name in relative],
            {
                "case_id": str(spec["name"]),
                "n_align": N_ALIGN,
                "generator": "tests.common.fixtures",
            },
        )
        return entry, False
