"""Fixture reproducibility, storage precision and cache integrity."""
import copy

import numpy as np
import pytest

from tests.common.cases import load_cases
from tests.common.fixtures import fixture_identity, make_logits, materialize_fixture
from tests.common.reference import mhc_sinkhorn_reference


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_storage_and_reference_rounding(tmp_path, dtype):
    spec = dict(name='storage', shape=[2,6,6], dtype=dtype, iterations=2,
                eps=1e-6, seed=37, pattern='uniform')
    np.testing.assert_array_equal(make_logits(spec), make_logits(spec))
    folder, cached = materialize_fixture(spec, tmp_path)
    assert not cached
    stored = np.fromfile(folder / 'logits.bin', dtype=dtype).reshape(spec['shape'])
    np.testing.assert_array_equal(stored, make_logits(spec).astype(dtype))
    golden = np.fromfile(folder / 'expected_weights.bin', dtype=dtype).reshape(spec['shape'])
    expected = mhc_sinkhorn_reference(stored.astype(np.float32), 2, 1e-6).output.astype(dtype)
    np.testing.assert_array_equal(golden, expected)
    assert materialize_fixture(spec, tmp_path) == (folder, True)


@pytest.mark.parametrize('filename', ['logits.bin', 'expected_weights.bin', 'case.conf'])
@pytest.mark.parametrize('damage', ['truncate', 'same_size', 'delete'])
def test_cache_restores_damaged_fixture(tmp_path, filename, damage):
    spec = dict(name='cache', shape=[1,4,4], iterations=2, eps=1e-6, seed=7)
    folder, _ = materialize_fixture(spec, tmp_path)
    path = folder / filename
    original = path.read_bytes()
    if damage == 'delete':
        path.unlink()
    else:
        path.write_bytes(b'bad' if damage == 'truncate' else bytes([original[0] ^ 1]) + original[1:])
    rebuilt, cached = materialize_fixture(spec, tmp_path)
    assert not cached and rebuilt == folder
    assert path.read_bytes() == original


def test_cache_separates_generation_and_comparison_parameters(tmp_path):
    base = dict(name='identity', shape=[1,4,4], dtype='float32', iterations=2, eps=1e-6, seed=7)
    original, _ = materialize_fixture(base, tmp_path)
    for variant in [dict(seed=8), dict(dtype='float16'), dict(iterations=20),
                    dict(eps=0.1), dict(atol=0.01), dict(pattern='wide'), dict(verify=False)]:
        folder, cached = materialize_fixture({**base, **variant}, tmp_path)
        assert not cached and folder != original
    assert materialize_fixture(base, tmp_path) == (original, True)


@pytest.mark.parametrize('n,digest', [
    (6, '468033690b57145eea467fb8e21b48d442b1ad78d5518e7e5e4f43940eaefaa6'),
    (8, '6054174e0d8d455d2a37ffdc43c7ad9ed39b7cd9e142be85237057cc46837fa9'),
])
def test_special_inputs_preserve_expected_bytes(n, digest):
    import hashlib
    data = make_logits(dict(shape=[81,n,n], iterations=20, pattern='multi_special', seed=817+n))
    assert hashlib.sha256(data.tobytes()).hexdigest() == digest


def test_contract_dataset_recovers_from_invalid_cache(tmp_path):
    from tests.ut.gen_data import ensure_data, check
    folder = ensure_data(tmp_path)
    assert check(folder) == []
    manifest = folder / 'manifest.json'
    before = manifest.stat().st_mtime_ns
    assert ensure_data(tmp_path) == folder
    assert manifest.stat().st_mtime_ns == before
    manifest.write_text('{}')
    assert ensure_data(tmp_path) == folder
    assert check(folder) == []


@pytest.mark.parametrize('case,digest', [(1, 'f38666fbe647212cfd4908643c61c0111ab5ab4df218dd1c1a12b4cf3bac9d0c'), (2, '91d241bbda2ef84c03c83293c7045690650dcc8637db0179bacca481ec763a5c'), (3, '390c204a99cef6a6f8438e2537d2c831bfec451d0242a2cea45e611ea9356a18'), (4, 'cd32e9c0fb3c1c6d891d4b10f842931e5605d57fff399d3c46c297fb77b5742f'), (5, 'd47eb544ba78155aca72f72113461379271ced2f99af3b2fbd6c13ca06425e58')])
def test_mock_input_bytes(case, digest):
    import hashlib
    from tests.common.cases import load_cases
    spec = next(s for s in load_cases('mock') if s['mock_input'] == case)
    data = make_logits(spec).astype(np.float16)
    assert hashlib.sha256(data.tobytes()).hexdigest() == digest


def test_mock_generator_participates_in_cache_identity(monkeypatch):
    from tests.common.cases import load_cases
    from tests.common import fixtures
    spec = load_cases('mock')[0]
    before = fixtures.fixture_identity(spec)
    original = fixtures.sha256_file
    monkeypatch.setattr(fixtures, 'sha256_file', lambda p: 'mock-generator-override' if p.name == 'mock_inputs.py' else original(p))
    assert fixtures.fixture_identity(spec) != before


def test_matrix_order_changes_cache_identity_and_preserves_complete_matrices():
    spec = copy.deepcopy(load_cases('mock')[-1])
    x = make_logits(spec).reshape(16, 6, 6)
    original = fixture_identity(spec)
    spec['matrix_order'] = list(reversed(range(16)))
    assert fixture_identity(spec) != original
    reordered = make_logits(spec).reshape(16, 6, 6)
    assert np.array_equal(reordered[0], x[-1])
    assert sorted(m.tobytes() for m in reordered) == sorted(m.tobytes() for m in x)


@pytest.mark.parametrize('order', [[0] * 16, list(range(15)), list(range(1, 17)),
                                 [False] + list(range(1, 16))])
def test_invalid_matrix_order_cannot_silently_change_the_workload(order):
    spec = dict(load_cases('mock')[-1], matrix_order=order)
    with pytest.raises(ValueError, match='matrix_order'):
        make_logits(spec)


def test_uniform_matrix_override_is_local_and_changes_cache_identity():
    spec = dict(load_cases('mock')[-1])
    original = make_logits(spec).reshape(16, 6, 6)
    variant = dict(spec, uniform_matrices=[1, 15])
    modified = make_logits(variant).reshape(16, 6, 6)
    assert fixture_identity(variant) != fixture_identity(spec)
    assert np.all(modified[[1, 15]] == 0)
    np.testing.assert_array_equal(modified[[0, 2, 14]], original[[0, 2, 14]])


@pytest.mark.parametrize('indices', [[-1], [16], [1, 1], [False], '1'])
def test_invalid_uniform_matrix_indices(indices):
    with pytest.raises(ValueError, match='uniform_matrices'):
        make_logits(dict(load_cases('mock')[-1], uniform_matrices=indices))
