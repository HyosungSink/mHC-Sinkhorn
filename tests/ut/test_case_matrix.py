"""Suite selection, workload difficulty and local test infrastructure."""
import hashlib
import json
import math
from collections import Counter

import numpy as np
import pytest
from tests.common.cases import load_cases
from tests.common.fixtures import _fixture_descriptor, make_logits, validate_spec
from tests.common.reference import mhc_sinkhorn_reference


def test_suites_are_disjoint_and_full_excludes_mock():
    suites = {key: {c['name'] for c in load_cases(key)} for key in ('coverage', 'correctness', 'mock')}
    assert not (suites['coverage'] & suites['correctness'])
    assert not (suites['mock'] & (suites['coverage'] | suites['correctness']))
    assert {c['name'] for c in load_cases('full')} == suites['coverage'] | suites['correctness']
    assert len(load_cases('custom', [next(iter(suites['mock']))])) == 1
    with pytest.raises(ValueError):
        load_cases('coverage', [next(iter(suites['mock']))])
    with pytest.raises(ValueError):
        load_cases('custom')


@pytest.mark.parametrize('spec', load_cases('full') + load_cases('mock'), ids=lambda s: s['name'])
def test_fixture_inputs_remain_loadable(spec):
    validate_spec(spec)
    assert make_logits(spec).shape == tuple(spec['shape'])


def test_coverage_includes_schema_types_and_iteration_boundaries():
    cases = load_cases('coverage')
    for dtype in ('float16', 'float32'):
        for n in (4, 6, 8):
            for iterations in (1, 2, 20, 100):
                assert any(c['dtype'] == dtype and c['shape'][-1] == n and c['iterations'] == iterations for c in cases)
    assert {len(c['shape']) for c in cases} == {3, 4}


def test_sensitivity_is_a_correctness_subset_and_tags_do_not_fall_back():
    cases = load_cases('correctness', tags=['sensitivity'])
    assert len(cases) == 3
    assert all(c['measurement'] == 'correctness' and c['verify'] for c in cases)
    assert {c['shape'][-1] for c in cases} == {4, 6, 8}
    with pytest.raises(ValueError, match='No cases'):
        load_cases('coverage', tags=['sensitivity'])
    with pytest.raises(ValueError, match='No cases'):
        load_cases('correctness', tags=['sensitivity', 'missing'])


def test_submission_sources_exclude_unlisted_local_headers(tmp_path, monkeypatch):
    import json
    from tests.st.scripts import build_ops
    code = tmp_path / 'code'
    code.mkdir()
    (code / 'entry.cpp').write_text('#include "not_uploaded.h"\n')
    (code / 'not_uploaded.h').write_text('')
    (tmp_path / 'template-manifest.json').write_text(json.dumps({'files': [{'path': 'entry.cpp'}]}))
    monkeypatch.setattr(build_ops, 'REPO_ROOT', tmp_path)
    assert build_ops.working_source_files(True) == [code / 'entry.cpp']
    assert len(build_ops.working_source_files(False)) == 2
    (tmp_path / 'template-manifest.json').write_text(json.dumps({'files': [{'path': '../escape.h'}]}))
    with pytest.raises(build_ops.MockFailure):
        build_ops.working_source_files(True)


def test_infrastructure_failure_cannot_leave_a_passing_report(tmp_path, monkeypatch):
    import json
    from tests.st.scripts import run_tests
    report = tmp_path / 'run.tsv'
    report.with_suffix('.json').write_text('{"passed":true,"status":"Pass"}')
    def failure(_argv):
        raise run_tests.MockFailure('device unavailable')
    monkeypatch.setattr(run_tests, '_main', failure)
    assert run_tests.main(['--report', str(report)]) == 2
    result = json.loads(report.with_suffix('.json').read_text())
    assert result['status'] == 'INCONCLUSIVE' and not result['passed']


def test_mock_workload_configuration():
    protocols = set()
    for spec in load_cases('mock'):
        validate_spec(spec)
        assert spec['verify'] and spec['measurement'] == 'kernel'
        assert all(math.isfinite(spec[key]) and spec[key] >= 0 for key in ('atol', 'rtol'))
        repeat = spec.get('repeat', 1)
        warmup = spec.get('warmup', 4)
        warmup_us = spec.get('warmup_us', 0)
        assert type(repeat) is int and repeat > 0
        assert type(warmup) is int and warmup >= 0
        assert type(warmup_us) is int and warmup_us >= 0
        protocols.add((warmup, repeat, warmup_us))
    assert len(protocols) == 1


def matrix_errors(actual, expected, spec):
    actual = actual.astype(spec['dtype']).astype(np.float64)
    expected = expected.astype(spec['dtype']).astype(np.float64)
    mismatch = np.abs(actual - expected) > spec['atol'] + spec['rtol'] * np.abs(expected)
    return np.count_nonzero(mismatch.reshape(-1, spec['shape'][-1] ** 2), axis=1)


@pytest.mark.parametrize('spec', load_cases('mock'), ids=lambda s: s['name'])
def test_mock_requires_more_than_one_normalization_round(spec):
    x = make_logits(spec).astype(spec['dtype']).astype(np.float32)
    reference = mhc_sinkhorn_reference(x, spec['iterations'], spec['eps'])
    # Keep this difficulty check independent of the case's comparison allowance.
    errors = matrix_errors(reference.norm_out[1], reference.output,
                           dict(spec, atol=1e-3, rtol=1e-3))
    assert np.any(errors > 0), 'one normalization round must not solve the whole workload'


def test_easy_first_batch_exposes_premature_batch_exit():
    spec = load_cases('correctness', tags=['convergence_order'])[0]
    assert spec['name'] == 'convergence_half_n6_easy_first'
    x = make_logits(spec).astype(spec['dtype']).astype(np.float32)
    reference = mhc_sinkhorn_reference(x, spec['iterations'], spec['eps'])
    errors = matrix_errors(reference.norm_out[3], reference.output, spec)
    assert errors[0] == 0
    assert np.any(errors[1:4] > 0), 'later matrices in the aligned batch must still need work'


@pytest.mark.parametrize('suite,counts,digest', [
    ('coverage', {'float16': 15, 'float32': 15},
     '44d97f2ea8326f0f4f2c0a60dc53f1d09b5fd5e37b7dd808a05009af60fcb846'),
    ('correctness', {'float16': 47, 'float32': 58},
     'fb7657924e42f200c365d32a87fa7b48ec4804e32ed018aef3ee93d5c9c20253'),
    ('mock', {'float16': 5},
     'f61b11a23f454a982e745a719a27e31c11955d448cb1b60b44998ba620a04e11'),
], ids=['coverage', 'correctness', 'mock'])
def test_device_suite_inventory(suite, counts, digest):
    """Case names and dtypes define each suite's complete inventory."""
    cases = load_cases(suite)
    assert Counter(spec['dtype'] for spec in cases) == counts
    inventory = sorted((spec['name'], spec['dtype']) for spec in cases)
    encoded = json.dumps(inventory, separators=(',', ':')).encode('utf-8')
    assert hashlib.sha256(encoded).hexdigest() == digest, f'{suite}: case inventory mismatch'


def test_device_suite_tolerances_match_configuration():
    mock_tolerances = {
        'mock_case_1': 0.01, 'mock_case_2': 0.02, 'mock_case_3': 0.01,
        'mock_case_4': 0.01, 'mock_case_5': 0.01,
    }
    for spec in load_cases('full') + load_cases('mock'):
        assert spec.get('verify', True) is True
        expected = mock_tolerances.get(spec['name'], 0.001)
        descriptor = dict(line.split('=', 1) for line in _fixture_descriptor(spec, True).splitlines())
        for key in ('atol', 'rtol'):
            actual = float(descriptor[f'verify.{key}'])
            assert actual == expected, f'{spec["name"]}: {key} must be {expected}, got {actual}'
