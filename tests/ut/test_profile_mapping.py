"""Case/report identity, capture retries and complete MSPTI replica statistics."""
import subprocess

import pytest

from tests.st.scripts import build_ops as ops

HEADER = 'case_id\tstatus\tstage\tacl_status\tdevice_us_p50\twarmup\trepeat\tworkspace\tmin\tmax\tmean\tmode\tinner\tsamples\tmismatches\telements\n'


def row(name, status='Passed', stage='complete', mean=10):
    return f'{name}\t{status}\t{stage}\t0\t{mean}\t2\t30\t0\t1\t100\t{mean}\tmspti\t1\t30\t0\t16\n'


def test_report_maps_by_case_identifier_and_rejects_duplicates(tmp_path):
    report = tmp_path / 'run.tsv'
    report.write_text(HEADER + row('second', mean=20) + row('first', mean=10))
    parsed = ops.parse_runner_report(report)
    assert parsed['first']['local_elapsed_us'] == 10
    assert parsed['second']['local_elapsed_us'] == 20
    report.write_text(report.read_text() + row('first'))
    with pytest.raises(ops.MockFailure, match='duplicate'):
        ops.parse_runner_report(report)


def test_batch_retries_only_incomplete_capture(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        name = command[-1]
        calls.append(name)
        incomplete = name == 'first' and calls.count(name) == 1
        status = 'Runner Error' if incomplete else ('Wrong Answer' if name == 'second' else 'Passed')
        stage = 'mspti_incomplete_capture' if incomplete else 'complete'
        from pathlib import Path
        Path(command[command.index('--report') + 1]).write_text(HEADER + row(name, status, stage))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(ops.subprocess, 'run', run)
    report = tmp_path / 'run.tsv'
    result = ops.run_fixture_batch(['runner', '--report', str(report)], ['first','second'], report, {})
    assert calls == ['first','first','second']
    assert result['first']['capture_attempts'] == 2
    assert result['second']['local_status'] == 'Wrong Answer'
    assert set(ops.parse_runner_report(report)) == {'first','second'}


def test_replica_statistics_keep_slow_measurements(tmp_path):
    report = tmp_path / 'run.tsv'
    report.write_text(HEADER + ''.join(row(str(i), mean=v) for i,v in enumerate([10,1000,20])))
    replicas = list(ops.parse_runner_report(report).values())
    summary = ops.summarize_replica_rows(replicas, warmup=2, repeat=30)
    assert summary['collection_complete'] and summary['sample_count'] == 90
    assert summary['local_elapsed_us'] == 20
    assert summary['replicas'][1]['us_mean'] == 1000
    for variant in [dict(sample_count=29), dict(timing_mode='sync'), dict(warmup=1),
                    dict(us_mean=float('nan')), dict(us_mean=0)]:
        invalid = [replicas[0], {**replicas[1], **variant}, replicas[2]]
        summary = ops.summarize_replica_rows(invalid, warmup=2, repeat=30)
        assert not summary['collection_complete'] and summary['local_elapsed_us'] is None


def test_time_warmup_requires_elapsed_budget_and_keeps_actual_launch_counts(tmp_path):
    report = tmp_path / 'run.tsv'
    report.write_text(HEADER.rstrip('\n') + '\twarmup_elapsed_us\n' +
                      ''.join(row(str(i), mean=v).rstrip('\n') + f'\t{elapsed}\n'
                              for i, (v, elapsed) in enumerate([(10,260), (1000,275), (20,290)])))
    replicas = list(ops.parse_runner_report(report).values())
    for observed, count in zip(replicas, [5,6,7]):
        observed['warmup'] = count
    summary = ops.summarize_replica_rows(replicas, warmup=1, repeat=30, warmup_us=260)
    assert summary['collection_complete'] and summary['local_elapsed_us'] == 20
    assert [r['warmup'] for r in summary['replicas']] == [5,6,7]
    for invalid in [dict(warmup=0), dict(warmup_elapsed_us=259.99),
                    dict(warmup_elapsed_us=float('nan')), dict(warmup_elapsed_us=float('inf')),
                    dict(sample_count=29)]:
        changed = [replicas[0], {**replicas[1], **invalid}, replicas[2]]
        assert not ops.summarize_replica_rows(changed, 1, 30, 260)['collection_complete']


def test_negative_warmup_duration_is_rejected_before_execution():
    from tests.st.scripts.run_tests import _main
    with pytest.raises(ValueError, match='measurement protocol'):
        _main(['--suite', 'mock', '--list', '--warmup-us', '-1'])


@pytest.mark.parametrize('misidentify', [False, True])
def test_case_repeat_counts_follow_fixture_identity(tmp_path, monkeypatch, misidentify):
    fixtures = []
    for index, name in enumerate(['second', 'first']):
        folder = tmp_path / f'fixture-{index}'
        folder.mkdir()
        (folder / 'case.conf').write_text(f'case_id={name}\n')
        fixtures.append(folder)
    calls = []
    def run(command, **kwargs):
        from pathlib import Path
        name = (Path(command[-1]) / 'case.conf').read_text().strip().split('=')[1]
        repeat = int(command[command.index('--repeat') + 1])
        warmup_us = int(command[command.index('--warmup-us') + 1])
        calls.append((name, repeat, warmup_us))
        fields = row('unexpected' if misidentify else name).strip().split('\t')
        fields[6] = fields[13] = str(repeat)
        Path(command[command.index('--report') + 1]).write_text(
            HEADER.rstrip('\n') + '\twarmup_elapsed_us\n' +
            '\t'.join(fields) + f'\t{warmup_us + 5}\n')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(ops.subprocess, 'run', run)
    report = tmp_path / 'run.tsv'
    command = ['runner', '--report', str(report), '--warmup', '2', '--repeat', '1',
               '--warmup-us', '260', '--inner', '1']
    repeats = {'first':1, 'second':3}
    durations = {'first':260, 'second':300}
    if misidentify:
        with pytest.raises(ops.MockFailure, match='fixture identifier'):
            ops.run_fixture_replicas(command, fixtures, report, {}, repeats, durations)
    else:
        result = ops.run_fixture_replicas(command, fixtures, report, {}, repeats, durations)
        assert calls == [('second',3,300), ('first',1,260)] * 3
        assert result['first']['sample_count'] == 3 and result['second']['sample_count'] == 9
        assert all(r['collection_complete'] for r in result.values())


def test_device_preparation_runs_in_a_separate_unprofiled_process(tmp_path, monkeypatch):
    calls=[]
    def run(command, **kwargs):
        calls.append((command,kwargs['env']))
        if '--prepare-only' not in command:
            from pathlib import Path
            Path(command[command.index('--report')+1]).write_text(HEADER+row('first'))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(ops.subprocess,'run',run)
    report=tmp_path/'run.tsv'
    ops.run_fixture_batch(['runner','--report',str(report),'--mode','mspti','--state-library','state.so'],
                          ['first'],report,{'LD_PRELOAD':'/toolkit/libmspti.so:/existing/monitor.so'})
    assert len(calls)==2
    assert '--prepare-only' in calls[0][0] and '--prepare-only' not in calls[1][0]
    assert calls[0][0][calls[0][0].index('--mode')+1]=='sync'
    assert calls[0][1]['LD_PRELOAD']=='/existing/monitor.so'
    assert calls[1][1]['LD_PRELOAD']=='/toolkit/libmspti.so:/existing/monitor.so'
