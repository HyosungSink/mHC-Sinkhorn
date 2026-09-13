"""Build the local operator and run the selected device test suites."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

from tests.st.scripts.build_ops import (
    DEFAULT_WORK_ROOT, REPO_ROOT, MockFailure, OperatorCompileFailure, working_source_files, ensure_runner, ensure_working_operator_build, run_checked,
    run_fixture_batch, run_fixture_replicas, runner_environment, parse_runner_report, verify_execution_binding, ensure_device_state_library,
)
from tests.common.fixtures import materialize_fixture
from tests.common.cases import load_cases


def source_digest(submitted_only=False) -> str:
    digest = hashlib.sha256()
    for path in working_source_files(submitted_only):
        if path.is_file():
            digest.update(str(path.relative_to(REPO_ROOT)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def partition_specs(specs: list[dict]) -> tuple[list[dict], list[dict]]:
    timed, correctness = [], []
    for spec in specs:
        measurement = spec.get("measurement", "kernel")
        if measurement not in {"kernel", "correctness"}:
            raise MockFailure("unknown preflight measurement mode")
        (timed if measurement == "kernel" else correctness).append(spec)
    return timed, correctness


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", choices=("coverage", "correctness", "full", "mock", "custom"), default="full")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--tag", action="append", default=[], help="require all selected tags")
    parser.add_argument("--build-dir", type=Path,
                        help="optional root for immutable source-specific build entries")
    parser.add_argument("--report", type=Path, default=DEFAULT_WORK_ROOT / "preflight.tsv")
    parser.add_argument("--jobs", type=int, choices=range(1, 65), default=16)
    parser.add_argument("--warmup", type=int, default=5, help="minimum Mock untimed launches per process")
    parser.add_argument("--repeat", type=int, help="override every selected Mock's measured launches per process")
    parser.add_argument("--interval-us", type=int, default=0, help="idle interval before each Mock launch, outside kernel timing")
    parser.add_argument('--device-state', choices=('unprepared','nonzero-ub'), default='nonzero-ub')
    parser.add_argument("--warmup-us", type=int,
                        help="override per-case warmup time after first sync; 0 uses counts only")
    args = parser.parse_args(argv)
    if args.warmup < 0 or (args.repeat is not None and args.repeat <= 0) or args.interval_us < 0 or (args.warmup_us is not None and args.warmup_us < 0):
        raise ValueError("invalid Mock measurement protocol")
    specs = load_cases(args.suite, args.case, args.tag)
    if args.list:
        for spec in specs:
            print(spec["name"])
        return 0
    submitted_only = any("sensitivity" in spec.get("tags", []) for spec in specs)
    before = source_digest(submitted_only)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        build_dir, build_manifest, _ = ensure_working_operator_build(
            before, DEFAULT_WORK_ROOT, args.jobs, args.build_dir, submitted_only=submitted_only)
    except OperatorCompileFailure as error:
        args.report.with_suffix(".json").write_text(json.dumps({
            "passed": False, "status": "Compile Error", "source_sha256": before,
            "build_log": str(error.log_path), "compiler_exit_code": error.returncode,
            "submitted_only": submitted_only}, indent=2) + "\n")
        print(f"Compile Error: {error.log_path}")
        return 1
    if source_digest(submitted_only) != before:
        raise MockFailure("source digest mismatch after preflight build")
    runner, manifest, _ = ensure_runner(build_dir, build_manifest, DEFAULT_WORK_ROOT)
    if not specs or any(spec.get("verify", True) is not True for spec in specs):
        raise MockFailure("preflight requires verified cases")
    timed_specs, correctness_specs = partition_specs(specs)
    timed_fixtures = [materialize_fixture(spec, DEFAULT_WORK_ROOT)[0] for spec in timed_specs]
    correctness_fixtures = [materialize_fixture(spec, DEFAULT_WORK_ROOT)[0] for spec in correctness_specs]
    repeats_by_case = {spec["name"]: args.repeat if args.repeat is not None else spec.get("repeat", 1)
                       for spec in timed_specs}
    if any(type(value) is not int or value <= 0 for value in repeats_by_case.values()):
        raise ValueError("invalid Mock case repeat count")
    warmup_us_by_case = {spec["name"]: args.warmup_us if args.warmup_us is not None else spec.get("warmup_us", 260)
                         for spec in timed_specs}
    if any(type(value) is not int or value < 0 for value in warmup_us_by_case.values()):
        raise ValueError("invalid Mock case warmup duration")
    benchmark = {"warmup": args.warmup, "repeat": args.repeat or 1, "replicas": 3,
                 "case_repeats": repeats_by_case,
                 "case_warmup_us": warmup_us_by_case,
                 "interval_us": args.interval_us, "inner": 1, "mode": "mspti",
                 "estimator": "median_of_means", 'device_state': args.device_state}
    benchmark["warmup_us"] = args.warmup_us if args.warmup_us is not None else 260
    preparation_args = ["--warmup-us", str(benchmark["warmup_us"])]
    if timed_fixtures and args.device_state == 'nonzero-ub':
        library = ensure_device_state_library(DEFAULT_WORK_ROOT, args.jobs)
        preparation_args += ['--state-library', str(library)]
    with (DEFAULT_WORK_ROOT / "npu.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        measure = run_fixture_replicas if benchmark.get("replicas", 1) == 3 else run_fixture_batch
        rows = measure([
            str(runner), "--report", str(args.report), "--device", "0",
            "--mode", "mspti", "--warmup", str(benchmark["warmup"]),
            "--repeat", str(benchmark["repeat"]), "--inner", "1",
            "--interval-us", str(benchmark["interval_us"]),
            "--output-dir", str(args.report.with_suffix(".outputs")),
            *preparation_args,
        ], timed_fixtures, args.report, runner_environment(build_dir, mspti=True),
           repeats_by_case, warmup_us_by_case) if timed_fixtures else {}
        if correctness_fixtures:
            count = 3 if submitted_only else 1
            all_rows = []
            for replica in range(count):
                correctness_report = args.report.with_name(args.report.stem + f".correctness{replica}.tsv")
                outputs = args.report.with_suffix(".outputs") / f"correctness{replica}"
                environment = runner_environment(build_dir, mspti=False)
                environment.pop("MHCS_CASE_ORACLE_ROUND_OVERRIDE", None)
                try:
                    completed = subprocess.run([
                        str(runner), "--report", str(correctness_report), "--device", "0",
                        "--mode", "sync", "--warmup", "1", "--repeat", "3", "--inner", "1",
                        "--output-dir", str(outputs), *map(str, correctness_fixtures),
                    ], env=environment, cwd=REPO_ROOT, timeout=120, check=False)
                except subprocess.TimeoutExpired as error:
                    raise MockFailure("correctness runner watchdog expired; result is inconclusive") from error
                if completed.returncode != 0:
                    raise MockFailure("correctness runner infrastructure failed")
                replica_rows = parse_runner_report(correctness_report)
                if set(replica_rows) != {s["name"] for s in correctness_specs}:
                    raise MockFailure("missing or unexpected correctness results")
                for row in replica_rows.values():
                    if row["local_status"] not in {"Passed", "Wrong Answer"}:
                        raise MockFailure("abnormal device execution; result is inconclusive")
                    row["diagnostic_sync_us"] = row.pop("local_elapsed_us")
                    row["local_elapsed_us"] = None
                verify_execution_binding(build_dir, build_manifest, runner, manifest, outputs)
                all_rows.append(replica_rows)
            for spec in correctness_specs:
                name = spec["name"]
                replicas = [replica[name] for replica in all_rows]
                row = dict(next((r for r in replicas if r["local_status"] != "Passed"), replicas[0]))
                row["replicas"] = replicas
                rows[name] = row
    loaded = verify_execution_binding(build_dir, build_manifest, runner, manifest,
                                      args.report.with_suffix(".outputs"))
    passed = (source_digest(submitted_only) == before and len(rows) == len(specs) and
              all(row["local_status"] == "Passed" and
                  all(r["local_status"] == "Passed" for r in row.get("replicas", [])) and
                  row.get("collection_complete", True) for row in rows.values()))
    for name, row in rows.items():
        print(name, row["local_status"], row["local_elapsed_us"], row.get("mismatch_count"), flush=True)
    args.report.with_suffix(".json").write_text(json.dumps({
        "passed": passed, "status": "Pass" if passed else "Wrong Answer",
        "submitted_only": submitted_only, "source_sha256": before, "specs_sha256": hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(),
        "runner_cache_key": manifest["cache_key"], "runner_manifest": manifest,
        "operator_build_directory": str(build_dir), "operator_build_manifest": build_manifest,
        "loaded_libraries": loaded, "benchmark": benchmark, "cases": rows,
    }, ensure_ascii=False, indent=2) + "\n")
    print("PASS" if passed else "FAIL", len(rows), flush=True)
    return 0 if passed else 1


def main(argv=None) -> int:
    try:
        return _main(argv)
    except (MockFailure, OSError, ValueError, KeyError) as error:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--report", type=Path, default=DEFAULT_WORK_ROOT / "preflight.tsv")
        args, _ = parser.parse_known_args(argv)
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.with_suffix(".json").write_text(json.dumps({
                "passed": False, "status": "INCONCLUSIVE", "error": str(error)}, indent=2) + "\n")
        except OSError:
            pass
        print(f"INCONCLUSIVE: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
