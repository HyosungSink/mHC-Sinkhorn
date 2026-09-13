"""Isolated operator builds and ACLNN runner compilation for local device tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import math
import statistics
from pathlib import Path
from typing import Mapping

from tests.st.scripts.artifact_cache import (
    ImmutableArtifactCache,
    sha256_bytes,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SOURCE = REPO_ROOT / "tests/st/runner/mhc_sinkhorn_runner.cpp"
DEFAULT_WORK_ROOT = Path("/tmp/cannjudge/mhcsinkhorn/local-tests")


class MockFailure(RuntimeError):
    """A local Mock workload could not be executed or verified."""


class OperatorCompileFailure(MockFailure):
    """A compiler failure with a retained exact-source build log."""

    def __init__(self, log_path: Path, returncode: int, identity: dict):
        super().__init__(f"operator compilation failed ({returncode}); see {log_path}")
        self.log_path = log_path
        self.returncode = returncode
        self.identity = identity


def run_checked(
    command,
    *,
    cwd: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if completed.returncode != 0:
        detail = ""
        if completed.stdout:
            detail = "\n" + "\n".join(completed.stdout.splitlines()[-80:])
        raise MockFailure(f"command failed ({completed.returncode}): {' '.join(command)}{detail}")
    return completed


def output_of(command, *, cwd: Path = REPO_ROOT) -> str:
    return run_checked(command, cwd=cwd, capture=True).stdout.strip()


def toolkit_root() -> Path:
    candidates = [
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        os.environ.get("ASCEND_HOME_PATH"),
        "/home/developer/Ascend/cann-8.5.2",
        "/usr/local/Ascend/ascend-toolkit/latest",
    ]
    for value in candidates:
        if value:
            path = Path(value).resolve()
            if (path / "include/acl/acl.h").is_file() and (path / "lib64").is_dir():
                return path
    raise MockFailure("cannot locate the CANN toolkit")


def toolkit_identity(toolkit: Path) -> dict[str, str]:
    version_candidates = list(toolkit.glob("**/version.info"))[:8]
    version = "unknown"
    for path in version_candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            version = sha256_bytes(text.encode("utf-8"))
            break
    return {
        "root": str(toolkit),
        "version_digest": version,
        "cxx": output_of(("c++", "--version")).splitlines()[0],
        "cmake": output_of(("cmake", "--version")).splitlines()[0],
    }


def build_artifacts(entry: Path) -> list[Path]:
    required = [
        Path("libcust_opapi.so"),
        Path("autogen/aclnn_mhc_sinkhorn.h"),
        Path("autogen/aclnn_mhc_sinkhorn.cpp"),
        Path("op_host/libcustom_ascendc_cust_optiling.so"),
    ]
    for relative in required:
        if not (entry / relative).is_file():
            raise MockFailure(f"isolated build omitted required artifact: {relative}")
    paths = list(required)
    for root in (
        entry / "op_kernel/ascendc_kernels/binary",
        entry / "tmp/vendors/custom",
    ):
        if not root.is_dir():
            raise MockFailure(f"isolated build omitted artifact tree: {root}")
        paths.extend(path.relative_to(entry) for path in root.rglob("*") if path.is_file())
    return paths


def working_source_files(submitted_only: bool = False) -> list[Path]:
    """Select source files without requiring Git metadata."""
    if not submitted_only:
        return sorted(p for p in (REPO_ROOT / "code").rglob("*") if p.is_file())
    import json
    manifest = json.loads((REPO_ROOT / "template-manifest.json").read_text())
    files = []
    for item in manifest["files"]:
        path = (REPO_ROOT / "code" / item["path"]).resolve()
        if not path.is_relative_to((REPO_ROOT / "code").resolve()) or not path.is_file():
            raise MockFailure(f"invalid or missing submitted source: {item['path']}")
        files.append(path)
    if not files:
        raise MockFailure("submission manifest has no sources")
    return sorted(set(files))


def ensure_working_operator_build(source_sha256: str, work_root: Path, jobs: int = 16,
                                  cache_root: Path | None = None,
                                  submitted_only: bool = False) -> tuple[Path, dict[str, object], bool]:
    """Compile a frozen working-source snapshot; never reuse an unrelated incremental tree."""
    if not 1 <= jobs <= 64:
        raise MockFailure("build parallelism must be within 1..64")
    toolkit = toolkit_root()
    identity = {"schema_version": 1, "kind": "working_operator_build", "operator": "MhcSinkhorn",
                "source_sha256": source_sha256, "soc": "ascend910b", "toolkit": toolkit_identity(toolkit)}
    selected_sources = working_source_files(submitted_only)
    identity["submitted_only"] = submitted_only
    identity["source_files"] = [str(p.relative_to(REPO_ROOT)) for p in selected_sources]
    cache = ImmutableArtifactCache(cache_root or work_root / "working-build-cache")
    with cache.lock(identity):
        try:
            cached = cache.lookup(identity)
        except Exception:
            cache.quarantine(cache.entry_path(identity))
            cached = None
        if cached is not None:
            return cached[0], cached[1], True
        entry = cache.entry_path(identity)
        if entry.exists():
            cache.quarantine(entry)
        source = entry / "source" / "code"
        for original in selected_sources:
            target = entry / "source" / original.relative_to(REPO_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        import hashlib
        digest = hashlib.sha256()
        source_files = sorted(p for p in source.rglob("*") if p.is_file())
        for path in source_files:
            digest.update(("code/" + str(path.relative_to(source))).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        if digest.hexdigest() != source_sha256:
            raise MockFailure("source digest mismatch while creating the build snapshot")
        environment = dict(os.environ, ASCEND_TOOLKIT_HOME=str(toolkit), ASCEND_HOME_PATH=str(toolkit))
        cmake_dirs = sorted(str(p.parent) for p in toolkit.glob("**/ASCConfig.cmake"))
        environment["CMAKE_PREFIX_PATH"] = ":".join(cmake_dirs + [environment.get("CMAKE_PREFIX_PATH", "")])
        log_path = entry / "compile.log"
        with log_path.open("w") as log:
            for command in (("cmake", "-S", str(source), "-B", str(entry)),
                            ("cmake", "--build", str(entry), "-j", str(jobs))):
                log.write("COMMAND " + " ".join(command) + "\n")
                log.flush()
                completed = subprocess.run(command, env=environment, text=True, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise OperatorCompileFailure(log_path, completed.returncode, identity)
        manifest = cache.seal(entry, identity,
            build_artifacts(entry) + [p.relative_to(entry) for p in source_files],
            {"isolation": "frozen_working_source_snapshot", "source_sha256": source_sha256})
        return entry, manifest, False


def runner_identity(build_dir: Path, build_manifest: Mapping[str, object]) -> dict[str, object]:
    toolkit = toolkit_root()
    mspti_include = toolkit / "tools/mspti/include"
    return {
        "schema_version": 1,
        "kind": "aclnn_fixture_runner",
        "build_cache_key": build_manifest["cache_key"],
        "runner_source_sha256": sha256_file(RUNNER_SOURCE),
        "generated_header_sha256": sha256_file(build_dir / "autogen/aclnn_mhc_sinkhorn.h"),
        "opapi_sha256": sha256_file(build_dir / "libcust_opapi.so"),
        "toolkit": toolkit_identity(toolkit),
        "compile_options": [
            "-std=c++17",
            "-O2",
            "-I",
            str(mspti_include),
            "-lcust_opapi",
            "-lascendcl",
            "-lnnopbase",
            "-lmspti",
            "-lmsprofiler",
            "-Wl,--allow-shlib-undefined",
        ],
    }


def ensure_runner(
    build_dir: Path, build_manifest: Mapping[str, object], work_root: Path
) -> tuple[Path, dict[str, object], bool]:
    toolkit = toolkit_root()
    identity = runner_identity(build_dir, build_manifest)
    cache = ImmutableArtifactCache(work_root / "runner-cache")
    with cache.lock(identity):
        try:
            cached = cache.lookup(identity)
        except Exception:
            cache.quarantine(cache.entry_path(identity))
            cached = None
        if cached is not None:
            return cached[0] / "bin/mhc_sinkhorn_runner", cached[1], True
        entry = cache.entry_path(identity)
        if entry.exists():
            shutil.rmtree(entry)
        binary = entry / "bin/mhc_sinkhorn_runner"
        binary.parent.mkdir(parents=True, exist_ok=True)
        mspti_include = toolkit / "tools/mspti/include"
        mspti_lib_dir = toolkit / "tools/mspti/lib64"
        rpath = [
            str(build_dir),
            str(build_dir / "op_host"),
            str(toolkit / "lib64"),
        ]
        if mspti_lib_dir.is_dir():
            rpath.append(str(mspti_lib_dir))
        run_checked(
            (
                "c++",
                "-std=c++17",
                "-O2",
                str(RUNNER_SOURCE),
                "-I",
                str(build_dir / "autogen"),
                "-I",
                str(toolkit / "include"),
                "-I",
                str(mspti_include),
                "-L",
                str(build_dir),
                "-L",
                str(toolkit / "lib64"),
                *(f"-Wl,-rpath,{path}" for path in rpath),
                "-lcust_opapi",
                "-lascendcl",
                "-lnnopbase",
                "-lmspti",
                "-lmsprofiler",
                # The CANN toolkit libraries reference driver-only symbols that
                # are resolved at run time through LD_LIBRARY_PATH.
                "-Wl,--allow-shlib-undefined",
                "-o",
                str(binary),
            )
        )
        manifest = cache.seal(
            entry,
            identity,
            [Path("bin/mhc_sinkhorn_runner")],
            {"source": str(RUNNER_SOURCE)},
        )
        return binary, manifest, False


def runner_environment(build_dir: Path, mspti: bool = False) -> dict[str, str]:
    toolkit = toolkit_root()
    environment = dict(os.environ)
    if mspti:
        # MSPTI only delivers activity records when the library is preloaded;
        # linking it is not enough on this platform.
        preload = str(toolkit / "lib64/libmspti.so")
        if environment.get("LD_PRELOAD"):
            preload = f"{preload}:{environment['LD_PRELOAD']}"
        environment["LD_PRELOAD"] = preload
    library_paths = [
        str(build_dir),
        str(build_dir / "op_host"),
        str(toolkit / "lib64"),
        "/usr/local/Ascend/driver/lib64",
        "/usr/local/Ascend/driver/lib64/common",
        "/usr/local/Ascend/driver/lib64/driver",
        "/usr/local/Ascend/driver/lib64/inner",
    ]
    if environment.get("LD_LIBRARY_PATH"):
        library_paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
    built_opp = build_dir / "tmp/vendors/custom"
    if not built_opp.is_dir():
        raise MockFailure("operator package is missing from the selected build")
    environment["ASCEND_CUSTOM_OPP_PATH"] = str(built_opp)
    environment["ASCEND_OPP_PATH"] = str(toolkit / "opp")
    environment.setdefault("ASCEND_AICPU_PATH", str(toolkit))
    environment.setdefault("ASCEND_TOOLKIT_HOME", str(toolkit))
    environment.setdefault("ASCEND_HOME_PATH", str(toolkit))
    return environment


def verify_execution_binding(build_dir: Path, build_manifest: Mapping[str, object], runner: Path,
                             runner_manifest: Mapping[str, object], output_dir: Path) -> dict[str, object]:
    loaded = {}
    expected_hashes = {str(build_dir / a["path"]): a["sha256"] for a in build_manifest["artifacts"]}
    for mapping in output_dir.rglob("*.libraries.tsv"):
        paths = {line[line.index("/"):] for line in mapping.read_text().splitlines() if "/" in line}
        if not paths or any(path.endswith(" (deleted)") or path not in expected_hashes for path in paths):
            raise MockFailure("runner loaded an unexpected operator library")
        hashes = {path: sha256_file(Path(path)) for path in sorted(paths)}
        if any(value != expected_hashes[path] for path, value in hashes.items()):
            raise MockFailure("loaded operator library differs from the sealed build")
        loaded[str(mapping)] = hashes
    if not loaded:
        raise MockFailure("runner did not record its loaded operator libraries")
    for artifact in build_manifest["artifacts"]:
        if sha256_file(build_dir / artifact["path"]) != artifact["sha256"]:
            raise MockFailure("operator artifact digest mismatch during measurement")
    runner_artifact = next(a for a in runner_manifest["artifacts"] if a["path"].endswith("mhc_sinkhorn_runner"))
    if sha256_file(runner) != runner_artifact["sha256"]:
        raise MockFailure("runner digest mismatch during measurement")
    return loaded


def ensure_device_state_library(work_root: Path = DEFAULT_WORK_ROOT, jobs: int = 16) -> Path:
    """Content-addressed auxiliary kernel, independent of the tested operator."""
    toolkit = toolkit_root()
    sources = sorted((RUNNER_SOURCE.parent/'device_state').iterdir())
    identity = {'schema_version': 1, 'kind': 'mock_device_state',
                'sources': {p.name: sha256_file(p) for p in sources},
                'toolkit': toolkit_identity(toolkit), 'soc': 'Ascend910B3', 'build_type': 'Release'}
    cache = ImmutableArtifactCache(work_root/'device-state-cache')
    with cache.lock(identity):
        try:
            cached = cache.lookup(identity)
        except Exception:
            cache.quarantine(cache.entry_path(identity)); cached = None
        if cached is not None:
            return cached[0]/'build/lib/libmock_ub_conditioner.so'
        entry = cache.entry_path(identity)
        if entry.exists():
            cache.quarantine(entry)
        source = entry/'source'; source.mkdir(parents=True)
        for p in sources:
            shutil.copyfile(p, source/p.name)
        environment = dict(os.environ, ASCEND_TOOLKIT_HOME=str(toolkit), ASCEND_HOME_PATH=str(toolkit))
        with (entry/'compile.log').open('w') as log:
            for command in (["cmake", "-S", str(source), "-B", str(entry/'build'),
                             "-DCMAKE_BUILD_TYPE=Release", f"-DASCEND_CANN_PACKAGE_PATH={toolkit}"],
                            ["cmake", "--build", str(entry/'build'), "-j", str(jobs)]):
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise MockFailure(f'device-state infrastructure build failed: {entry / "compile.log"}')
        library = Path('build/lib/libmock_ub_conditioner.so')
        cache.seal(entry, identity, [library]+[Path('source')/p.name for p in sources], {})
        return entry/library


def parse_runner_report(path: Path) -> dict[str, dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t")[:8] != [
        "case_id",
        "status",
        "stage",
        "acl_status",
        "device_us_p50",
        "warmup",
        "repeat",
        "workspace",
    ]:
        raise MockFailure("runner report header is missing or incompatible")
    results: dict[str, dict[str, object]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 8:
            raise MockFailure("runner emitted an invalid report row")
        case_id, status, stage, acl_status, timing, warmup, repeat, workspace = fields[:8]
        if case_id in results:
            raise MockFailure(f"runner emitted duplicate case id: {case_id}")
        extras = fields[8:]
        result: dict[str, object] = {
            "local_status": status,
            "stage": stage,
            "acl_status": int(acl_status),
            "local_elapsed_us": None if timing == "-" else float(timing),
            "warmup": int(warmup),
            "repeat": int(repeat),
            "workspace_bytes": int(workspace),
        }
        if len(extras) >= 3 and timing != "-":
            result["us_min"] = float(extras[0])
            result["us_max"] = float(extras[1])
            result["us_mean"] = float(extras[2])
        if len(fields) >= 16:
            result.update(timing_mode=fields[11], inner=int(fields[12]),
                          sample_count=int(fields[13]), mismatch_count=int(fields[14]),
                          element_count=int(fields[15]))
        if len(fields) >= 17: result["warmup_elapsed_us"] = float(fields[16])
        results[case_id] = result
    return results


def run_fixture_batch(command: list[str], fixtures: list[Path], report: Path,
                      environment: Mapping[str, str],
                      repeats_by_case: Mapping[str, int] | None = None,
                      warmup_us_by_case: Mapping[str, int] | None = None) -> dict[str, dict[str, object]]:
    """Keep MSPTI/device lifetime within one case; retain each raw report."""
    rows: dict[str, dict[str, object]] = {}
    report.parent.mkdir(parents=True, exist_ok=True)
    combined: list[str] = []
    report_index = command.index("--report") + 1
    for index, fixture in enumerate(fixtures):
        case_id = None
        if repeats_by_case is not None or warmup_us_by_case is not None:
            descriptor = (Path(fixture) / "case.conf").read_text().splitlines()
            case_id = next((line.split("=", 1)[1] for line in descriptor if line.startswith("case_id=")), None)
            if repeats_by_case is not None and (case_id not in repeats_by_case or type(repeats_by_case[case_id]) is not int or repeats_by_case[case_id] <= 0):
                raise MockFailure("missing or invalid per-case repeat count")
            if warmup_us_by_case is not None and (case_id not in warmup_us_by_case or type(warmup_us_by_case[case_id]) is not int or warmup_us_by_case[case_id] < 0):
                raise MockFailure("missing or invalid per-case warmup duration")
        attempts = []
        for attempt in range(1, 4):
            case_report = report.with_name(f"{report.stem}.case{index + 1}.attempt{attempt}.tsv")
            case_command = list(command)
            case_command[report_index] = str(case_report)
            if repeats_by_case is not None:
                case_command[case_command.index("--repeat") + 1] = str(repeats_by_case[case_id])
            if warmup_us_by_case is not None:
                case_command[case_command.index("--warmup-us") + 1] = str(warmup_us_by_case[case_id])
            if '--state-library' in case_command:
                prepare_command = list(case_command)
                prepare_command[report_index] = str(case_report) + '.prepare.tsv'
                prepare_command[prepare_command.index('--mode') + 1] = 'sync'
                prepare_command.append('--prepare-only')
                prepare_env = dict(environment)
                if prepare_env.get('LD_PRELOAD'):
                    prepare_env['LD_PRELOAD'] = ':'.join(p for p in prepare_env['LD_PRELOAD'].split(':')
                                                        if Path(p).name != 'libmspti.so')
                prepared = subprocess.run([*prepare_command, str(fixture)], env=prepare_env,
                                          cwd=REPO_ROOT, timeout=120, check=False)
                if prepared.returncode:
                    raise MockFailure('device preparation process failed; result is inconclusive')
            # A watchdog timeout is an infrastructure error.
            completed = subprocess.run([*case_command, str(fixture)], env=dict(environment),
                                       cwd=REPO_ROOT, timeout=120, check=False)
            if completed.returncode != 0:
                raise MockFailure(f"runner infrastructure failed with exit {completed.returncode}")
            case_rows = parse_runner_report(case_report)
            if case_id is not None and set(case_rows) != {case_id}:
                raise MockFailure("runner report does not match its fixture identifier")
            attempts.append({"report": str(case_report), "command": [*case_command, str(fixture)], "results": case_rows})
            incomplete = any(row["local_status"] == "Runner Error" and
                             row["stage"] == "mspti_incomplete_capture"
                             for row in case_rows.values())
            # Retry collection failure only. Duration and output mismatches are
            # definitive results and do not trigger another collection attempt.
            if not incomplete:
                break
        attempt_report = report.with_name(f"{report.stem}.case{index + 1}.attempts.json")
        import json
        attempt_report.write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
        for row in case_rows.values():
            row["capture_attempts"] = len(attempts)
        if set(rows) & set(case_rows):
            raise MockFailure("duplicate fixture identifier")
        rows.update(case_rows)
        lines = case_report.read_text(encoding="utf-8").splitlines()
        combined.extend(lines if index == 0 else lines[1:])
    report.write_text("\n".join(combined) + "\n", encoding="utf-8")
    return rows


def summarize_replica_rows(replicas: list[dict[str, object]], warmup: int = 1,
                           repeat: int = 10, warmup_us: int = 0) -> dict[str, object]:
    """Summarize three complete short captures without selecting durations."""
    if len(replicas) != 3:
        raise MockFailure("short-window measurement requires three replicas")
    def warmup_complete(row):
        if not warmup_us:
            return row.get("warmup") == warmup
        elapsed = row.get("warmup_elapsed_us")
        return (row.get("warmup", 0) >= max(1, warmup) and
                isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and
                elapsed >= warmup_us)

    summary = dict(replicas[0])
    complete = all(
        row.get("timing_mode") == "mspti" and warmup_complete(row) and
        row.get("repeat") == repeat and row.get("inner") == 1 and row.get("sample_count") == repeat and
        isinstance(row.get("us_mean"), (int, float)) and
        math.isfinite(row["us_mean"]) and row["us_mean"] > 0 for row in replicas)
    summary.update(
        replicas=replicas, replica_count=3, estimator="median_of_means",
        collection_complete=complete,
        sample_count=sum(int(row.get("sample_count", 0)) for row in replicas),
        local_elapsed_us=statistics.median([row["us_mean"] for row in replicas]) if complete else None,
    )
    if complete:
        summary.update(us_mean=statistics.mean([row["us_mean"] for row in replicas]),
                       us_min=min(row["us_min"] for row in replicas),
                       us_max=max(row["us_max"] for row in replicas))
    return summary


def run_fixture_replicas(command: list[str], fixtures: list[Path], report: Path,
                         environment: Mapping[str, str],
                         repeats_by_case: Mapping[str, int] | None = None,
                         warmup_us_by_case: Mapping[str, int] | None = None) -> dict[str, dict[str, object]]:
    """Use independent device lifetimes; retain all three means and raw traces."""
    import json
    groups = []
    reports = []
    report_index = command.index("--report") + 1
    warmup = int(command[command.index("--warmup") + 1])
    warmup_us = int(command[command.index("--warmup-us") + 1]) if "--warmup-us" in command else 0
    repeat = int(command[command.index("--repeat") + 1])
    if warmup < 0:
        raise MockFailure("warmup must be nonnegative")
    for replica in range(1, 4):
        replica_report = report.with_name(f"{report.stem}.replica{replica}.tsv")
        replica_command = list(command)
        replica_command[report_index] = str(replica_report)
        if "--output-dir" in replica_command:
            output_index = replica_command.index("--output-dir") + 1
            replica_command[output_index] = str(Path(command[output_index]) / f"replica{replica}")
        groups.append(run_fixture_batch(replica_command, fixtures, replica_report, environment,
                                        repeats_by_case, warmup_us_by_case))
        reports.append(str(replica_report))
    if any(group.keys() != groups[0].keys() for group in groups[1:]):
        raise MockFailure("replica reports contain different case identifiers")
    rows = {name: summarize_replica_rows([group[name] for group in groups], warmup,
                                       repeats_by_case[name] if repeats_by_case is not None else repeat,
                                       warmup_us_by_case[name] if warmup_us_by_case is not None else warmup_us)
            for name in groups[0]}
    report.with_suffix(".replicas.json").write_text(json.dumps({
        "benchmark": {"warmup": warmup, "warmup_us": warmup_us, "repeat": repeat,
                      "case_repeats": dict(repeats_by_case or {}),
                      "case_warmup_us": dict(warmup_us_by_case or {}), "inner": 1, "mode": "mspti"},
        "raw_reports": reports, "cases": rows,
    }, ensure_ascii=False, indent=2) + "\n")
    return rows
