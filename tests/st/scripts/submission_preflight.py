"""Validate the official submission constraints locally.

Two submission-contract checks run here:

1. Forbidden identifiers.  The official runner scans every submitted source
   (comments included) and aborts with ``submission preflight failed:
   <file>:<line>: forbidden identifier <token>``.
2. Submitted file set.  The runner uploads exactly the files listed in
   ``template-manifest.json``. Files under ``code/`` that are absent from the
   manifest are not transferred, so includes must resolve within that set.

``--build`` additionally compiles a tree that contains only the submitted files,
which is the closest local equivalent of the judge build.

Usage:
    python3 -m tests.st.scripts.submission_preflight
    python3 -m tests.st.scripts.submission_preflight --build
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "template-manifest.json"
FORBIDDEN = ["printf", "cout", "cerr", "puts", "write", "syscall", "dlopen", "dlsym", "asm", "freopen"]
SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".cc"}
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def scan_forbidden(root: Path) -> list[str]:
    problems = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for token in FORBIDDEN:
                if re.search(rf"\b{re.escape(token)}\b", line):
                    problems.append(f"{path.relative_to(root.parent)}:{line_no}: "
                                    f"forbidden identifier {token!r}: {line.strip()}")
    return problems


def manifest_files(manifest: Path) -> set[Path]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {Path("code") / entry["path"] for entry in data["files"]}


def scan_includes(repo_root: Path, submitted: set[Path]) -> list[str]:
    """Flag local includes that resolve to a repository file outside the upload."""
    problems = []
    for relative in sorted(submitted):
        if relative.suffix not in SOURCE_SUFFIXES:
            continue
        absolute = repo_root / relative
        if not absolute.is_file():
            problems.append(f"{relative}: listed in the manifest but missing on disk")
            continue
        for line_no, line in enumerate(absolute.read_text(errors="replace").splitlines(), 1):
            match = INCLUDE_PATTERN.match(line)
            if match is None:
                continue
            resolved = (absolute.parent / match.group(1)).resolve()
            try:
                resolved_relative = resolved.relative_to(repo_root.resolve())
            except ValueError:
                continue  # toolkit / system header from the include path
            if not resolved.is_file():
                continue  # resolved by the compiler's include path (e.g. AscendC)
            if resolved_relative not in submitted:
                problems.append(f"{relative}:{line_no}: include is not in the submitted file set: "
                                f"{resolved_relative} -> {match.group(1)}")
    return problems


def simulate_build(repo_root: Path, submitted: set[Path], work_dir: Path, jobs: int) -> list[str]:
    """Compile a tree that contains only the submitted files."""
    sys.path.insert(0, str(repo_root))
    from tests.st.scripts.build_ops import toolkit_root  # noqa: E402

    toolkit = toolkit_root()
    shutil.rmtree(work_dir, ignore_errors=True)
    tree = work_dir / "code"
    for relative in sorted(submitted):
        target = tree / relative.relative_to("code")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative, target)

    environment = dict(os.environ, ASCEND_TOOLKIT_HOME=str(toolkit), ASCEND_HOME_PATH=str(toolkit))
    cmake_dirs = sorted(str(path.parent) for path in toolkit.glob("**/ASCConfig.cmake"))
    if cmake_dirs:
        existing = environment.get("CMAKE_PREFIX_PATH")
        environment["CMAKE_PREFIX_PATH"] = ":".join(cmake_dirs + ([existing] if existing else []))

    problems = []
    for command in (("cmake", "-S", str(tree), "-B", str(work_dir / "build")),
                    ("cmake", "--build", str(work_dir / "build"), "-j", str(jobs))):
        completed = subprocess.run(command, env=environment, text=True, capture_output=True)
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-40:])
            problems.append(f"simulated judge build failed: {' '.join(command)}\n{tail}")
            break
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "code")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build", action="store_true", help="also compile the submitted file set")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/cannjudge/mhcsinkhorn/judge-sim"))
    parser.add_argument("--jobs", type=int, choices=range(1, 65), default=16)
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"source root is missing: {args.root}")
        return 2

    problems = scan_forbidden(args.root)
    submitted = manifest_files(args.manifest)
    problems += scan_includes(REPO_ROOT, submitted)
    if args.build and not problems:
        problems += simulate_build(REPO_ROOT, submitted, args.work_dir, args.jobs)

    for problem in problems:
        print(problem)
    print("preflight clean" if not problems else f"preflight found {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
