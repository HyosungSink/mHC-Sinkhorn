"""Content-addressed storage for operator builds and test fixtures."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Mapping


SCHEMA_VERSION = 1


class CacheValidationError(RuntimeError):
    """A supposedly reusable entry does not match its immutable manifest."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_records(root: Path, relative_paths: Iterable[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise CacheValidationError(f"artifact path escapes cache entry: {relative}")
        name = relative.as_posix()
        if name in seen:
            continue
        seen.add(name)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise CacheValidationError(f"artifact is not a regular file: {path}")
        records.append({"path": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    if not records:
        raise CacheValidationError("an immutable cache entry cannot have an empty artifact set")
    return records


class ImmutableArtifactCache:
    """Stores complete build trees and verifies every declared file before reuse."""

    manifest_name = ".artifact-cache.json"

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def key_for(identity: Mapping[str, object]) -> str:
        return sha256_bytes(canonical_json(identity).encode("utf-8"))

    def entry_path(self, identity: Mapping[str, object]) -> Path:
        key = self.key_for(identity)
        return self.root / "entries" / key[:2] / key

    @contextlib.contextmanager
    def lock(self, identity: Mapping[str, object]) -> Iterator[None]:
        key = self.key_for(identity)
        path = self.root / "locks" / f"{key}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def seal(
        self,
        entry: Path,
        identity: Mapping[str, object],
        relative_paths: Iterable[Path],
        provenance: Mapping[str, object],
    ) -> dict[str, object]:
        expected = self.entry_path(identity)
        if entry.resolve() != expected.resolve():
            raise CacheValidationError(f"entry path does not match cache identity: {entry}")
        records = artifact_records(entry, relative_paths)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "cache_key": self.key_for(identity),
            "identity": dict(identity),
            "artifacts": records,
            "artifacts_sha256": sha256_bytes(canonical_json(records).encode("utf-8")),
            "provenance": dict(provenance),
        }
        atomic_json(entry / self.manifest_name, payload)
        return payload

    def lookup(self, identity: Mapping[str, object]) -> tuple[Path, dict[str, object]] | None:
        entry = self.entry_path(identity)
        manifest_path = entry / self.manifest_name
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CacheValidationError(f"unreadable cache manifest: {manifest_path}") from error
        key = self.key_for(identity)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("cache_key") != key
            or manifest.get("identity") != dict(identity)
        ):
            raise CacheValidationError(f"cache identity mismatch: {entry}")
        records = manifest.get("artifacts")
        if not isinstance(records, list) or manifest.get("artifacts_sha256") != sha256_bytes(
            canonical_json(records).encode("utf-8")
        ):
            raise CacheValidationError(f"cache artifact manifest mismatch: {entry}")
        for record in records:
            if not isinstance(record, dict):
                raise CacheValidationError(f"invalid artifact record: {entry}")
            relative = Path(str(record.get("path", "")))
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise CacheValidationError(f"unsafe artifact record: {entry}")
            path = entry / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record.get("size")
                or sha256_file(path) != record.get("sha256")
            ):
                raise CacheValidationError(f"cached artifact digest mismatch: {path}")
        return entry, manifest

    def quarantine(self, entry: Path) -> Path:
        destination = entry.with_name(f"{entry.name}.invalid.{os.getpid()}")
        suffix = 0
        while destination.exists():
            suffix += 1
            destination = entry.with_name(f"{entry.name}.invalid.{os.getpid()}.{suffix}")
        entry.replace(destination)
        return destination
