from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("numpy", "scipy", "pydantic", "pyyaml")


def collect_provenance(
    *,
    command: str,
    arguments: Mapping[str, Any],
    configs: Sequence[str | Path],
    repository: str | Path | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if not command:
        msg = "command must not be empty"
        raise ValueError(msg)
    root = Path.cwd() if repository is None else Path(repository)
    timestamp = created_at or datetime.now(UTC)
    return {
        "command": command,
        "arguments": dict(arguments),
        "created_at": timestamp.astimezone(UTC).isoformat(timespec="seconds"),
        "git": git_state(root),
        "configs": [
            {"path": Path(path).as_posix(), "sha256": file_sha256(path)}
            for path in configs
        ],
        "environment": environment_versions(),
    }


def git_state(repository: str | Path) -> dict[str, Any]:
    commit = _git(repository, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _git(repository, "status", "--porcelain", "--untracked-files=no")
    branch = _git(repository, "rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status),
        "branch": branch,
    }


def environment_versions() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "packages": packages,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: str | Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
