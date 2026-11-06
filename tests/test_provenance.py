from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from neuroacoustic_resonator.analysis.provenance import (
    collect_provenance,
    file_sha256,
    git_state,
)


def init_repository(path: Path) -> None:
    for args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test"),
        ("config", "commit.gpgsign", "false"),
    ):
        subprocess.run(["git", *args], cwd=path, check=True)


def commit_all(path: Path) -> None:
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_provenance_records_commit_configs_and_environment(tmp_path) -> None:
    init_repository(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    commit_all(tmp_path)

    provenance = collect_provenance(
        command="run",
        arguments={"permutation_samples": 10},
        configs=(config,),
        repository=tmp_path,
        created_at=datetime(2026, 11, 6, 12, 30, tzinfo=UTC),
    )

    assert provenance["command"] == "run"
    assert provenance["arguments"] == {"permutation_samples": 10}
    assert provenance["created_at"] == "2026-11-06T12:30:00+00:00"
    assert len(provenance["git"]["commit"]) == 40
    assert provenance["git"]["dirty"] is False
    assert provenance["configs"] == [
        {"path": config.as_posix(), "sha256": file_sha256(config)}
    ]
    assert provenance["environment"]["python"]
    assert provenance["environment"]["packages"]["numpy"]


def test_git_state_flags_uncommitted_tracked_changes(tmp_path) -> None:
    init_repository(tmp_path)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("a", encoding="utf-8")
    commit_all(tmp_path)
    (tmp_path / "untracked.txt").write_text("ignored", encoding="utf-8")

    assert git_state(tmp_path)["dirty"] is False

    tracked.write_text("b", encoding="utf-8")

    assert git_state(tmp_path)["dirty"] is True


def test_git_state_outside_repository_is_empty(tmp_path) -> None:
    assert git_state(tmp_path) == {"commit": None, "dirty": None, "branch": None}


def test_config_hash_changes_with_content(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    before = file_sha256(config)
    config.write_text("seed: 2\n", encoding="utf-8")

    assert file_sha256(config) != before
