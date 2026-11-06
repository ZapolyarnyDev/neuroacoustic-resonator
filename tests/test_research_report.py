from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neuroacoustic_resonator.analysis import research_report
from neuroacoustic_resonator.analysis.research_report import (
    format_p,
    render_report,
    require_current_commit,
    write_research_report,
)

COMMIT = "38ca1980123456789abcdef0123456789abcdef0"


def summary_fixture(*, commit: str = COMMIT, dirty: bool = False) -> dict[str, Any]:
    return {
        "causal_difference": {"stimulus_fraction": 0.045, "field_seed_fraction": 0.911},
        "cross_seed_classification": {
            "balanced_accuracy": 0.5416666,
            "chance_level": 0.25,
            "by_seed_root": [
                {"balanced_accuracy": 0.375},
                {"balanced_accuracy": 0.5},
                {"balanced_accuracy": 0.75},
            ],
            "checkpoint_clustered_bootstrap": {"balanced_accuracy_ci": [0.274, 0.601]},
            "checkpoint_label_permutation": {"p_value": 0.00019998},
        },
        "regional_representations": {
            "input": {"balanced_accuracy": 0.4166, "permutation_p_value": 0.0039},
            "output": {"balanced_accuracy": 0.75, "permutation_p_value": 0.0001},
            "combined": {"balanced_accuracy": 0.2916, "permutation_p_value": 0.37},
        },
        "provenance": {
            "command": "run",
            "arguments": {"permutation_samples": 10000, "bootstrap_samples": 5000},
            "created_at": "2026-11-06T15:04:00+00:00",
            "git": {"commit": commit, "dirty": dirty, "branch": "chore/stage-0"},
            "configs": [
                {"path": "configs/controlled_equilibration.yaml", "sha256": "a1b2c3d4"},
                {"path": "configs/field_only.yaml", "sha256": "e5f6a7b8"},
            ],
            "environment": {"python": "3.13.1", "packages": {"numpy": "2.5.2"}},
        },
    }


def test_render_report_lists_headline_and_regions() -> None:
    report = render_report(summary_fixture())

    assert report.startswith("# Controlled equilibration — 2026-11-06 · 38ca198\n")
    assert "controlled_equilibration.yaml `a1b2c3d`" in report
    assert (
        "Команда: `run` (permutation_samples 10000, bootstrap_samples 5000)" in report
    )
    assert "| Balanced accuracy (output pattern) | 0.542 (chance 0.25) |" in report
    assert "| 95% CI | [0.274, 0.601] |" in report
    assert "| p (перестановки) | 0.0002 |" in report
    assert "| По seed roots | 0.375 / 0.500 / 0.750 |" in report
    assert "| Доля дисперсии: стимул / seed | 0.045 / 0.911 |" in report
    assert "| input | 0.417 | 0.004 |" in report
    assert "| combined | 0.292 | 0.37 |" in report
    assert "| assoc |" not in report


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.37, "0.37"), (0.018, "0.018"), (0.0039, "0.004"), (9.999e-05, "0.0001")],
)
def test_format_p(value, expected) -> None:
    assert format_p(value) == expected


@pytest.mark.parametrize(
    ("recorded", "current", "message"),
    [
        (
            {"commit": COMMIT, "dirty": True},
            {"commit": COMMIT, "dirty": False},
            "dirty",
        ),
        (
            {"commit": COMMIT, "dirty": False},
            {"commit": COMMIT, "dirty": True},
            "uncommitted",
        ),
        (
            {"commit": COMMIT, "dirty": False},
            {"commit": "f" * 40, "dirty": False},
            "HEAD",
        ),
        ({"commit": None, "dirty": None}, {"commit": COMMIT, "dirty": False}, "git"),
    ],
)
def test_require_current_commit_rejects_mismatches(recorded, current, message) -> None:
    with pytest.raises(ValueError, match=message):
        require_current_commit({"git": recorded}, current)


def test_write_research_report_names_file_by_date_and_commit(
    tmp_path, monkeypatch
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary_fixture()), encoding="utf-8")
    config_text = Path("configs/controlled_equilibration.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "controlled.yaml"
    config_path.write_text(
        config_text.replace(
            "output_summary: experiments/logs/controlled_equilibration_baseline.json",
            f"output_summary: {summary_path.as_posix()}",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        research_report,
        "git_state",
        lambda _: {"commit": COMMIT, "dirty": False, "branch": "chore/stage-0"},
    )

    output = write_research_report(config_path, tmp_path / "reports")

    assert output == tmp_path / "reports" / "2026-11-06-38ca198.md"
    assert output.read_text(encoding="utf-8") == render_report(summary_fixture())


def test_write_research_report_requires_provenance(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    config_text = Path("configs/controlled_equilibration.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "controlled.yaml"
    config_path.write_text(
        config_text.replace(
            "output_summary: experiments/logs/controlled_equilibration_baseline.json",
            f"output_summary: {summary_path.as_posix()}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance"):
        write_research_report(config_path, tmp_path / "reports")
