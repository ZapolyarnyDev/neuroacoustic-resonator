import csv

import pytest

from neuroacoustic_resonator.analysis.long_metrics import (
    collect_metrics,
    main,
    write_metrics_history,
)


def test_collect_metrics_samples_initial_interval_and_final(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 4
  seed: 1
steps: 5
""",
        encoding="utf-8",
    )

    history = collect_metrics(config_path, sample_interval=2)

    assert [metrics.step for metrics in history] == [0, 2, 4, 5]


def test_collect_metrics_accepts_step_override(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 4
  seed: 1
steps: 2
""",
        encoding="utf-8",
    )

    history = collect_metrics(config_path, steps=3)

    assert history.latest().step == 3


def test_write_history_rejects_unknown_suffix(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("field:\n  size: 4\nsteps: 1\n", encoding="utf-8")
    history = collect_metrics(config_path)

    with pytest.raises(ValueError, match="csv or .jsonl"):
        write_metrics_history(history, tmp_path / "metrics.txt")


def test_main_writes_metrics_csv(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "metrics.csv"
    config_path.write_text(
        """
field:
  size: 4
  seed: 1
steps: 2
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--steps",
            "3",
            "--sample-interval",
            "2",
            "--output",
            str(output_path),
        ]
    )

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert exit_code == 0
    assert [int(row["step"]) for row in rows] == [0, 2, 3]
