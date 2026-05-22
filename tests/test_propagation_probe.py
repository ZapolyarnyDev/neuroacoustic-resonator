from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from neuroacoustic_resonator.analysis.propagation_probe import (
    PropagationProbeConfig,
    first_threshold_crossing,
    lagged_correlation,
    run_propagation_probe,
)


def test_first_threshold_crossing_reports_one_based_step() -> None:
    values = np.asarray([0.0, 0.1, 0.3])

    assert first_threshold_crossing(values, 0.2) == 3
    assert first_threshold_crossing(values, 0.5) is None


def test_lagged_correlation_reports_delayed_relationship() -> None:
    source = np.asarray([0.0, 1.0, 0.5, 0.0, 0.0])
    target = np.asarray([0.0, 0.0, 1.0, 0.5, 0.0])

    result = lagged_correlation(source, target, max_lag=3)

    assert result["lag_steps"] == 1
    assert result["correlation"] == pytest.approx(1.0)


def test_propagation_probe_config_validates_values() -> None:
    with pytest.raises(ValueError, match="horizon"):
        PropagationProbeConfig(horizon=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        PropagationProbeConfig(warmup_steps=-1)
    with pytest.raises(ValueError, match="response_threshold"):
        PropagationProbeConfig(response_threshold=-0.1)


def test_run_propagation_probe_writes_csv_and_summary(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
  coupling_strength: 0.2
synthetic_input:
  enabled: false
steps: 8
""",
        encoding="utf-8",
    )
    csv_path = tmp_path / "probe.csv"
    summary_path = tmp_path / "probe.json"
    plot_path = tmp_path / "probe.png"

    summary = run_propagation_probe(
        PropagationProbeConfig(
            config_path=config_path,
            output_csv=csv_path,
            output_summary=summary_path,
            output_plot=plot_path,
            warmup_steps=2,
            horizon=8,
            impulse=0.45,
            response_threshold=0.0,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 8
    assert rows[0]["offset"] == "1"
    assert "output_activity_delta" in rows[0]
    assert "output_activity_delta_abs" in rows[0]
    assert "output_fast_response_score" in rows[0]
    assert "output_slow_drift_score" in rows[0]
    assert plot_path.exists()
    assert summary["response_reached_output"] is True
    assert "input_to_assoc" in summary
    assert "assoc_to_output" in summary
    assert "peak_delta_output_assoc_ratio" in summary
    assert isinstance(summary["output_peak_at_horizon_end"], bool)
    assert "slow_fast_peak_ratio" in summary
    assert loaded_summary["peak_output_activity_step"] >= 1


def test_run_propagation_probe_supports_multiple_horizons(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
  coupling_strength: 0.2
synthetic_input:
  enabled: false
steps: 8
""",
        encoding="utf-8",
    )
    csv_path = tmp_path / "probe.csv"
    summary_path = tmp_path / "probe.json"

    summary = run_propagation_probe(
        PropagationProbeConfig(
            config_path=config_path,
            output_csv=csv_path,
            output_summary=summary_path,
            output_plot=None,
            warmup_steps=2,
            horizon=8,
            horizons=(4, 8),
            impulse=0.45,
            response_threshold=0.0,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 12
    assert {row["horizon"] for row in rows} == {"4", "8"}
    assert set(summary["horizons"]) == {"4", "8"}
