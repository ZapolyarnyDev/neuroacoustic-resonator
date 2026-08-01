from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from neuroacoustic_resonator.analysis.propagation_probe import (
    PropagationProbeConfig,
    first_threshold_crossing,
    lagged_correlation,
    minimum_mask_distance,
    run_propagation_probe,
)
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.topology import GridTopology


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
    with pytest.raises(ValueError, match="duplicates"):
        PropagationProbeConfig(seeds=(1, 1))


def test_region_distance_respects_open_and_periodic_x_boundaries() -> None:
    regions = RegionMasks.from_size(8)
    open_topology = GridTopology((8, 8), boundary_x="open")
    periodic_topology = GridTopology((8, 8), boundary_x="periodic")

    assert minimum_mask_distance(regions.input, regions.output, open_topology) == 5
    assert minimum_mask_distance(regions.input, regions.output, periodic_topology) == 1


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
    assert rows[0]["version"] == "0.1"
    assert int(rows[0]["sequence"]) >= 0
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
    assert summary["environment"]["boundary_x"] == "periodic"
    assert summary["environment"]["graph_distances"]["input_to_output"] == 1
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


def test_controlled_probe_compares_coupled_and_uncoupled_trials(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
  coupling_strength: 0.2
  boundary_x: open
  boundary_y: periodic
synthetic_input:
  enabled: false
steps: 8
""",
        encoding="utf-8",
    )
    csv_path = tmp_path / "controlled.csv"
    summary_path = tmp_path / "controlled.json"

    summary = run_propagation_probe(
        PropagationProbeConfig(
            config_path=config_path,
            output_csv=csv_path,
            output_summary=summary_path,
            output_plot=None,
            warmup_steps=2,
            horizon=4,
            impulse=0.45,
            response_threshold=0.0,
            seeds=(1, 2),
            uncoupled_control=True,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 32
    assert {row["condition"] for row in rows} == {
        "coupled",
        "no_impulse",
        "uncoupled",
        "uncoupled_no_impulse",
    }
    assert {int(row["seed"]) for row in rows} == {1, 2}
    assert summary["aggregate"]["trials"] == 2
    assert len(summary["trials"]) == 2
    assert summary["environment"]["boundary_x"] == "open"
    for trial in summary["trials"]:
        assert trial["conditions"]["coupled"]["field"]["coupling_strength"] == 0.2
        assert trial["conditions"]["uncoupled"]["field"]["coupling_strength"] == 0.0
        assert (
            trial["conditions"]["uncoupled"]["field"]["coupling_homeostasis_rate"]
            == 0.0
        )
        assert "causal_peak_output_activity_delta" in trial["comparison"]
        assert "output_latency_steps" in trial["comparison"]
        assert "passed" in trial["comparison"]
