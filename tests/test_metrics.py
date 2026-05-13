import csv
import json

import pytest

from neuroacoustic_resonator import (
    FieldConfig,
    MetricsHistory,
    RegionMasks,
    RegionalActivityTracker,
    Simulation,
    compute_regional_activity_metrics,
)


def test_metrics_history_records_metrics() -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))
    history = MetricsHistory()

    history.append(simulation.snapshot().metrics)
    history.append(simulation.step().metrics)

    assert len(history) == 2
    assert history.latest().step == 1
    assert [metrics.step for metrics in history] == [0, 1]


def test_metrics_history_returns_rows() -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))
    metrics = simulation.step().metrics
    history = MetricsHistory([metrics])

    rows = history.to_rows()

    assert rows == [
        {
            "step": metrics.step,
            "mean_metabolite": metrics.mean_metabolite,
            "min_metabolite": metrics.min_metabolite,
            "mean_trace": metrics.mean_trace,
            "max_trace": metrics.max_trace,
            "global_synchrony": metrics.global_synchrony,
            "mean_local_synchrony": metrics.mean_local_synchrony,
            "max_local_synchrony": metrics.max_local_synchrony,
        }
    ]


def test_metrics_history_writes_csv(tmp_path) -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))
    history = MetricsHistory(frame.metrics for frame in simulation.run(2))

    output_path = history.write_csv(tmp_path / "metrics.csv")

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert output_path.exists()
    assert [int(row["step"]) for row in rows] == [1, 2]
    assert "global_synchrony" in rows[0]


def test_metrics_history_writes_jsonl(tmp_path) -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))
    history = MetricsHistory(frame.metrics for frame in simulation.run(2))

    output_path = history.write_jsonl(tmp_path / "metrics.jsonl")

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2]
    assert "mean_metabolite" in rows[0]


def test_compute_regional_activity_metrics_reports_output_activity() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    frame = simulation.step()

    metrics = compute_regional_activity_metrics(
        frame,
        regions,
        input_value=simulation.last_input_value,
    )

    assert metrics.step == 1
    assert metrics.input_activity >= 0.0
    assert metrics.assoc_activity >= 0.0
    assert metrics.output_activity >= 0.0
    assert metrics.left_to_right_ratio >= 0.0
    assert metrics.output_event_score == 0.0


def test_regional_activity_tracker_reports_output_event_score() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    tracker = RegionalActivityTracker()

    tracker.update(simulation.snapshot(), regions)
    simulation.field.apply_phase_impulse(regions.output, 0.5)
    changed = tracker.update(simulation.snapshot(), regions)

    assert changed.output_synchrony_delta != 0.0
    assert changed.output_event_score > 0.0


def test_compute_regional_activity_metrics_rejects_shape_mismatch() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(5)

    with pytest.raises(ValueError, match="matching shapes"):
        compute_regional_activity_metrics(simulation.snapshot(), regions)
