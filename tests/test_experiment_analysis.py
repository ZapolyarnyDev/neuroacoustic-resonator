import json

import numpy as np

from neuroacoustic_resonator.analysis.experiments import (
    ExperimentAnalysisConfig,
    export_experiment_plots,
    propagation_metrics,
    run_experiment_suite,
    tail_metrics,
)


def test_run_experiment_suite_writes_csv_json_and_plots(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "logs"
    config_path.write_text(
        """
field:
  size: 4
  seed: 1
  coupling_strength: 0.1
synthetic_input:
  enabled: false
steps: 4
""",
        encoding="utf-8",
    )

    summary = run_experiment_suite(
        ExperimentAnalysisConfig(
            config_path=config_path,
            output_dir=output_dir,
            warmup_steps=2,
            response_trials=2,
            response_horizon=4,
            response_pause=2,
            propagation_horizon=8,
            memory_horizon=4,
            memory_pause=4,
        )
    )

    assert (output_dir / "experiment_1_response_stability.csv").exists()
    assert (output_dir / "experiment_2_propagation_distance.csv").exists()
    assert (output_dir / "experiment_3_memory.csv").exists()
    summary_path = output_dir / "experiment_summary.json"
    assert summary_path.exists()
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["experiment_2"]["right"][
            "peak_step"
        ]
        >= 1
    )
    assert summary["outputs"]["plots"]


def test_export_experiment_plots_returns_png_paths(tmp_path) -> None:
    output_dir = tmp_path / "logs"
    run_experiment_suite(
        ExperimentAnalysisConfig(
            output_dir=output_dir,
            warmup_steps=2,
            response_trials=2,
            response_horizon=4,
            response_pause=2,
            propagation_horizon=8,
            memory_horizon=4,
            memory_pause=4,
        )
    )

    plot_paths = export_experiment_plots(output_dir)

    assert len(plot_paths) == 3
    assert all(path.exists() for path in plot_paths)


def test_tail_metrics_detects_unsaturated_window() -> None:
    metrics = tail_metrics(np.array([0.1, 0.2, 0.3, 0.4]))

    assert metrics["peak_at_window_end"]
    assert metrics["tail_slope"] > 0.0


def test_propagation_metrics_recommends_longer_horizon_for_rising_tail() -> None:
    metrics = propagation_metrics(np.array([0.1, 0.2, 0.3, 0.4]))

    assert metrics["increase_horizon_recommended"]
