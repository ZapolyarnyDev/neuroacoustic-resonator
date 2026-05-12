import json

from neuroacoustic_resonator.experiment_analysis import (
    ExperimentAnalysisConfig,
    export_experiment_plots,
    run_experiment_suite,
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
