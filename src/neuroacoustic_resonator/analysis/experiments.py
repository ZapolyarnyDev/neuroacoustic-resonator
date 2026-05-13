from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation


ExperimentRows = list[dict[str, Any]]
ExperimentSummary = dict[str, Any]


@dataclass(frozen=True)
class ExperimentAnalysisConfig:
    config_path: Path = Path("configs") / "synthetic_input.yaml"
    output_dir: Path = Path("experiments") / "logs"
    warmup_steps: int = 200
    impulse: float = 0.45
    response_trials: int = 10
    response_horizon: int = 64
    response_pause: int = 96
    propagation_horizon: int = 1024
    memory_horizon: int = 64
    memory_pause: int = 256

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {"config_path", "output_dir", "impulse"}:
                continue
            if value < 1:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        if self.impulse < 0.0:
            msg = "impulse must be non-negative"
            raise ValueError(msg)


def run_experiment_suite(config: ExperimentAnalysisConfig) -> ExperimentSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sim_config = SimulationConfig.from_file(config.config_path)
    field_config = sim_config.to_field_config()
    regions = RegionMasks.from_size(field_config.size)

    response_rows, response_summary = run_response_stability(
        config,
        field_config=field_config,
        regions=regions,
    )
    propagation_rows, propagation_summary = run_propagation_distance(
        config,
        field_config=field_config,
        regions=regions,
    )
    memory_rows, memory_summary = run_memory_experiment(
        config,
        field_config=field_config,
        regions=regions,
    )

    paths = {
        "response_csv": config.output_dir / "experiment_1_response_stability.csv",
        "propagation_csv": config.output_dir / "experiment_2_propagation_distance.csv",
        "memory_csv": config.output_dir / "experiment_3_memory.csv",
        "summary_json": config.output_dir / "experiment_summary.json",
    }
    write_rows(paths["response_csv"], response_rows)
    write_rows(paths["propagation_csv"], propagation_rows)
    write_rows(paths["memory_csv"], memory_rows)

    summary: ExperimentSummary = {
        "config": str(config.config_path),
        "parameters": {
            "warmup_steps": config.warmup_steps,
            "impulse": config.impulse,
            "response_trials": config.response_trials,
            "response_horizon": config.response_horizon,
            "response_pause": config.response_pause,
            "propagation_horizon": config.propagation_horizon,
            "memory_horizon": config.memory_horizon,
            "memory_pause": config.memory_pause,
        },
        "experiment_1": response_summary,
        "experiment_2": propagation_summary,
        "experiment_3": memory_summary,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_paths = export_experiment_plots(config.output_dir)
    summary["outputs"]["plots"] = [str(path) for path in plot_paths]
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_response_stability(
    config: ExperimentAnalysisConfig,
    *,
    field_config: Any,
    regions: RegionMasks,
) -> tuple[ExperimentRows, ExperimentSummary]:
    simulation = Simulation(config=field_config)
    advance(simulation, config.warmup_steps)
    rows: ExperimentRows = []
    curves: list[np.ndarray] = []
    peaks: list[float] = []

    for trial in range(1, config.response_trials + 1):
        before = snapshot_metrics(simulation, regions)
        apply_impulse(simulation, regions, config.impulse)
        trial_rows = collect_response_rows(
            simulation,
            regions,
            baseline=before,
            horizon=config.response_horizon,
            experiment="response_stability",
            trial=trial,
        )
        rows.extend(trial_rows)
        curve = column(trial_rows, "right_delta")
        curves.append(curve)
        peaks.append(float(np.max(np.abs(curve))))
        advance(simulation, config.response_pause)

    correlations = pairwise_correlations(curves)
    summary = {
        "mean_pairwise_corr_right_delta": safe_nanmean(correlations),
        "min_pairwise_corr_right_delta": safe_nanmin(correlations),
        "max_pairwise_corr_right_delta": safe_nanmax(correlations),
        "peak_right_delta_mean": float(np.mean(peaks)),
        "peak_right_delta_std": float(np.std(peaks)),
        "peak_right_delta_cv": coefficient_of_variation(peaks),
    }
    return rows, summary


def run_propagation_distance(
    config: ExperimentAnalysisConfig,
    *,
    field_config: Any,
    regions: RegionMasks,
) -> tuple[ExperimentRows, ExperimentSummary]:
    simulation = Simulation(config=field_config)
    advance(simulation, config.warmup_steps)
    baseline = snapshot_metrics(simulation, regions)
    apply_impulse(simulation, regions, config.impulse)
    rows: ExperimentRows = []

    for offset in range(1, config.propagation_horizon + 1):
        row = step_metrics(simulation, regions)
        row["experiment"] = "propagation_distance"
        row["offset"] = offset
        row["left_delta"] = row["left_synchrony"] - baseline["left_synchrony"]
        row["assoc_delta"] = row["assoc_synchrony"] - baseline["assoc_synchrony"]
        row["right_delta"] = row["right_synchrony"] - baseline["right_synchrony"]
        row["left_delta_abs"] = abs(row["left_delta"])
        row["assoc_delta_abs"] = abs(row["assoc_delta"])
        row["right_delta_abs"] = abs(row["right_delta"])
        row["raw_right_left_ratio"] = row["right_synchrony"] / max(
            row["left_synchrony"],
            1e-12,
        )
        row["delta_right_left_ratio"] = row["right_delta_abs"] / max(
            row["left_delta_abs"],
            1e-12,
        )
        rows.append(row)

    left_delta = column(rows, "left_delta_abs")
    assoc_delta = column(rows, "assoc_delta_abs")
    right_delta = column(rows, "right_delta_abs")
    left_peak = propagation_metrics(left_delta)
    assoc_peak = propagation_metrics(assoc_delta)
    right_peak = propagation_metrics(right_delta)
    summary = {
        "baseline_left_synchrony": baseline["left_synchrony"],
        "baseline_right_synchrony": baseline["right_synchrony"],
        "left": left_peak,
        "assoc": assoc_peak,
        "right": right_peak,
        "peak_delta_right_left_ratio": right_peak["peak"]
        / max(
            left_peak["peak"],
            1e-12,
        ),
        "mean_raw_right_left_ratio": float(
            np.mean(column(rows, "raw_right_left_ratio"))
        ),
        "right_left_peak_delay_steps": right_peak["peak_step"] - left_peak["peak_step"],
        "right_left_half_delay_steps": right_peak["half_peak_step"]
        - left_peak["half_peak_step"],
        "increase_horizon_recommended": any(
            region["increase_horizon_recommended"]
            for region in (left_peak, assoc_peak, right_peak)
        ),
    }
    return rows, summary


def run_memory_experiment(
    config: ExperimentAnalysisConfig,
    *,
    field_config: Any,
    regions: RegionMasks,
) -> tuple[ExperimentRows, ExperimentSummary]:
    simulation = Simulation(config=field_config)
    advance(simulation, config.warmup_steps)
    pre_first = snapshot_metrics(simulation, regions)
    apply_impulse(simulation, regions, config.impulse)
    first_rows = collect_response_rows(
        simulation,
        regions,
        baseline=pre_first,
        horizon=config.memory_horizon,
        experiment="memory",
        pulse=1,
    )
    advance(simulation, config.memory_pause)
    pre_second = snapshot_metrics(simulation, regions)
    apply_impulse(simulation, regions, config.impulse)
    second_rows = collect_response_rows(
        simulation,
        regions,
        baseline=pre_second,
        horizon=config.memory_horizon,
        experiment="memory",
        pulse=2,
    )

    first_curve = column(first_rows, "right_delta")
    second_curve = column(second_rows, "right_delta")
    summary = {
        "right_response_corr": safe_correlation(first_curve, second_curve),
        "right_response_rmse": rmse(first_curve, second_curve),
        "first_peak_right_delta": float(np.max(np.abs(first_curve))),
        "second_peak_right_delta": float(np.max(np.abs(second_curve))),
        "pre_first_mean_trace": pre_first["mean_trace"],
        "pre_second_mean_trace": pre_second["mean_trace"],
        "pre_first_left_trace": pre_first["left_trace"],
        "pre_second_left_trace": pre_second["left_trace"],
        "pre_first_right_trace": pre_first["right_trace"],
        "pre_second_right_trace": pre_second["right_trace"],
    }
    return first_rows + second_rows, summary


def collect_response_rows(
    simulation: Simulation,
    regions: RegionMasks,
    *,
    baseline: dict[str, float],
    horizon: int,
    experiment: str,
    trial: int | None = None,
    pulse: int | None = None,
) -> ExperimentRows:
    rows: ExperimentRows = []
    for offset in range(1, horizon + 1):
        row = step_metrics(simulation, regions)
        row["experiment"] = experiment
        row["offset"] = offset
        if trial is not None:
            row["trial"] = trial
        if pulse is not None:
            row["pulse"] = pulse
        row["left_delta"] = row["left_synchrony"] - baseline["left_synchrony"]
        row["assoc_delta"] = row["assoc_synchrony"] - baseline["assoc_synchrony"]
        row["right_delta"] = row["right_synchrony"] - baseline["right_synchrony"]
        rows.append(row)
    return rows


def snapshot_metrics(simulation: Simulation, regions: RegionMasks) -> dict[str, Any]:
    frame = simulation.snapshot()
    local = frame.local_synchrony
    return {
        "step": float(simulation.step_index),
        "global_synchrony": frame.metrics.global_synchrony,
        "mean_metabolite": frame.metrics.mean_metabolite,
        "mean_trace": frame.metrics.mean_trace,
        "left_synchrony": region_mean(local, regions.input),
        "assoc_synchrony": region_mean(local, regions.assoc),
        "right_synchrony": region_mean(local, regions.output),
        "left_trace": region_mean(frame.state.trace, regions.input),
        "assoc_trace": region_mean(frame.state.trace, regions.assoc),
        "right_trace": region_mean(frame.state.trace, regions.output),
    }


def step_metrics(simulation: Simulation, regions: RegionMasks) -> dict[str, Any]:
    simulation.step()
    return snapshot_metrics(simulation, regions)


def apply_impulse(simulation: Simulation, regions: RegionMasks, amount: float) -> None:
    simulation.field.apply_phase_impulse(regions.input, amount)


def advance(simulation: Simulation, steps: int) -> None:
    for _ in range(steps):
        simulation.step()


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask]))


def column(rows: ExperimentRows, key: str) -> np.ndarray:
    return np.array([row[key] for row in rows], dtype=np.float64)


def peak_metrics(values: np.ndarray) -> dict[str, float | int]:
    peak = float(np.max(values))
    peak_step = int(np.argmax(values) + 1)
    half_peak_step = time_to_fraction(values, 0.5)
    area = float(np.sum(values))
    return {
        "peak": peak,
        "peak_step": peak_step,
        "half_peak_step": half_peak_step,
        "area": area,
    }


def propagation_metrics(values: np.ndarray) -> dict[str, float | int | bool]:
    metrics = peak_metrics(values)
    tail = tail_metrics(values)
    metrics.update(tail)
    metrics["increase_horizon_recommended"] = bool(
        metrics["peak_at_window_end"] or tail["tail_slope"] > 0.0
    )
    return metrics


def tail_metrics(
    values: np.ndarray, *, tail_fraction: float = 0.1
) -> dict[str, float | bool]:
    if values.size == 0:
        return {
            "peak_at_window_end": False,
            "tail_slope": 0.0,
            "tail_mean_delta": 0.0,
            "tail_last_delta": 0.0,
        }

    tail_length = max(2, round(values.size * tail_fraction))
    tail = values[-tail_length:]
    x = np.arange(tail.size, dtype=np.float64)
    slope = float(np.polyfit(x, tail, deg=1)[0]) if tail.size > 1 else 0.0
    return {
        "peak_at_window_end": int(np.argmax(values)) == values.size - 1,
        "tail_slope": slope,
        "tail_mean_delta": float(np.mean(tail)),
        "tail_last_delta": float(tail[-1]),
    }


def time_to_fraction(values: np.ndarray, fraction: float) -> int:
    peak = float(np.max(values))
    if peak <= 0.0:
        return -1
    threshold = peak * fraction
    indices = np.flatnonzero(values >= threshold)
    return int(indices[0] + 1) if indices.size else -1


def pairwise_correlations(curves: list[np.ndarray]) -> np.ndarray:
    correlations: list[float] = []
    for left_index in range(len(curves)):
        for right_index in range(left_index + 1, len(curves)):
            correlations.append(
                safe_correlation(curves[left_index], curves[right_index])
            )
    return np.array(correlations, dtype=np.float64)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def safe_nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else float("nan")


def safe_nanmin(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if values.size else float("nan")


def safe_nanmax(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if values.size else float("nan")


def coefficient_of_variation(values: list[float]) -> float:
    mean = float(np.mean(values))
    if mean == 0.0:
        return float("nan")
    return float(np.std(values) / mean)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def write_rows(path: Path, rows: ExperimentRows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> ExperimentRows:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def export_experiment_plots(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_response_stability(
            output_dir / "experiment_1_response_stability.csv",
            plot_dir / "response-stability.png",
        ),
        plot_propagation(
            output_dir / "experiment_2_propagation_distance.csv",
            plot_dir / "propagation-distance.png",
        ),
        plot_memory(
            output_dir / "experiment_3_memory.csv",
            plot_dir / "memory.png",
        ),
    ]
    return paths


def plot_response_stability(csv_path: Path, output_path: Path) -> Path:
    rows = read_rows(csv_path)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    trials = sorted({int(row["trial"]) for row in rows})
    for trial in trials:
        subset = [row for row in rows if int(row["trial"]) == trial]
        axis.plot(
            [int(row["offset"]) for row in subset],
            [float(row["right_delta"]) for row in subset],
            alpha=0.6,
        )
    axis.set_title("Response stability: right synchrony delta")
    axis.set_xlabel("steps after impulse")
    axis.set_ylabel("right synchrony delta")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def plot_propagation(csv_path: Path, output_path: Path) -> Path:
    rows = read_rows(csv_path)
    x = [int(row["offset"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(x, [float(row["left_delta_abs"]) for row in rows], label="left")
    axis.plot(x, [float(row["assoc_delta_abs"]) for row in rows], label="assoc")
    axis.plot(x, [float(row["right_delta_abs"]) for row in rows], label="right")
    axis.set_title("Propagation distance: synchrony delta")
    axis.set_xlabel("steps after impulse")
    axis.set_ylabel("abs synchrony delta")
    axis.legend()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def plot_memory(csv_path: Path, output_path: Path) -> Path:
    rows = read_rows(csv_path)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for pulse in (1, 2):
        subset = [row for row in rows if int(row["pulse"]) == pulse]
        axis.plot(
            [int(row["offset"]) for row in subset],
            [float(row["right_delta"]) for row in subset],
            label=f"pulse {pulse}",
        )
    axis.set_title("Memory: pulse 1 vs pulse 2")
    axis.set_xlabel("steps after impulse")
    axis.set_ylabel("right synchrony delta")
    axis.legend()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run response, propagation, and memory experiments.",
    )
    parser.add_argument(
        "--config", type=Path, default=ExperimentAnalysisConfig.config_path
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ExperimentAnalysisConfig.output_dir
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--impulse", type=float, default=0.45)
    parser.add_argument("--response-trials", type=int, default=10)
    parser.add_argument("--response-horizon", type=int, default=64)
    parser.add_argument("--response-pause", type=int, default=96)
    parser.add_argument("--propagation-horizon", type=int, default=1024)
    parser.add_argument("--memory-horizon", type=int, default=64)
    parser.add_argument("--memory-pause", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_experiment_suite(
        ExperimentAnalysisConfig(
            config_path=args.config,
            output_dir=args.output_dir,
            warmup_steps=args.warmup_steps,
            impulse=args.impulse,
            response_trials=args.response_trials,
            response_horizon=args.response_horizon,
            response_pause=args.response_pause,
            propagation_horizon=args.propagation_horizon,
            memory_horizon=args.memory_horizon,
            memory_pause=args.memory_pause,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0
