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

from neuroacoustic_resonator.analysis.metrics import (
    RegionalActivityMetrics,
)
from neuroacoustic_resonator.analysis.protocol_stream import ProtocolAnalysisStream
from neuroacoustic_resonator.configuration import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.topology import GridTopology

ProbeRows = list[dict[str, Any]]
ProbeSummary = dict[str, Any]


@dataclass(frozen=True)
class PropagationProbeConfig:
    config_path: Path = Path("configs") / "synthetic_input.yaml"
    output_csv: Path = Path("experiments") / "logs" / "propagation_probe.csv"
    output_summary: Path = (
        Path("experiments") / "logs" / "propagation_probe_summary.json"
    )
    output_plot: Path | None = Path("experiments") / "logs" / "propagation_probe.png"
    warmup_steps: int = 200
    horizon: int = 512
    horizons: tuple[int, ...] = ()
    impulse: float = 0.45
    response_threshold: float = 0.02
    seeds: tuple[int, ...] = ()
    uncoupled_control: bool = False

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
            raise ValueError(msg)
        if self.horizon < 1:
            msg = "horizon must be positive"
            raise ValueError(msg)
        if any(horizon < 1 for horizon in self.horizons):
            msg = "horizons must be positive"
            raise ValueError(msg)
        if self.impulse < 0.0:
            msg = "impulse must be non-negative"
            raise ValueError(msg)
        if self.response_threshold < 0.0:
            msg = "response_threshold must be non-negative"
            raise ValueError(msg)
        if len(set(self.seeds)) != len(self.seeds):
            msg = "seeds must not contain duplicates"
            raise ValueError(msg)


def run_propagation_probe(config: PropagationProbeConfig) -> ProbeSummary:
    if config.seeds or config.uncoupled_control:
        return run_controlled_probe(config)
    if config.horizons:
        return run_multi_horizon_probe(config)

    rows, baseline = collect_probe_rows(config, horizon=config.horizon)
    summary = summarize_probe_rows(
        rows,
        baseline=baseline,
        config=config,
        horizon=config.horizon,
    )
    summary["environment"] = propagation_environment(
        SimulationConfig.from_file(config.config_path)
    )
    write_probe_rows(config.output_csv, rows)
    write_probe_summary(config.output_summary, summary)
    if config.output_plot is not None:
        write_probe_plot(config.output_plot, rows)
        summary["plot_path"] = str(config.output_plot)
        write_probe_summary(config.output_summary, summary)
    return summary


def run_controlled_probe(config: PropagationProbeConfig) -> ProbeSummary:
    source_config = SimulationConfig.from_file(config.config_path)
    seeds = controlled_seeds(config, source_config)
    horizons = config.horizons or (config.horizon,)
    all_rows: ProbeRows = []
    trials: list[dict[str, Any]] = []
    for seed in seeds:
        coupled_config = simulation_config_for_trial(source_config, seed=seed)
        conditions = [
            ("coupled", coupled_config, config.impulse),
            ("no_impulse", coupled_config, 0.0),
        ]
        if config.uncoupled_control:
            uncoupled_config = simulation_config_for_trial(
                source_config,
                seed=seed,
                uncoupled=True,
            )
            conditions.append(
                (
                    "uncoupled",
                    uncoupled_config,
                    config.impulse,
                )
            )
            conditions.append(("uncoupled_no_impulse", uncoupled_config, 0.0))
        for horizon in horizons:
            horizon_summaries: dict[str, ProbeSummary] = {}
            condition_rows: dict[str, ProbeRows] = {}
            for condition, trial_config, impulse in conditions:
                rows, baseline = collect_probe_rows(
                    config,
                    horizon=horizon,
                    simulation_config=trial_config,
                    impulse=impulse,
                )
                for row in rows:
                    row["seed"] = seed
                    row["condition"] = condition
                    row["horizon"] = horizon
                all_rows.extend(rows)
                condition_rows[condition] = rows
                condition_summary = summarize_probe_rows(
                    rows,
                    baseline=baseline,
                    config=config,
                    horizon=horizon,
                )
                condition_summary["field"] = {
                    "seed": trial_config.field.seed,
                    "coupling_strength": trial_config.field.coupling_strength,
                    "coupling_homeostasis_rate": (
                        trial_config.field.coupling_homeostasis_rate
                    ),
                }
                horizon_summaries[condition] = condition_summary
            trial: dict[str, Any] = {
                "seed": seed,
                "horizon": horizon,
                "conditions": horizon_summaries,
            }
            if config.uncoupled_control:
                trial["comparison"] = compare_controlled_conditions(
                    horizon_summaries["coupled"],
                    condition_rows["coupled"],
                    condition_rows["no_impulse"],
                    condition_rows["uncoupled"],
                    condition_rows["uncoupled_no_impulse"],
                    response_threshold=config.response_threshold,
                )
            trials.append(trial)

    summary = {
        "config": str(config.config_path),
        "parameters": {
            "warmup_steps": config.warmup_steps,
            "horizons": list(horizons),
            "impulse": config.impulse,
            "response_threshold": config.response_threshold,
            "seeds": list(seeds),
            "uncoupled_control": config.uncoupled_control,
        },
        "environment": propagation_environment(source_config),
        "trials": trials,
        "aggregate": aggregate_controlled_trials(trials),
    }
    write_probe_rows(config.output_csv, all_rows)
    write_probe_summary(config.output_summary, summary)
    if config.output_plot is not None:
        write_probe_plot(config.output_plot, all_rows)
        summary["plot_path"] = str(config.output_plot)
        write_probe_summary(config.output_summary, summary)
    return summary


def controlled_seeds(
    config: PropagationProbeConfig,
    source_config: SimulationConfig,
) -> tuple[int, ...]:
    if config.seeds:
        return config.seeds
    if source_config.field.seed is None:
        msg = "controlled propagation requires explicit seeds"
        raise ValueError(msg)
    return (source_config.field.seed,)


def simulation_config_for_trial(
    source: SimulationConfig,
    *,
    seed: int,
    uncoupled: bool = False,
) -> SimulationConfig:
    changes: dict[str, Any] = {"seed": seed}
    if uncoupled:
        changes.update(
            coupling_strength=0.0,
            coupling_homeostasis_rate=0.0,
        )
    field = source.field.model_copy(update=changes)
    return source.model_copy(update={"field": field})


def run_multi_horizon_probe(config: PropagationProbeConfig) -> ProbeSummary:
    all_rows: ProbeRows = []
    horizon_summaries: dict[str, Any] = {}
    for horizon in config.horizons:
        rows, baseline = collect_probe_rows(config, horizon=horizon)
        for row in rows:
            row["horizon"] = horizon
        all_rows.extend(rows)
        horizon_summaries[str(horizon)] = summarize_probe_rows(
            rows,
            baseline=baseline,
            config=config,
            horizon=horizon,
        )

    summary: ProbeSummary = {
        "config": str(config.config_path),
        "parameters": {
            "warmup_steps": config.warmup_steps,
            "horizons": list(config.horizons),
            "impulse": config.impulse,
            "response_threshold": config.response_threshold,
        },
        "environment": propagation_environment(
            SimulationConfig.from_file(config.config_path)
        ),
        "horizons": horizon_summaries,
    }
    write_probe_rows(config.output_csv, all_rows)
    write_probe_summary(config.output_summary, summary)
    if config.output_plot is not None:
        write_probe_plot(config.output_plot, all_rows)
        summary["plot_path"] = str(config.output_plot)
        write_probe_summary(config.output_summary, summary)
    return summary


def collect_probe_rows(
    config: PropagationProbeConfig,
    *,
    horizon: int,
    simulation_config: SimulationConfig | None = None,
    impulse: float | None = None,
) -> tuple[ProbeRows, RegionalActivityMetrics]:
    sim_config = simulation_config or SimulationConfig.from_file(config.config_path)
    simulation = sim_config.create_simulation()
    regions = RegionMasks.from_size(sim_config.field.size)
    protocol_stream = ProtocolAnalysisStream.from_config(sim_config, regions)

    baseline_observation = protocol_stream.observe(
        simulation.snapshot(),
        input_value=simulation.last_input_value,
    )
    if baseline_observation is None:
        msg = "protocol must emit the initial propagation frame"
        raise RuntimeError(msg)
    for _ in range(config.warmup_steps):
        observation = protocol_stream.observe(
            simulation.step(),
            input_value=simulation.last_input_value,
        )
        if observation is not None:
            baseline_observation = observation

    baseline = baseline_observation.activity
    applied_impulse = config.impulse if impulse is None else impulse
    if applied_impulse != 0.0:
        simulation.field.apply_phase_impulse(regions.input, applied_impulse)

    rows: ProbeRows = []
    previous = baseline
    for offset in range(1, horizon + 1):
        observation = protocol_stream.observe(
            simulation.step(),
            input_value=simulation.last_input_value,
        )
        if observation is None:
            continue
        metrics = observation.activity
        row = propagation_probe_row(
            metrics,
            baseline=baseline,
            previous=previous,
            offset=offset,
        )
        row["sequence"] = observation.frame.sequence
        row["version"] = observation.frame.version
        row["time_seconds"] = observation.frame.time_seconds
        rows.append(row)
        previous = metrics

    return rows, baseline


def propagation_probe_row(
    metrics: RegionalActivityMetrics,
    *,
    baseline: RegionalActivityMetrics,
    previous: RegionalActivityMetrics,
    offset: int,
) -> dict[str, Any]:
    input_delta = metrics.input_activity - baseline.input_activity
    assoc_delta = metrics.assoc_activity - baseline.assoc_activity
    output_delta = metrics.output_activity - baseline.output_activity
    output_slope = metrics.output_activity - previous.output_activity
    return {
        "step": metrics.step,
        "offset": offset,
        "input_value": metrics.input_value,
        "input_activity": metrics.input_activity,
        "input_fast_activity": metrics.input_fast_activity,
        "input_slow_activity": metrics.input_slow_activity,
        "assoc_activity": metrics.assoc_activity,
        "assoc_fast_activity": metrics.assoc_fast_activity,
        "assoc_slow_activity": metrics.assoc_slow_activity,
        "output_activity": metrics.output_activity,
        "output_fast_activity": metrics.output_fast_activity,
        "output_slow_activity": metrics.output_slow_activity,
        "input_activity_delta": input_delta,
        "assoc_activity_delta": assoc_delta,
        "output_activity_delta": output_delta,
        "input_activity_delta_abs": abs(input_delta),
        "assoc_activity_delta_abs": abs(assoc_delta),
        "output_activity_delta_abs": abs(output_delta),
        "output_activity_slope": output_slope,
        "left_to_right_ratio": metrics.left_to_right_ratio,
        "output_trace": metrics.output_trace,
        "output_synchrony": metrics.output_synchrony,
        "output_fast_delta": metrics.output_fast_delta,
        "output_slow_delta": metrics.output_slow_delta,
        "output_event_score": metrics.output_event_score,
        "output_fast_response_score": metrics.output_fast_response_score,
        "output_slow_drift_score": metrics.output_slow_drift_score,
        "delta_right_left_ratio": abs(output_delta) / max(abs(input_delta), 1e-12),
    }


def summarize_probe_rows(
    rows: ProbeRows,
    *,
    baseline: RegionalActivityMetrics,
    config: PropagationProbeConfig,
    horizon: int,
) -> ProbeSummary:
    if not rows:
        msg = "probe rows must not be empty"
        raise ValueError(msg)

    output_delta = column(rows, "output_activity_delta")
    output_event_score = column(rows, "output_event_score")
    output_fast_score = column(rows, "output_fast_response_score")
    output_slow_score = column(rows, "output_slow_drift_score")
    left_to_right_ratio = column(rows, "left_to_right_ratio")
    input_delta = column(rows, "input_activity_delta")
    assoc_delta = column(rows, "assoc_activity_delta")
    output_abs = np.abs(output_delta)
    input_abs = np.abs(input_delta)
    assoc_abs = np.abs(assoc_delta)
    peak_input_index = int(np.argmax(input_abs))
    peak_assoc_index = int(np.argmax(assoc_abs))
    peak_output_index = int(np.argmax(output_abs))
    peak_event_index = int(np.argmax(output_event_score))
    input_latency = first_threshold_crossing(input_abs, config.response_threshold)
    assoc_latency = first_threshold_crossing(assoc_abs, config.response_threshold)
    output_latency = first_threshold_crossing(output_abs, config.response_threshold)
    input_assoc = lagged_correlation(input_delta, assoc_delta)
    assoc_output = lagged_correlation(assoc_delta, output_delta)
    input_output = lagged_correlation(input_delta, output_delta)
    return {
        "config": str(config.config_path),
        "parameters": {
            "warmup_steps": config.warmup_steps,
            "horizon": horizon,
            "impulse": config.impulse,
            "response_threshold": config.response_threshold,
        },
        "baseline": asdict(baseline),
        "peak_input_activity_delta": float(input_abs[peak_input_index]),
        "peak_input_activity_step": int(rows[peak_input_index]["offset"]),
        "peak_assoc_activity_delta": float(assoc_abs[peak_assoc_index]),
        "peak_assoc_activity_step": int(rows[peak_assoc_index]["offset"]),
        "peak_output_activity_delta": float(output_abs[peak_output_index]),
        "peak_output_activity_step": int(rows[peak_output_index]["offset"]),
        "input_peak_at_horizon_end": peak_input_index == len(rows) - 1,
        "assoc_peak_at_horizon_end": peak_assoc_index == len(rows) - 1,
        "output_peak_at_horizon_end": peak_output_index == len(rows) - 1,
        "peak_output_event_score": float(output_event_score[peak_event_index]),
        "peak_output_event_step": int(rows[peak_event_index]["offset"]),
        "peak_output_fast_response_score": float(np.max(output_fast_score)),
        "peak_output_slow_drift_score": float(np.max(output_slow_score)),
        "slow_fast_peak_ratio": float(
            np.max(output_slow_score) / max(np.max(output_fast_score), 1e-12)
        ),
        "mean_left_to_right_ratio": float(np.mean(left_to_right_ratio)),
        "max_left_to_right_ratio": float(np.max(left_to_right_ratio)),
        "input_latency_steps": input_latency,
        "assoc_latency_steps": assoc_latency,
        "response_latency_steps": output_latency,
        "response_reached_assoc": assoc_latency is not None,
        "response_reached_output": output_latency is not None,
        "peak_delta_right_left_ratio": float(
            output_abs[peak_output_index] / max(np.max(input_abs), 1e-12)
        ),
        "peak_delta_assoc_input_ratio": float(
            assoc_abs[peak_assoc_index] / max(np.max(input_abs), 1e-12)
        ),
        "peak_delta_output_assoc_ratio": float(
            output_abs[peak_output_index] / max(np.max(assoc_abs), 1e-12)
        ),
        "input_to_assoc": input_assoc,
        "assoc_to_output": assoc_output,
        "input_to_output": input_output,
    }


def compare_controlled_conditions(
    coupled: ProbeSummary,
    coupled_rows: ProbeRows,
    no_impulse_rows: ProbeRows,
    uncoupled_rows: ProbeRows,
    uncoupled_no_impulse_rows: ProbeRows,
    *,
    response_threshold: float,
) -> dict[str, Any]:
    causal_input = controlled_effect(
        coupled_rows,
        no_impulse_rows,
        "input_activity_delta",
    )
    causal_assoc = controlled_effect(
        coupled_rows,
        no_impulse_rows,
        "assoc_activity_delta",
    )
    causal_output = controlled_effect(
        coupled_rows,
        no_impulse_rows,
        "output_activity_delta",
    )
    uncoupled_output = controlled_effect(
        uncoupled_rows,
        uncoupled_no_impulse_rows,
        "output_activity_delta",
    )
    input_latency = first_threshold_crossing(causal_input, response_threshold)
    assoc_latency = first_threshold_crossing(causal_assoc, response_threshold)
    output_latency = first_threshold_crossing(causal_output, response_threshold)
    causal_input_peak = peak_value_and_step(causal_input)
    causal_assoc_peak = peak_value_and_step(causal_assoc)
    causal_output_peak = peak_value_and_step(causal_output)
    control_peak = float(np.max(uncoupled_output))
    causal_to_control_ratio = (
        None if control_peak <= 1e-12 else causal_output_peak[0] / control_peak
    )
    assoc_precedes_output = (
        isinstance(assoc_latency, int)
        and isinstance(output_latency, int)
        and assoc_latency < output_latency
    )
    output_above_control = causal_output_peak[0] > control_peak
    return {
        "input_latency_steps": input_latency,
        "assoc_latency_steps": assoc_latency,
        "output_latency_steps": output_latency,
        "causal_peak_input_activity_delta": causal_input_peak[0],
        "causal_peak_input_activity_step": causal_input_peak[1],
        "causal_peak_assoc_activity_delta": causal_assoc_peak[0],
        "causal_peak_assoc_activity_step": causal_assoc_peak[1],
        "causal_peak_output_activity_delta": causal_output_peak[0],
        "causal_peak_output_activity_step": causal_output_peak[1],
        "coupled_peak_output_activity_delta": float(
            coupled["peak_output_activity_delta"]
        ),
        "uncoupled_peak_output_activity_delta": control_peak,
        "causal_to_uncoupled_peak_ratio": causal_to_control_ratio,
        "assoc_precedes_output": assoc_precedes_output,
        "output_above_control": output_above_control,
        "passed": bool(
            output_latency is not None
            and assoc_precedes_output
            and output_above_control
        ),
    }


def controlled_effect(
    coupled_rows: ProbeRows,
    no_impulse_rows: ProbeRows,
    key: str,
) -> np.ndarray:
    if len(coupled_rows) != len(no_impulse_rows):
        msg = "controlled propagation rows must have matching lengths"
        raise ValueError(msg)
    return np.abs(column(coupled_rows, key) - column(no_impulse_rows, key))


def peak_value_and_step(values: np.ndarray) -> tuple[float, int | None]:
    index = int(np.argmax(values))
    peak = float(values[index])
    return peak, None if peak <= 1e-12 else index + 1


def aggregate_controlled_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [trial["comparison"] for trial in trials if "comparison" in trial]
    coupled = [trial["conditions"]["coupled"] for trial in trials]
    if not comparisons:
        reached_output = [
            bool(summary["response_reached_output"]) for summary in coupled
        ]
        return {
            "trials": len(trials),
            "coupled_output_reach_rate": float(np.mean(reached_output)),
            "gate_passed": False,
        }
    reached_output = [
        comparison["output_latency_steps"] is not None for comparison in comparisons
    ]
    ratios = [
        float(comparison["causal_to_uncoupled_peak_ratio"])
        for comparison in comparisons
        if comparison["causal_to_uncoupled_peak_ratio"] is not None
    ]
    passed = [bool(comparison["passed"]) for comparison in comparisons]
    return {
        "trials": len(trials),
        "coupled_output_reach_rate": float(np.mean(reached_output)),
        "assoc_precedes_output_rate": float(
            np.mean(
                [
                    bool(comparison["assoc_precedes_output"])
                    for comparison in comparisons
                ]
            )
        ),
        "output_above_control_rate": float(
            np.mean(
                [bool(comparison["output_above_control"]) for comparison in comparisons]
            )
        ),
        "mean_causal_to_uncoupled_peak_ratio": (
            None if not ratios else float(np.mean(ratios))
        ),
        "passed_trials": int(sum(passed)),
        "gate_passed": all(passed),
    }


def propagation_environment(config: SimulationConfig) -> dict[str, Any]:
    regions = RegionMasks.from_size(config.field.size)
    topology = GridTopology.from_size(
        config.field.size,
        boundary_x=config.field.boundary_x,
        boundary_y=config.field.boundary_y,
    )
    return {
        "size": config.field.size,
        "boundary_x": config.field.boundary_x,
        "boundary_y": config.field.boundary_y,
        "graph_distances": {
            "input_to_assoc": minimum_mask_distance(
                regions.input,
                regions.assoc,
                topology,
            ),
            "assoc_to_output": minimum_mask_distance(
                regions.assoc,
                regions.output,
                topology,
            ),
            "input_to_output": minimum_mask_distance(
                regions.input,
                regions.output,
                topology,
            ),
        },
    }


def minimum_mask_distance(
    source: np.ndarray,
    target: np.ndarray,
    topology: GridTopology,
) -> int:
    if source.shape != topology.shape or target.shape != topology.shape:
        msg = "region masks and topology must have matching shapes"
        raise ValueError(msg)
    if not np.any(source) or not np.any(target):
        msg = "region masks must not be empty"
        raise ValueError(msg)
    target_coordinates = np.argwhere(target)
    best = sum(topology.shape)
    for source_y, source_x in np.argwhere(source):
        delta_y = np.abs(target_coordinates[:, 0] - source_y)
        delta_x = np.abs(target_coordinates[:, 1] - source_x)
        if topology.boundary_y == "periodic":
            delta_y = np.minimum(delta_y, topology.shape[0] - delta_y)
        if topology.boundary_x == "periodic":
            delta_x = np.minimum(delta_x, topology.shape[1] - delta_x)
        best = min(best, int(np.min(delta_y + delta_x)))
    return best


def write_probe_rows(path: str | Path, rows: ProbeRows) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_probe_summary(path: str | Path, summary: ProbeSummary) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def write_probe_plot(path: str | Path, rows: ProbeRows) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    horizons = sorted({int(row.get("horizon", 0)) for row in rows})
    if "condition" in rows[0]:
        groups = sorted(
            {
                (int(row["seed"]), str(row["condition"]), int(row["horizon"]))
                for row in rows
            }
        )
        for seed, condition, horizon in groups:
            plot_rows = [
                row
                for row in rows
                if int(row["seed"]) == seed
                and str(row["condition"]) == condition
                and int(row["horizon"]) == horizon
            ]
            axis.plot(
                column(plot_rows, "offset"),
                column(plot_rows, "output_activity_delta"),
                label=f"{condition} seed={seed} h={horizon}",
                alpha=0.75,
            )
    elif horizons == [0]:
        plot_rows = rows
        axis.plot(
            column(plot_rows, "offset"),
            column(plot_rows, "input_activity_delta"),
            label="input delta",
        )
        axis.plot(
            column(plot_rows, "offset"),
            column(plot_rows, "assoc_activity_delta"),
            label="assoc delta",
        )
        axis.plot(
            column(plot_rows, "offset"),
            column(plot_rows, "output_activity_delta"),
            label="output delta",
        )
    else:
        for horizon in horizons:
            plot_rows = [row for row in rows if int(row["horizon"]) == horizon]
            axis.plot(
                column(plot_rows, "offset"),
                column(plot_rows, "output_activity_delta"),
                label=f"output delta h={horizon}",
                alpha=0.8,
            )
        longest_horizon = max(horizons)
        longest_rows = [row for row in rows if int(row["horizon"]) == longest_horizon]
        axis.plot(
            column(longest_rows, "offset"),
            column(longest_rows, "input_activity_delta"),
            label=f"input delta h={longest_horizon}",
            linewidth=1.0,
            alpha=0.6,
        )
        axis.plot(
            column(longest_rows, "offset"),
            column(longest_rows, "assoc_activity_delta"),
            label=f"assoc delta h={longest_horizon}",
            linewidth=1.0,
            alpha=0.6,
        )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set_title("Baseline-corrected propagation: input -> assoc -> output")
    axis.set_xlabel("steps after input impulse")
    axis.set_ylabel("activity delta from baseline")
    axis.legend()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def column(rows: ProbeRows, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def first_threshold_crossing(values: np.ndarray, threshold: float) -> int | None:
    crossings = np.flatnonzero(values >= threshold)
    if crossings.size == 0:
        return None
    return int(crossings[0] + 1)


def lagged_correlation(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_lag: int | None = None,
) -> dict[str, float | int | None]:
    if source.size != target.size:
        msg = "source and target must have matching sizes"
        raise ValueError(msg)
    if source.size < 3:
        return {"correlation": None, "lag_steps": None}

    max_lag = min(source.size - 2, 256 if max_lag is None else max_lag)
    best_correlation: float | None = None
    best_lag: int | None = None
    for lag in range(max_lag + 1):
        left = source[: source.size - lag] if lag else source
        right = target[lag:] if lag else target
        correlation = safe_correlation(left, right)
        if correlation is None:
            continue
        if best_correlation is None or abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_lag = lag

    return {
        "correlation": best_correlation,
        "lag_steps": best_lag,
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size < 3:
        return None
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe left-to-right propagation after an input-region impulse.",
    )
    parser.add_argument(
        "--config", type=Path, default=PropagationProbeConfig.config_path
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--horizons", type=int, nargs="*", default=())
    parser.add_argument("--impulse", type=float, default=0.45)
    parser.add_argument("--response-threshold", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, nargs="*", default=())
    parser.add_argument("--uncoupled-control", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PropagationProbeConfig.output_csv,
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=PropagationProbeConfig.output_summary,
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PropagationProbeConfig.output_plot,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PropagationProbeConfig(
        config_path=args.config,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        output_plot=args.output_plot,
        warmup_steps=args.warmup_steps,
        horizon=args.horizon,
        horizons=tuple(args.horizons),
        impulse=args.impulse,
        response_threshold=args.response_threshold,
        seeds=tuple(args.seeds),
        uncoupled_control=args.uncoupled_control,
    )
    summary = run_propagation_probe(config)
    if "aggregate" in summary:
        aggregate = summary["aggregate"]
        print(
            "Controlled propagation probe: "
            f"trials={aggregate['trials']} "
            f"reached_output={aggregate['coupled_output_reach_rate']:.3f} "
            f"assoc_precedes_output={aggregate.get('assoc_precedes_output_rate', 0.0):.3f} "
            f"above_control={aggregate.get('output_above_control_rate', 0.0):.3f} "
            f"gate_passed={aggregate['gate_passed']}"
        )
        horizon_summary = None
    elif "horizons" in summary:
        horizon_summary = summary["horizons"][str(config.horizons[-1])]
    else:
        horizon_summary = summary
    if horizon_summary is not None:
        print(
            "Propagation probe: "
            f"reached_output={horizon_summary['response_reached_output']} "
            f"latency={horizon_summary['response_latency_steps']} "
            f"peak_output_delta={horizon_summary['peak_output_activity_delta']:.6f} "
            f"peak_ratio={horizon_summary['peak_delta_right_left_ratio']:.6f}"
        )
    print(f"Wrote rows: {config.output_csv}")
    print(f"Wrote summary: {config.output_summary}")
    if config.output_plot is not None:
        print(f"Wrote plot: {config.output_plot}")
    return 0
