from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.metrics import (
    RegionalActivityMetrics,
    RegionalActivityTracker,
)
from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation

ProbeRows = list[dict[str, Any]]
ProbeSummary = dict[str, Any]


@dataclass(frozen=True)
class PropagationProbeConfig:
    config_path: Path = Path("configs") / "synthetic_input.yaml"
    output_csv: Path = Path("experiments") / "logs" / "propagation_probe.csv"
    output_summary: Path = (
        Path("experiments") / "logs" / "propagation_probe_summary.json"
    )
    warmup_steps: int = 200
    horizon: int = 512
    impulse: float = 0.45
    response_threshold: float = 0.02

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
            raise ValueError(msg)
        if self.horizon < 1:
            msg = "horizon must be positive"
            raise ValueError(msg)
        if self.impulse < 0.0:
            msg = "impulse must be non-negative"
            raise ValueError(msg)
        if self.response_threshold < 0.0:
            msg = "response_threshold must be non-negative"
            raise ValueError(msg)


def run_propagation_probe(config: PropagationProbeConfig) -> ProbeSummary:
    sim_config = SimulationConfig.from_file(config.config_path)
    simulation = Simulation.from_config(sim_config)
    regions = RegionMasks.from_size(sim_config.field.size)
    tracker = RegionalActivityTracker()

    for _ in range(config.warmup_steps):
        simulation.step()

    baseline_frame = simulation.snapshot()
    baseline = tracker.update(
        baseline_frame,
        regions,
        input_value=simulation.last_input_value,
    )
    simulation.field.apply_phase_impulse(regions.input, config.impulse)

    rows: ProbeRows = []
    previous = baseline
    for offset in range(1, config.horizon + 1):
        frame = simulation.step()
        metrics = tracker.update(
            frame,
            regions,
            input_value=simulation.last_input_value,
        )
        row = propagation_probe_row(
            metrics,
            baseline=baseline,
            previous=previous,
            offset=offset,
        )
        rows.append(row)
        previous = metrics

    summary = summarize_probe_rows(
        rows,
        baseline=baseline,
        config=config,
    )
    write_probe_rows(config.output_csv, rows)
    write_probe_summary(config.output_summary, summary)
    return summary


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
    peak_output_index = int(np.argmax(output_abs))
    peak_event_index = int(np.argmax(output_event_score))
    response_latency = first_threshold_crossing(
        output_abs,
        config.response_threshold,
    )
    return {
        "config": str(config.config_path),
        "parameters": {
            "warmup_steps": config.warmup_steps,
            "horizon": config.horizon,
            "impulse": config.impulse,
            "response_threshold": config.response_threshold,
        },
        "baseline": asdict(baseline),
        "peak_input_activity_delta": float(np.max(input_abs)),
        "peak_assoc_activity_delta": float(np.max(assoc_abs)),
        "peak_output_activity_delta": float(output_abs[peak_output_index]),
        "peak_output_activity_step": int(rows[peak_output_index]["offset"]),
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
        "response_latency_steps": response_latency,
        "response_reached_output": response_latency is not None,
        "peak_delta_right_left_ratio": float(
            output_abs[peak_output_index] / max(np.max(input_abs), 1e-12)
        ),
    }


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


def column(rows: ProbeRows, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def first_threshold_crossing(values: np.ndarray, threshold: float) -> int | None:
    crossings = np.flatnonzero(values >= threshold)
    if crossings.size == 0:
        return None
    return int(crossings[0] + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe left-to-right propagation after an input-region impulse.",
    )
    parser.add_argument(
        "--config", type=Path, default=PropagationProbeConfig.config_path
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--impulse", type=float, default=0.45)
    parser.add_argument("--response-threshold", type=float, default=0.02)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PropagationProbeConfig(
        config_path=args.config,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        warmup_steps=args.warmup_steps,
        horizon=args.horizon,
        impulse=args.impulse,
        response_threshold=args.response_threshold,
    )
    summary = run_propagation_probe(config)
    print(
        "Propagation probe: "
        f"reached_output={summary['response_reached_output']} "
        f"latency={summary['response_latency_steps']} "
        f"peak_output_delta={summary['peak_output_activity_delta']:.6f} "
        f"peak_ratio={summary['peak_delta_right_left_ratio']:.6f}"
    )
    print(f"Wrote rows: {config.output_csv}")
    print(f"Wrote summary: {config.output_summary}")
    return 0
