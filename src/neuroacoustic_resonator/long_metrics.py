from __future__ import annotations

import argparse
from pathlib import Path

from neuroacoustic_resonator.config import SimulationConfig
from neuroacoustic_resonator.metrics import MetricsHistory
from neuroacoustic_resonator.simulation import Simulation


def default_metrics_output_path(
    config_path: Path,
    steps: int,
    output_format: str,
) -> Path:
    return (
        Path("outputs")
        / "metrics"
        / f"{config_path.stem}-{steps}-steps.{output_format}"
    )


def collect_metrics(
    config_path: str | Path,
    *,
    steps: int | None = None,
    sample_interval: int = 1,
) -> MetricsHistory:
    if steps is not None and steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if sample_interval < 1:
        msg = "sample_interval must be positive"
        raise ValueError(msg)

    config = SimulationConfig.from_file(config_path)
    total_steps = steps or config.steps
    simulation = Simulation.from_config(config)
    history = MetricsHistory([simulation.snapshot().metrics])

    for _ in range(total_steps):
        frame = simulation.step()
        is_sample_step = frame.metrics.step % sample_interval == 0
        is_final_step = frame.metrics.step == total_steps
        if is_sample_step or is_final_step:
            history.append(frame.metrics)

    return history


def write_metrics_history(history: MetricsHistory, output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix == ".csv":
        return history.write_csv(output)
    if output.suffix == ".jsonl":
        return history.write_jsonl(output)

    msg = "output path must end with .csv or .jsonl"
    raise ValueError(msg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a field simulation and record metrics over time.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "default.yaml",
        help="YAML simulation config path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps. Defaults to the config value.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1,
        help="Record metrics every N steps. The final step is always recorded.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Output format used when --output is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metrics output path. Suffix must be .csv or .jsonl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig.from_file(args.config)
    total_steps = args.steps or config.steps
    output_path = args.output or default_metrics_output_path(
        args.config,
        total_steps,
        args.format,
    )

    history = collect_metrics(
        args.config,
        steps=total_steps,
        sample_interval=args.sample_interval,
    )
    written_path = write_metrics_history(history, output_path)
    latest = history.latest()

    print(
        f"Recorded {len(history)} metric rows for {total_steps} steps: {written_path}"
    )
    print(
        "Latest metrics: "
        f"step={latest.step}, "
        f"global_synchrony={latest.global_synchrony:.6f}, "
        f"mean_metabolite={latest.mean_metabolite:.6f}, "
        f"mean_trace={latest.mean_trace:.6f}"
    )
    return 0
