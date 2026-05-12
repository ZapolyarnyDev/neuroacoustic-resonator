from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from neuroacoustic_resonator.field import FieldConfig, OscillatorField

OutputFormat = Literal["csv", "jsonl"]


@dataclass(frozen=True)
class BenchmarkResult:
    size: int
    steps: int
    warmup_steps: int
    repeat: int
    elapsed_seconds: float
    steps_per_second: float
    milliseconds_per_step: float
    cells: int
    cell_steps_per_second: float


def default_benchmark_output_path(output_format: OutputFormat) -> Path:
    return Path("outputs") / "benchmarks" / f"field-step.{output_format}"


def benchmark_field_step(
    *,
    size: int,
    steps: int,
    warmup_steps: int = 10,
    repeat: int = 0,
    seed: int | None = 1,
) -> BenchmarkResult:
    if size <= 1:
        msg = "size must be greater than 1"
        raise ValueError(msg)
    if steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if warmup_steps < 0:
        msg = "warmup_steps must be non-negative"
        raise ValueError(msg)

    field = OscillatorField(FieldConfig(size=size, seed=seed))
    for _ in range(warmup_steps):
        field.step()

    started_at = perf_counter()
    for _ in range(steps):
        field.step()
    elapsed_seconds = perf_counter() - started_at

    steps_per_second = steps / elapsed_seconds
    cells = size * size
    return BenchmarkResult(
        size=size,
        steps=steps,
        warmup_steps=warmup_steps,
        repeat=repeat,
        elapsed_seconds=elapsed_seconds,
        steps_per_second=steps_per_second,
        milliseconds_per_step=1000.0 / steps_per_second,
        cells=cells,
        cell_steps_per_second=cells * steps_per_second,
    )


def benchmark_sizes(
    sizes: list[int],
    *,
    steps: int,
    warmup_steps: int = 10,
    repeats: int = 3,
    seed: int | None = 1,
) -> list[BenchmarkResult]:
    if not sizes:
        msg = "at least one size is required"
        raise ValueError(msg)
    if repeats < 1:
        msg = "repeats must be positive"
        raise ValueError(msg)

    return [
        benchmark_field_step(
            size=size,
            steps=steps,
            warmup_steps=warmup_steps,
            repeat=repeat_index,
            seed=seed,
        )
        for size in sizes
        for repeat_index in range(1, repeats + 1)
    ]


def write_benchmark_results(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]

    if output.suffix == ".csv":
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return output

    if output.suffix == ".jsonl":
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True))
                stream.write("\n")
        return output

    msg = "output path must end with .csv or .jsonl"
    raise ValueError(msg)


def parse_sizes(values: list[str]) -> list[int]:
    sizes: list[int] = []
    for value in values:
        sizes.extend(int(part) for part in value.split(",") if part)
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OscillatorField.step() for one or more field sizes.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["64", "128", "200"],
        help="Field sizes, either space-separated or comma-separated.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Measured steps per repeat.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Unmeasured warmup steps before each repeat.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Benchmark repeats per size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used to initialize each field.",
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
        help="Benchmark output path. Suffix must be .csv or .jsonl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_format = cast(OutputFormat, args.format)
    output_path = args.output or default_benchmark_output_path(output_format)
    results = benchmark_sizes(
        parse_sizes(args.sizes),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        seed=args.seed,
    )
    written_path = write_benchmark_results(results, output_path)

    print(f"Recorded {len(results)} benchmark rows: {written_path}")
    for result in results:
        print(
            f"size={result.size} repeat={result.repeat} "
            f"{result.steps_per_second:.2f} steps/s "
            f"({result.milliseconds_per_step:.3f} ms/step)"
        )
    return 0
