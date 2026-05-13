import csv

import pytest

from neuroacoustic_resonator.analysis.benchmark import (
    benchmark_field_step,
    benchmark_sizes,
    main,
    parse_sizes,
    write_benchmark_results,
)


def test_benchmark_field_step_returns_positive_rates() -> None:
    result = benchmark_field_step(size=4, steps=2, warmup_steps=1)

    assert result.size == 4
    assert result.steps == 2
    assert result.cells == 16
    assert result.elapsed_seconds > 0.0
    assert result.steps_per_second > 0.0
    assert result.milliseconds_per_step > 0.0
    assert result.cell_steps_per_second > 0.0


def test_benchmark_sizes_repeats_each_size() -> None:
    results = benchmark_sizes([3, 4], steps=1, warmup_steps=0, repeats=2)

    assert [(result.size, result.repeat) for result in results] == [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
    ]


def test_parse_sizes_accepts_spaces_and_commas() -> None:
    assert parse_sizes(["4,8", "16"]) == [4, 8, 16]


def test_write_results_rejects_unknown_suffix(tmp_path) -> None:
    result = benchmark_field_step(size=4, steps=1, warmup_steps=0)

    with pytest.raises(ValueError, match="csv or .jsonl"):
        write_benchmark_results([result], tmp_path / "benchmark.txt")


def test_main_writes_benchmark_csv(tmp_path) -> None:
    output_path = tmp_path / "benchmark.csv"

    exit_code = main(
        [
            "--sizes",
            "4",
            "--steps",
            "2",
            "--warmup-steps",
            "1",
            "--repeats",
            "1",
            "--output",
            str(output_path),
        ]
    )

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert exit_code == 0
    assert [int(row["size"]) for row in rows] == [4]
    assert float(rows[0]["steps_per_second"]) > 0.0
