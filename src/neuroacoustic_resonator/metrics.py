from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from neuroacoustic_resonator.field import FieldMetrics

MetricsRow = dict[str, Any]

METRICS_FIELD_NAMES = tuple(field.name for field in fields(FieldMetrics))


class MetricsHistory:
    def __init__(self, metrics: Iterable[FieldMetrics] = ()) -> None:
        self._metrics = list(metrics)

    def __iter__(self) -> Iterator[FieldMetrics]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def append(self, metrics: FieldMetrics) -> None:
        self._metrics.append(metrics)

    def extend(self, metrics: Iterable[FieldMetrics]) -> None:
        self._metrics.extend(metrics)

    def latest(self) -> FieldMetrics:
        if not self._metrics:
            msg = "metrics history is empty"
            raise ValueError(msg)
        return self._metrics[-1]

    def to_rows(self) -> list[MetricsRow]:
        return [asdict(metrics) for metrics in self._metrics]

    def write_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRICS_FIELD_NAMES)
            writer.writeheader()
            writer.writerows(self.to_rows())

        return output_path

    def write_jsonl(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as stream:
            for row in self.to_rows():
                stream.write(json.dumps(row, sort_keys=True))
                stream.write("\n")

        return output_path
