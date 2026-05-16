from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.core.field import FieldMetrics
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import SimulationFrame

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


@dataclass(frozen=True)
class RegionalActivityMetrics:
    step: int
    input_value: float
    input_activity: float
    input_fast_activity: float
    input_slow_activity: float
    assoc_activity: float
    assoc_fast_activity: float
    assoc_slow_activity: float
    output_activity: float
    output_activity_baseline: float
    output_response_activity: float
    output_fast_activity: float
    output_slow_activity: float
    left_to_right_ratio: float
    output_trace: float
    output_synchrony: float
    output_fast_delta: float
    output_slow_delta: float
    output_trace_delta: float
    output_synchrony_delta: float
    output_activity_delta: float
    output_event_score: float
    output_fast_response_score: float
    output_slow_drift_score: float


class RegionalActivityTracker:
    def __init__(self, *, baseline_smoothing: float = 0.02) -> None:
        if not 0.0 < baseline_smoothing <= 1.0:
            msg = "baseline_smoothing must be in (0, 1]"
            raise ValueError(msg)
        self._previous: RegionalActivityMetrics | None = None
        self._output_activity_baseline: float | None = None
        self.baseline_smoothing = baseline_smoothing

    def update(
        self,
        frame: SimulationFrame,
        regions: RegionMasks,
        *,
        input_value: float = 0.0,
    ) -> RegionalActivityMetrics:
        metrics = compute_regional_activity_metrics(
            frame,
            regions,
            input_value=input_value,
            previous=self._previous,
            output_activity_baseline=self._output_activity_baseline,
        )
        self._previous = metrics
        self._output_activity_baseline = (
            metrics.output_activity
            if self._output_activity_baseline is None
            else self._output_activity_baseline
            + self.baseline_smoothing
            * (metrics.output_activity - self._output_activity_baseline)
        )
        return metrics


def compute_regional_activity_metrics(
    frame: SimulationFrame,
    regions: RegionMasks,
    *,
    input_value: float = 0.0,
    previous: RegionalActivityMetrics | None = None,
    output_activity_baseline: float | None = None,
) -> RegionalActivityMetrics:
    if frame.state.phase.shape != regions.shape:
        msg = "frame and regions must have matching shapes"
        raise ValueError(msg)

    input_fast_activity = region_fast_activity(frame, regions.input)
    input_slow_activity = region_slow_activity(frame, regions.input)
    assoc_fast_activity = region_fast_activity(frame, regions.assoc)
    assoc_slow_activity = region_slow_activity(frame, regions.assoc)
    output_fast_activity = region_fast_activity(frame, regions.output)
    output_slow_activity = region_slow_activity(frame, regions.output)
    input_activity = combine_region_activity(input_fast_activity, input_slow_activity)
    assoc_activity = combine_region_activity(assoc_fast_activity, assoc_slow_activity)
    output_activity = combine_region_activity(
        output_fast_activity, output_slow_activity
    )
    output_activity_baseline = (
        output_activity
        if output_activity_baseline is None
        else output_activity_baseline
    )
    output_response_activity = output_activity - output_activity_baseline
    output_trace = region_mean(frame.state.trace, regions.output)
    output_synchrony = region_mean(frame.local_synchrony, regions.output)

    if previous is None:
        output_fast_delta = 0.0
        output_slow_delta = 0.0
        output_trace_delta = 0.0
        output_synchrony_delta = 0.0
        output_activity_delta = 0.0
    else:
        output_fast_delta = output_fast_activity - previous.output_fast_activity
        output_slow_delta = output_slow_activity - previous.output_slow_activity
        output_trace_delta = output_trace - previous.output_trace
        output_synchrony_delta = output_synchrony - previous.output_synchrony
        output_activity_delta = output_activity - previous.output_activity

    output_event_score = (
        abs(output_activity_delta)
        + abs(output_trace_delta)
        + abs(output_synchrony_delta)
    )
    return RegionalActivityMetrics(
        step=frame.metrics.step,
        input_value=float(input_value),
        input_activity=input_activity,
        input_fast_activity=input_fast_activity,
        input_slow_activity=input_slow_activity,
        assoc_activity=assoc_activity,
        assoc_fast_activity=assoc_fast_activity,
        assoc_slow_activity=assoc_slow_activity,
        output_activity=output_activity,
        output_activity_baseline=output_activity_baseline,
        output_response_activity=output_response_activity,
        output_fast_activity=output_fast_activity,
        output_slow_activity=output_slow_activity,
        left_to_right_ratio=output_activity / max(input_activity, 1e-12),
        output_trace=output_trace,
        output_synchrony=output_synchrony,
        output_fast_delta=output_fast_delta,
        output_slow_delta=output_slow_delta,
        output_trace_delta=output_trace_delta,
        output_synchrony_delta=output_synchrony_delta,
        output_activity_delta=output_activity_delta,
        output_event_score=output_event_score,
        output_fast_response_score=abs(output_fast_delta),
        output_slow_drift_score=abs(output_slow_delta),
    )


def region_activity(frame: SimulationFrame, mask: np.ndarray) -> float:
    return combine_region_activity(
        region_fast_activity(frame, mask),
        region_slow_activity(frame, mask),
    )


def combine_region_activity(fast_activity: float, slow_activity: float) -> float:
    return slow_activity + 0.25 * fast_activity


def region_fast_activity(frame: SimulationFrame, mask: np.ndarray) -> float:
    return region_mean(frame.local_synchrony, mask)


def region_slow_activity(frame: SimulationFrame, mask: np.ndarray) -> float:
    trace = region_mean(frame.state.trace, mask)
    metabolite_depletion = region_mean(1.0 - frame.state.metabolite, mask)
    return trace + metabolite_depletion


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.shape != mask.shape:
        msg = "values and mask must have matching shapes"
        raise ValueError(msg)
    return float(np.mean(values[mask]))
