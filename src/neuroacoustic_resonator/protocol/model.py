from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal

from neuroacoustic_resonator.protocol._validation import (
    require_non_empty,
    require_non_negative,
    require_non_negative_int,
    require_not_greater,
    require_optional_non_empty,
    require_positive_int,
    require_unit_interval,
)

PROTOCOL_VERSION = "0.1"

RegionName = Literal["input", "assoc", "output"]
PatternTransitionKind = Literal["started", "changed", "ended"]


@dataclass(frozen=True, slots=True)
class FieldSnapshot:
    global_synchrony: float
    mean_local_synchrony: float
    mean_metabolite: float
    min_metabolite: float
    mean_trace: float
    max_trace: float

    def __post_init__(self) -> None:
        require_unit_interval("global_synchrony", self.global_synchrony)
        require_unit_interval("mean_local_synchrony", self.mean_local_synchrony)
        require_unit_interval("mean_metabolite", self.mean_metabolite)
        require_unit_interval("min_metabolite", self.min_metabolite)
        require_non_negative("mean_trace", self.mean_trace)
        require_non_negative("max_trace", self.max_trace)
        require_not_greater(
            "min_metabolite",
            self.min_metabolite,
            "mean_metabolite",
            self.mean_metabolite,
        )
        require_not_greater(
            "mean_trace",
            self.mean_trace,
            "max_trace",
            self.max_trace,
        )


@dataclass(frozen=True, slots=True)
class RegionSnapshot:
    name: RegionName
    phase_coherence: float
    mean_local_synchrony: float
    mean_metabolite: float
    min_metabolite: float
    mean_trace: float
    max_trace: float
    mean_frequency: float
    frequency_spread: float
    mean_coupling: float
    coupling_spread: float

    def __post_init__(self) -> None:
        if self.name not in {"input", "assoc", "output"}:
            msg = f"unsupported region name: {self.name!r}"
            raise ValueError(msg)
        require_unit_interval("phase_coherence", self.phase_coherence)
        require_unit_interval("mean_local_synchrony", self.mean_local_synchrony)
        require_unit_interval("mean_metabolite", self.mean_metabolite)
        require_unit_interval("min_metabolite", self.min_metabolite)
        require_non_negative("mean_trace", self.mean_trace)
        require_non_negative("max_trace", self.max_trace)
        require_non_negative("mean_frequency", self.mean_frequency)
        require_non_negative("frequency_spread", self.frequency_spread)
        require_non_negative("mean_coupling", self.mean_coupling)
        require_non_negative("coupling_spread", self.coupling_spread)
        require_not_greater(
            "min_metabolite",
            self.min_metabolite,
            "mean_metabolite",
            self.mean_metabolite,
        )
        require_not_greater(
            "mean_trace",
            self.mean_trace,
            "max_trace",
            self.max_trace,
        )


@dataclass(frozen=True, slots=True)
class PatternSnapshot:
    phase_order_1: float
    phase_order_2: float
    phase_order_3: float
    trace_mean: float
    trace_contrast: float
    metabolite_stress: float
    metabolite_contrast: float
    trace_phase_lock: float
    metabolite_phase_lock: float
    frequency_mean: float
    frequency_spread: float

    def __post_init__(self) -> None:
        require_unit_interval("phase_order_1", self.phase_order_1)
        require_unit_interval("phase_order_2", self.phase_order_2)
        require_unit_interval("phase_order_3", self.phase_order_3)
        require_non_negative("trace_mean", self.trace_mean)
        require_non_negative("trace_contrast", self.trace_contrast)
        require_unit_interval("metabolite_stress", self.metabolite_stress)
        require_unit_interval("metabolite_contrast", self.metabolite_contrast)
        require_unit_interval("trace_phase_lock", self.trace_phase_lock)
        require_unit_interval("metabolite_phase_lock", self.metabolite_phase_lock)
        require_non_negative("frequency_mean", self.frequency_mean)
        require_non_negative("frequency_spread", self.frequency_spread)


@dataclass(frozen=True, slots=True)
class ActivePattern:
    label: str
    confidence: float
    intensity: float
    novelty: float
    is_novel: bool
    age_frames: int

    def __post_init__(self) -> None:
        require_non_empty("label", self.label)
        require_unit_interval("confidence", self.confidence)
        require_unit_interval("intensity", self.intensity)
        require_unit_interval("novelty", self.novelty)
        require_positive_int("age_frames", self.age_frames)


@dataclass(frozen=True, slots=True)
class PatternTransition:
    kind: PatternTransitionKind
    from_label: str | None
    to_label: str | None

    def __post_init__(self) -> None:
        if self.kind not in {"started", "changed", "ended"}:
            msg = f"unsupported pattern transition: {self.kind!r}"
            raise ValueError(msg)
        require_optional_non_empty("from_label", self.from_label)
        require_optional_non_empty("to_label", self.to_label)
        if self.kind == "started" and (
            self.from_label is not None or self.to_label is None
        ):
            msg = "started transition must move from no label to a label"
            raise ValueError(msg)
        if self.kind == "changed" and (
            self.from_label is None
            or self.to_label is None
            or self.from_label == self.to_label
        ):
            msg = "changed transition must move between distinct labels"
            raise ValueError(msg)
        if self.kind == "ended" and (
            self.from_label is None or self.to_label is not None
        ):
            msg = "ended transition must move from a label to no label"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SoundProtocolFrame:
    sequence: int
    step: int
    time_seconds: float
    field: FieldSnapshot
    input_region: RegionSnapshot
    assoc_region: RegionSnapshot
    output_region: RegionSnapshot
    pattern: PatternSnapshot
    active_pattern: ActivePattern | None
    transition: PatternTransition | None
    version: str = dataclass_field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        require_non_negative_int("sequence", self.sequence)
        require_non_negative_int("step", self.step)
        require_non_negative("time_seconds", self.time_seconds)
        if self.input_region.name != "input":
            msg = "input_region must be named 'input'"
            raise ValueError(msg)
        if self.assoc_region.name != "assoc":
            msg = "assoc_region must be named 'assoc'"
            raise ValueError(msg)
        if self.output_region.name != "output":
            msg = "output_region must be named 'output'"
            raise ValueError(msg)
