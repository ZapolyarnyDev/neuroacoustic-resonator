from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TextIO, cast

from neuroacoustic_resonator.protocol.model import (
    PROTOCOL_VERSION,
    ActivePattern,
    FieldSnapshot,
    PatternSnapshot,
    PatternTransition,
    PatternTransitionKind,
    RegionName,
    RegionSnapshot,
    SoundProtocolFrame,
)


class ProtocolDecodeError(ValueError):
    pass


def frame_to_dict(frame: SoundProtocolFrame) -> dict[str, object]:
    return {
        "active_pattern": (
            None
            if frame.active_pattern is None
            else {
                "age_frames": frame.active_pattern.age_frames,
                "confidence": frame.active_pattern.confidence,
                "intensity": frame.active_pattern.intensity,
                "is_novel": frame.active_pattern.is_novel,
                "label": frame.active_pattern.label,
                "novelty": frame.active_pattern.novelty,
            }
        ),
        "assoc_region": _region_to_dict(frame.assoc_region),
        "field": {
            "global_synchrony": frame.field.global_synchrony,
            "max_trace": frame.field.max_trace,
            "mean_local_synchrony": frame.field.mean_local_synchrony,
            "mean_metabolite": frame.field.mean_metabolite,
            "mean_trace": frame.field.mean_trace,
            "min_metabolite": frame.field.min_metabolite,
        },
        "input_region": _region_to_dict(frame.input_region),
        "output_region": _region_to_dict(frame.output_region),
        "pattern": {
            "frequency_mean": frame.pattern.frequency_mean,
            "frequency_spread": frame.pattern.frequency_spread,
            "metabolite_contrast": frame.pattern.metabolite_contrast,
            "metabolite_phase_lock": frame.pattern.metabolite_phase_lock,
            "metabolite_stress": frame.pattern.metabolite_stress,
            "phase_order_1": frame.pattern.phase_order_1,
            "phase_order_2": frame.pattern.phase_order_2,
            "phase_order_3": frame.pattern.phase_order_3,
            "trace_contrast": frame.pattern.trace_contrast,
            "trace_mean": frame.pattern.trace_mean,
            "trace_phase_lock": frame.pattern.trace_phase_lock,
        },
        "sequence": frame.sequence,
        "step": frame.step,
        "time_seconds": frame.time_seconds,
        "transition": (
            None
            if frame.transition is None
            else {
                "from_label": frame.transition.from_label,
                "kind": frame.transition.kind,
                "to_label": frame.transition.to_label,
            }
        ),
        "version": frame.version,
    }


def frame_from_dict(value: object) -> SoundProtocolFrame:
    root = _require_object(
        value,
        "$",
        {
            "active_pattern",
            "assoc_region",
            "field",
            "input_region",
            "output_region",
            "pattern",
            "sequence",
            "step",
            "time_seconds",
            "transition",
            "version",
        },
    )
    version = _require_string(root["version"], "$.version")
    if version != PROTOCOL_VERSION:
        msg = f"$.version must be {PROTOCOL_VERSION!r}, got {version!r}"
        raise ProtocolDecodeError(msg)
    try:
        return SoundProtocolFrame(
            sequence=_require_int(root["sequence"], "$.sequence"),
            step=_require_int(root["step"], "$.step"),
            time_seconds=_require_float(root["time_seconds"], "$.time_seconds"),
            field=_decode_field(root["field"]),
            input_region=_decode_region(root["input_region"], "$.input_region"),
            assoc_region=_decode_region(root["assoc_region"], "$.assoc_region"),
            output_region=_decode_region(root["output_region"], "$.output_region"),
            pattern=_decode_pattern(root["pattern"]),
            active_pattern=_decode_active_pattern(root["active_pattern"]),
            transition=_decode_transition(root["transition"]),
        )
    except ValueError as exc:
        raise ProtocolDecodeError(str(exc)) from exc


def encode_frame(frame: SoundProtocolFrame) -> str:
    return json.dumps(
        frame_to_dict(frame),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_frame(value: str) -> SoundProtocolFrame:
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_require_unique_json_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        msg = f"invalid protocol JSON: {exc.msg}"
        raise ProtocolDecodeError(msg) from exc
    return frame_from_dict(decoded)


class ProtocolJsonlWriter:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, frame: SoundProtocolFrame) -> None:
        self._stream.write(encode_frame(frame))
        self._stream.write("\n")


class ProtocolJsonlReader:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[SoundProtocolFrame]:
        for line_number, line in enumerate(self._stream, start=1):
            if not line.strip():
                msg = f"invalid protocol JSONL at line {line_number}: blank line"
                raise ProtocolDecodeError(msg)
            try:
                yield decode_frame(line)
            except ProtocolDecodeError as exc:
                msg = f"invalid protocol JSONL at line {line_number}: {exc}"
                raise ProtocolDecodeError(msg) from exc


class ProtocolReplay:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[SoundProtocolFrame]:
        with self.path.open(encoding="utf-8") as stream:
            yield from ProtocolJsonlReader(stream)


def write_protocol_jsonl(
    path: str | Path,
    frames: Iterable[SoundProtocolFrame],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        writer = ProtocolJsonlWriter(stream)
        for frame in frames:
            writer.write(frame)
    return destination


def read_protocol_jsonl(path: str | Path) -> list[SoundProtocolFrame]:
    return list(ProtocolReplay(path))


def _region_to_dict(region: RegionSnapshot) -> dict[str, object]:
    return {
        "coupling_spread": region.coupling_spread,
        "frequency_spread": region.frequency_spread,
        "max_trace": region.max_trace,
        "mean_coupling": region.mean_coupling,
        "mean_frequency": region.mean_frequency,
        "mean_local_synchrony": region.mean_local_synchrony,
        "mean_metabolite": region.mean_metabolite,
        "mean_trace": region.mean_trace,
        "min_metabolite": region.min_metabolite,
        "name": region.name,
        "phase_coherence": region.phase_coherence,
    }


def _decode_field(value: object) -> FieldSnapshot:
    path = "$.field"
    data = _require_object(
        value,
        path,
        {
            "global_synchrony",
            "max_trace",
            "mean_local_synchrony",
            "mean_metabolite",
            "mean_trace",
            "min_metabolite",
        },
    )
    return FieldSnapshot(
        global_synchrony=_require_float(
            data["global_synchrony"], f"{path}.global_synchrony"
        ),
        mean_local_synchrony=_require_float(
            data["mean_local_synchrony"], f"{path}.mean_local_synchrony"
        ),
        mean_metabolite=_require_float(
            data["mean_metabolite"], f"{path}.mean_metabolite"
        ),
        min_metabolite=_require_float(data["min_metabolite"], f"{path}.min_metabolite"),
        mean_trace=_require_float(data["mean_trace"], f"{path}.mean_trace"),
        max_trace=_require_float(data["max_trace"], f"{path}.max_trace"),
    )


def _decode_region(value: object, path: str) -> RegionSnapshot:
    data = _require_object(
        value,
        path,
        {
            "coupling_spread",
            "frequency_spread",
            "max_trace",
            "mean_coupling",
            "mean_frequency",
            "mean_local_synchrony",
            "mean_metabolite",
            "mean_trace",
            "min_metabolite",
            "name",
            "phase_coherence",
        },
    )
    return RegionSnapshot(
        name=cast(RegionName, _require_string(data["name"], f"{path}.name")),
        phase_coherence=_require_float(
            data["phase_coherence"], f"{path}.phase_coherence"
        ),
        mean_local_synchrony=_require_float(
            data["mean_local_synchrony"], f"{path}.mean_local_synchrony"
        ),
        mean_metabolite=_require_float(
            data["mean_metabolite"], f"{path}.mean_metabolite"
        ),
        min_metabolite=_require_float(data["min_metabolite"], f"{path}.min_metabolite"),
        mean_trace=_require_float(data["mean_trace"], f"{path}.mean_trace"),
        max_trace=_require_float(data["max_trace"], f"{path}.max_trace"),
        mean_frequency=_require_float(data["mean_frequency"], f"{path}.mean_frequency"),
        frequency_spread=_require_float(
            data["frequency_spread"], f"{path}.frequency_spread"
        ),
        mean_coupling=_require_float(data["mean_coupling"], f"{path}.mean_coupling"),
        coupling_spread=_require_float(
            data["coupling_spread"], f"{path}.coupling_spread"
        ),
    )


def _decode_pattern(value: object) -> PatternSnapshot:
    path = "$.pattern"
    fields = {
        "frequency_mean",
        "frequency_spread",
        "metabolite_contrast",
        "metabolite_phase_lock",
        "metabolite_stress",
        "phase_order_1",
        "phase_order_2",
        "phase_order_3",
        "trace_contrast",
        "trace_mean",
        "trace_phase_lock",
    }
    data = _require_object(value, path, fields)
    numbers = {name: _require_float(data[name], f"{path}.{name}") for name in fields}
    return PatternSnapshot(**numbers)


def _decode_active_pattern(value: object) -> ActivePattern | None:
    if value is None:
        return None
    path = "$.active_pattern"
    data = _require_object(
        value,
        path,
        {
            "age_frames",
            "confidence",
            "intensity",
            "is_novel",
            "label",
            "novelty",
        },
    )
    return ActivePattern(
        label=_require_string(data["label"], f"{path}.label"),
        confidence=_require_float(data["confidence"], f"{path}.confidence"),
        intensity=_require_float(data["intensity"], f"{path}.intensity"),
        novelty=_require_float(data["novelty"], f"{path}.novelty"),
        is_novel=_require_bool(data["is_novel"], f"{path}.is_novel"),
        age_frames=_require_int(data["age_frames"], f"{path}.age_frames"),
    )


def _decode_transition(value: object) -> PatternTransition | None:
    if value is None:
        return None
    path = "$.transition"
    data = _require_object(
        value,
        path,
        {"from_label", "kind", "to_label"},
    )
    return PatternTransition(
        kind=cast(
            PatternTransitionKind,
            _require_string(data["kind"], f"{path}.kind"),
        ),
        from_label=_require_optional_string(data["from_label"], f"{path}.from_label"),
        to_label=_require_optional_string(data["to_label"], f"{path}.to_label"),
    )


def _require_object(
    value: object,
    path: str,
    fields: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{path} must be an object"
        raise ProtocolDecodeError(msg)
    keys = set(value)
    if keys != fields:
        missing = sorted(fields - keys)
        unknown = sorted(keys - fields)
        details = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        msg = f"{path} has invalid fields: {', '.join(details)}"
        raise ProtocolDecodeError(msg)
    return cast(dict[str, object], value)


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{path} must be a number"
        raise ProtocolDecodeError(msg)
    result = float(value)
    if not math.isfinite(result):
        msg = f"{path} must be finite"
        raise ProtocolDecodeError(msg)
    return result


def _require_int(value: object, path: str) -> int:
    if type(value) is not int:
        msg = f"{path} must be an integer"
        raise ProtocolDecodeError(msg)
    return value


def _require_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        msg = f"{path} must be a boolean"
        raise ProtocolDecodeError(msg)
    return value


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        msg = f"{path} must be a string"
        raise ProtocolDecodeError(msg)
    return value


def _require_optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, path)


def _reject_json_constant(value: str) -> None:
    msg = f"non-finite JSON number is not allowed: {value}"
    raise ProtocolDecodeError(msg)


def _require_unique_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON field is not allowed: {key!r}"
            raise ProtocolDecodeError(msg)
        result[key] = value
    return result
