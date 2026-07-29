from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuroacoustic_resonator.protocol import (
    ActivePattern,
    FieldSnapshot,
    PatternSnapshot,
    PatternTransition,
    ProtocolDecodeError,
    ProtocolJsonlReader,
    ProtocolJsonlWriter,
    ProtocolReplay,
    RegionSnapshot,
    SoundProtocolFrame,
    decode_frame,
    encode_frame,
    frame_from_dict,
    frame_to_dict,
    read_protocol_jsonl,
    write_protocol_jsonl,
)


def _region(name: str) -> RegionSnapshot:
    return RegionSnapshot(
        name=name,  # type: ignore[arg-type]
        phase_coherence=0.7,
        mean_local_synchrony=0.6,
        mean_metabolite=0.8,
        min_metabolite=0.5,
        mean_trace=0.2,
        max_trace=0.4,
        mean_frequency=1.5,
        frequency_spread=0.3,
        mean_coupling=0.25,
        coupling_spread=0.05,
    )


def _frame() -> SoundProtocolFrame:
    return SoundProtocolFrame(
        sequence=3,
        step=12,
        time_seconds=0.6,
        field=FieldSnapshot(
            global_synchrony=0.4,
            mean_local_synchrony=0.5,
            mean_metabolite=0.75,
            min_metabolite=0.45,
            mean_trace=0.15,
            max_trace=0.35,
        ),
        input_region=_region("input"),
        assoc_region=_region("assoc"),
        output_region=_region("output"),
        pattern=PatternSnapshot(
            phase_order_1=0.7,
            phase_order_2=0.3,
            phase_order_3=0.2,
            trace_mean=0.2,
            trace_contrast=0.1,
            metabolite_stress=0.2,
            metabolite_contrast=0.15,
            trace_phase_lock=0.4,
            metabolite_phase_lock=0.35,
            frequency_mean=1.5,
            frequency_spread=0.3,
        ),
        active_pattern=ActivePattern(
            label="coherent",
            confidence=0.8,
            intensity=0.6,
            novelty=0.4,
            is_novel=True,
            age_frames=2,
        ),
        transition=PatternTransition(
            kind="started",
            from_label=None,
            to_label="coherent",
        ),
    )


def test_json_round_trip_preserves_complete_frame() -> None:
    frame = _frame()

    assert decode_frame(encode_frame(frame)) == frame
    assert frame_from_dict(frame_to_dict(frame)) == frame


def test_jsonl_writer_and_reader_preserve_frame_order() -> None:
    frames = [_frame(), replace(_frame(), sequence=4, step=16, time_seconds=0.8)]
    stream = io.StringIO()
    writer = ProtocolJsonlWriter(stream)

    for frame in frames:
        writer.write(frame)

    stream.seek(0)
    assert list(ProtocolJsonlReader(stream)) == frames


def test_protocol_replay_can_be_iterated_without_a_field(tmp_path) -> None:
    frames = [_frame(), replace(_frame(), sequence=4, step=16, time_seconds=0.8)]
    path = write_protocol_jsonl(tmp_path / "session.jsonl", iter(frames))
    replay = ProtocolReplay(path)

    assert list(replay) == frames
    assert list(replay) == frames
    assert read_protocol_jsonl(path) == frames


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.pop("step"), "missing"),
        (lambda data: data.update({"surprise": True}), "unknown"),
        (lambda data: data.update({"version": "9.9"}), "version must"),
        (lambda data: data.update({"sequence": True}), "sequence must"),
    ],
)
def test_decoder_rejects_invalid_schema(mutation, message: str) -> None:
    data = frame_to_dict(_frame())
    mutation(data)

    with pytest.raises(ProtocolDecodeError, match=message):
        frame_from_dict(data)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_decoder_rejects_non_finite_json_numbers(constant: str) -> None:
    encoded = encode_frame(_frame()).replace(
        '"global_synchrony":0.4',
        f'"global_synchrony":{constant}',
    )

    with pytest.raises(ProtocolDecodeError, match="non-finite"):
        decode_frame(encoded)


def test_jsonl_reader_rejects_blank_lines_with_line_number() -> None:
    stream = io.StringIO(f"{encode_frame(_frame())}\n\n")

    with pytest.raises(ProtocolDecodeError, match="line 2"):
        list(ProtocolJsonlReader(stream))


def test_decoder_rejects_duplicate_json_fields() -> None:
    encoded = encode_frame(_frame()).replace(
        '"sequence":3',
        '"sequence":3,"sequence":4',
    )

    with pytest.raises(ProtocolDecodeError, match="duplicate"):
        decode_frame(encoded)


def test_encoded_frame_is_strict_standard_json() -> None:
    encoded = encode_frame(_frame())

    assert json.loads(encoded)["version"] == "0.1"
    assert "\n" not in encoded


@given(
    sequence=st.integers(min_value=0, max_value=2**31),
    step=st.integers(min_value=0, max_value=2**31),
    time_seconds=st.floats(
        min_value=0.0,
        max_value=1e9,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_identity_values_survive_json_round_trip(
    sequence: int,
    step: int,
    time_seconds: float,
) -> None:
    frame = replace(
        _frame(),
        sequence=sequence,
        step=step,
        time_seconds=time_seconds,
    )

    assert decode_frame(encode_frame(frame)) == frame
