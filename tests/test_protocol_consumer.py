from __future__ import annotations

from dataclasses import replace

from neuroacoustic_resonator.protocol import (
    CaptureConsumer,
    FieldSnapshot,
    PatternSnapshot,
    ProtocolConsumer,
    ProtocolPipeline,
    ProtocolReplay,
    RegionSnapshot,
    SoundProtocolFrame,
    write_protocol_jsonl,
)


def _frame() -> SoundProtocolFrame:
    return SoundProtocolFrame(
        sequence=0,
        step=0,
        time_seconds=0.0,
        field=FieldSnapshot(
            global_synchrony=0.0,
            mean_local_synchrony=0.0,
            mean_metabolite=0.0,
            min_metabolite=0.0,
            mean_trace=0.0,
            max_trace=0.0,
        ),
        input_region=_region("input"),
        assoc_region=_region("assoc"),
        output_region=_region("output"),
        pattern=PatternSnapshot(
            phase_order_1=0.0,
            phase_order_2=0.0,
            phase_order_3=0.0,
            trace_mean=0.0,
            trace_contrast=0.0,
            metabolite_stress=0.0,
            metabolite_contrast=0.0,
            trace_phase_lock=0.0,
            metabolite_phase_lock=0.0,
            frequency_mean=0.0,
            frequency_spread=0.0,
        ),
        active_pattern=None,
        transition=None,
    )


def _region(name: str) -> RegionSnapshot:
    return RegionSnapshot(
        name=name,  # type: ignore[arg-type]
        phase_coherence=0.0,
        mean_local_synchrony=0.0,
        mean_metabolite=0.0,
        min_metabolite=0.0,
        mean_trace=0.0,
        max_trace=0.0,
        mean_frequency=0.0,
        frequency_spread=0.0,
        mean_coupling=0.0,
        coupling_spread=0.0,
    )


class OrderedConsumer:
    def __init__(self, name: str, calls: list[tuple[str, int]]) -> None:
        self._name = name
        self._calls = calls

    def consume(self, frame: SoundProtocolFrame) -> None:
        self._calls.append((self._name, frame.sequence))


def test_consumer_interface_uses_structural_typing() -> None:
    capture = CaptureConsumer()

    assert isinstance(capture, ProtocolConsumer)


def test_pipeline_dispatches_each_frame_to_consumers_in_order() -> None:
    calls: list[tuple[str, int]] = []
    pipeline = ProtocolPipeline(
        [
            OrderedConsumer("first", calls),
            OrderedConsumer("second", calls),
        ]
    )
    frames = [_frame(), replace(_frame(), sequence=1, step=1, time_seconds=0.1)]

    count = pipeline.run(frames)

    assert count == 2
    assert calls == [
        ("first", 0),
        ("second", 0),
        ("first", 1),
        ("second", 1),
    ]


def test_capture_consumer_can_be_cleared() -> None:
    capture = CaptureConsumer()
    capture.consume(_frame())

    capture.clear()

    assert capture.frames == []


def test_live_and_replayed_frames_use_the_same_pipeline(tmp_path) -> None:
    frames = [_frame(), replace(_frame(), sequence=1, step=1, time_seconds=0.1)]
    recording = write_protocol_jsonl(
        tmp_path / "recording.jsonl",
        iter(frames),
    )
    live_capture = CaptureConsumer()
    replay_capture = CaptureConsumer()

    live_count = ProtocolPipeline([live_capture]).run(frames)
    replay_count = ProtocolPipeline([replay_capture]).run(ProtocolReplay(recording))

    assert live_count == replay_count == 2
    assert live_capture.frames == replay_capture.frames == frames
