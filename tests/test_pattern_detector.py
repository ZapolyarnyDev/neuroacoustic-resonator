from hypothesis import given
from hypothesis import strategies as st

from neuroacoustic_resonator.analysis.pattern_detector import (
    TemporalPatternDetector,
    pattern_novelty,
)
from neuroacoustic_resonator.protocol import PatternSnapshot


def snapshot(
    label: str,
    *,
    trace: float = 0.2,
    frequency_mean: float = 1.0,
) -> PatternSnapshot:
    orders = {
        "idle": (0.1, 0.1, 0.1),
        "coherent": (0.95, 0.1, 0.1),
        "split": (0.1, 0.95, 0.1),
        "triadic": (0.1, 0.1, 0.95),
    }
    order_1, order_2, order_3 = orders[label]
    return PatternSnapshot(
        phase_order_1=order_1,
        phase_order_2=order_2,
        phase_order_3=order_3,
        trace_mean=0.0 if label == "idle" else trace,
        trace_contrast=0.02,
        metabolite_stress=0.0 if label == "idle" else 0.1,
        metabolite_contrast=0.03,
        trace_phase_lock=0.2,
        metabolite_phase_lock=0.2,
        frequency_mean=frequency_mean,
        frequency_spread=0.1,
    )


def detector(**overrides: float | int) -> TemporalPatternDetector:
    values: dict[str, float | int] = {
        "confidence_threshold": 0.0,
        "confirmation_frames": 2,
        "minimum_active_frames": 1,
        "hysteresis_margin": 0.0,
        "novelty_threshold": 0.1,
    }
    values.update(overrides)
    return TemporalPatternDetector(
        confidence_threshold=float(values["confidence_threshold"]),
        confirmation_frames=int(values["confirmation_frames"]),
        minimum_active_frames=int(values["minimum_active_frames"]),
        hysteresis_margin=float(values["hysteresis_margin"]),
        novelty_threshold=float(values["novelty_threshold"]),
    )


def test_detector_confirms_start_and_keeps_state_separate_from_transition() -> None:
    temporal = detector()

    first_active, first_transition = temporal.update(snapshot("coherent"))
    active, transition = temporal.update(snapshot("coherent"))
    continued, continued_transition = temporal.update(snapshot("coherent"))

    assert first_active is None
    assert first_transition is None
    assert active is not None
    assert active.label == "coherent"
    assert active.age_frames == 1
    assert transition is not None
    assert transition.kind == "started"
    assert continued is not None
    assert continued.age_frames == 2
    assert continued_transition is None


def test_detector_confirms_pattern_change() -> None:
    temporal = detector()
    temporal.update(snapshot("coherent"))
    temporal.update(snapshot("coherent"))

    pending, pending_transition = temporal.update(snapshot("split"))
    changed, transition = temporal.update(snapshot("split"))

    assert pending is not None
    assert pending.label == "coherent"
    assert pending_transition is None
    assert changed is not None
    assert changed.label == "split"
    assert transition is not None
    assert (transition.kind, transition.from_label, transition.to_label) == (
        "changed",
        "coherent",
        "split",
    )


def test_detector_confirms_pattern_end() -> None:
    temporal = detector()
    temporal.update(snapshot("coherent"))
    temporal.update(snapshot("coherent"))

    pending, pending_transition = temporal.update(snapshot("idle"))
    ended, transition = temporal.update(snapshot("idle"))

    assert pending is not None
    assert pending_transition is None
    assert ended is None
    assert transition is not None
    assert (transition.kind, transition.from_label, transition.to_label) == (
        "ended",
        "coherent",
        None,
    )


def test_detector_hysteresis_suppresses_weak_switch() -> None:
    temporal = detector(hysteresis_margin=1.0, confirmation_frames=1)
    temporal.update(snapshot("coherent"))

    active, transition = temporal.update(snapshot("split"))

    assert active is not None
    assert active.label == "coherent"
    assert transition is None


def test_detector_reports_novelty_and_reset_clears_history() -> None:
    temporal = detector(confirmation_frames=1, novelty_threshold=0.01)
    first, _ = temporal.update(snapshot("coherent"))
    second, _ = temporal.update(snapshot("coherent", frequency_mean=3.0))
    temporal.reset()
    restarted, transition = temporal.update(snapshot("coherent"))

    assert first is not None
    assert first.novelty == 0.0
    assert second is not None
    assert second.novelty > 0.0
    assert second.is_novel
    assert restarted is not None
    assert restarted.novelty == 0.0
    assert transition is not None
    assert transition.kind == "started"


@given(
    st.lists(
        st.sampled_from(["idle", "coherent", "split", "triadic"]),
        min_size=1,
        max_size=40,
    )
)
def test_detector_transition_stream_stays_consistent(labels: list[str]) -> None:
    temporal = detector(confirmation_frames=1)
    previous_active: str | None = None

    for label in labels:
        active, transition = temporal.update(snapshot(label))
        current_active = active.label if active is not None else None
        if transition is None:
            assert current_active == previous_active
        elif transition.kind == "started":
            assert previous_active is None
            assert transition.to_label == current_active
        elif transition.kind == "changed":
            assert transition.from_label == previous_active
            assert transition.to_label == current_active
        else:
            assert transition.kind == "ended"
            assert transition.from_label == previous_active
            assert current_active is None
        previous_active = current_active


@given(
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
def test_pattern_novelty_is_symmetric_and_bounded(
    left_frequency: float,
    right_frequency: float,
) -> None:
    left = snapshot("coherent", frequency_mean=left_frequency)
    right = snapshot("coherent", frequency_mean=right_frequency)

    forward = pattern_novelty(left, right)
    backward = pattern_novelty(right, left)

    assert 0.0 <= forward <= 1.0
    assert forward == backward
