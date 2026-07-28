from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest
from hypothesis import given
from hypothesis import strategies as st

from neuroacoustic_resonator.protocol import (
    PROTOCOL_VERSION,
    ActivePattern,
    FieldSnapshot,
    PatternSnapshot,
    PatternTransition,
    RegionSnapshot,
    SoundProtocolFrame,
)


def field_snapshot() -> FieldSnapshot:
    return FieldSnapshot(
        global_synchrony=0.4,
        mean_local_synchrony=0.5,
        mean_metabolite=0.8,
        min_metabolite=0.6,
        mean_trace=0.2,
        max_trace=0.3,
    )


def region_snapshot(name: str) -> RegionSnapshot:
    return RegionSnapshot(
        name=name,  # type: ignore[arg-type]
        phase_coherence=0.4,
        mean_local_synchrony=0.5,
        mean_metabolite=0.8,
        min_metabolite=0.6,
        mean_trace=0.2,
        max_trace=0.3,
        mean_frequency=1.0,
        frequency_spread=0.1,
        mean_coupling=0.2,
        coupling_spread=0.02,
    )


def pattern_snapshot() -> PatternSnapshot:
    return PatternSnapshot(
        phase_order_1=0.4,
        phase_order_2=0.3,
        phase_order_3=0.2,
        trace_mean=0.1,
        trace_contrast=0.02,
        metabolite_stress=0.2,
        metabolite_contrast=0.03,
        trace_phase_lock=0.4,
        metabolite_phase_lock=0.5,
        frequency_mean=1.0,
        frequency_spread=0.1,
    )


def test_protocol_frame_has_fixed_version_and_is_immutable() -> None:
    frame = SoundProtocolFrame(
        sequence=0,
        step=4,
        time_seconds=0.08,
        field=field_snapshot(),
        input_region=region_snapshot("input"),
        assoc_region=region_snapshot("assoc"),
        output_region=region_snapshot("output"),
        pattern=pattern_snapshot(),
        active_pattern=None,
        transition=None,
    )

    assert frame.version == PROTOCOL_VERSION
    with pytest.raises(FrozenInstanceError):
        frame.step = 5  # type: ignore[misc]


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_field_snapshot_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="global_synchrony must be finite"):
        FieldSnapshot(
            global_synchrony=value,
            mean_local_synchrony=0.5,
            mean_metabolite=0.8,
            min_metabolite=0.6,
            mean_trace=0.2,
            max_trace=0.3,
        )


@pytest.mark.parametrize(
    ("transition", "expected"),
    [
        (PatternTransition("started", None, "coherent"), ("started", None, "coherent")),
        (
            PatternTransition("changed", "coherent", "split"),
            ("changed", "coherent", "split"),
        ),
        (PatternTransition("ended", "split", None), ("ended", "split", None)),
    ],
)
def test_pattern_transition_accepts_valid_shapes(
    transition: PatternTransition,
    expected: tuple[str, str | None, str | None],
) -> None:
    assert (transition.kind, transition.from_label, transition.to_label) == expected


@pytest.mark.parametrize(
    ("kind", "from_label", "to_label"),
    [
        ("started", "coherent", "split"),
        ("started", None, None),
        ("changed", None, "split"),
        ("changed", "split", "split"),
        ("ended", None, None),
        ("ended", "split", "coherent"),
    ],
)
def test_pattern_transition_rejects_invalid_shapes(
    kind: str,
    from_label: str | None,
    to_label: str | None,
) -> None:
    with pytest.raises(ValueError):
        PatternTransition(
            kind=kind,  # type: ignore[arg-type]
            from_label=from_label,
            to_label=to_label,
        )


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_active_pattern_accepts_unit_interval_values(value: float) -> None:
    pattern = ActivePattern(
        label="coherent",
        confidence=value,
        intensity=value,
        novelty=value,
        is_novel=value >= 0.5,
        age_frames=1,
    )

    assert pattern.confidence == value


@given(
    st.floats(allow_nan=False, allow_infinity=False).filter(
        lambda value: value < 0.0 or value > 1.0
    )
)
def test_active_pattern_rejects_values_outside_unit_interval(value: float) -> None:
    with pytest.raises(ValueError):
        ActivePattern(
            label="coherent",
            confidence=value,
            intensity=0.5,
            novelty=0.5,
            is_novel=False,
            age_frames=1,
        )
