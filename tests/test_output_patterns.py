from __future__ import annotations

import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, OscillatorField, RegionMasks
from neuroacoustic_resonator.analysis.output_patterns import (
    compare_output_patterns,
    output_pattern_signature,
)


def copied_state(field: OscillatorField):
    state = field.state
    return type(state)(
        phase=state.phase.copy(),
        frequency=state.frequency.copy(),
        metabolite=state.metabolite.copy(),
        coupling=state.coupling.copy(),
        trace=state.trace.copy(),
    )


def test_output_pattern_signature_labels_split_phase_texture() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    state = copied_state(field)
    output_indices = np.argwhere(regions.output)
    for index, (row, column) in enumerate(output_indices):
        state.phase[row, column] = 0.0 if index % 2 == 0 else np.pi
        state.trace[row, column] = 0.8

    signature = output_pattern_signature(state, regions)

    assert signature.label == "split"
    assert signature.confidence > 0.0
    assert signature.features["phase_order_2"] > 0.9


def test_output_pattern_signature_labels_triadic_phase_texture() -> None:
    field = OscillatorField(FieldConfig(size=9, seed=1))
    regions = RegionMasks.from_size(9)
    state = copied_state(field)
    output_indices = np.argwhere(regions.output)
    for index, (row, column) in enumerate(output_indices):
        state.phase[row, column] = (index % 3) * 2.0 * np.pi / 3.0
        state.trace[row, column] = 0.8

    signature = output_pattern_signature(state, regions)

    assert signature.label == "triadic"
    assert signature.features["phase_order_3"] > 0.9


def test_compare_output_patterns_reports_distance_and_label_match() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    first = output_pattern_signature(field.state, regions)
    changed = copied_state(field)
    changed.phase[regions.output] += np.pi
    changed.trace[regions.output] = 1.0
    second = output_pattern_signature(changed, regions)

    comparison = compare_output_patterns(first, second)

    assert comparison["pattern_feature_distance"] > 0.0
    assert comparison["pattern_label_match"] in {0.0, 1.0}


def test_output_pattern_signature_rejects_shape_mismatch() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(5)

    with pytest.raises(ValueError, match="matching shapes"):
        output_pattern_signature(field.state, regions)
