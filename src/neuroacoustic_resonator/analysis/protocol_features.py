from __future__ import annotations

import numpy as np

from neuroacoustic_resonator.core.field import FieldState
from neuroacoustic_resonator.core.regions import BoolArray
from neuroacoustic_resonator.core.simulation import SimulationFrame
from neuroacoustic_resonator.protocol import (
    FieldSnapshot,
    PatternSnapshot,
    RegionName,
    RegionSnapshot,
)


def field_snapshot(frame: SimulationFrame) -> FieldSnapshot:
    return FieldSnapshot(
        global_synchrony=frame.metrics.global_synchrony,
        mean_local_synchrony=frame.metrics.mean_local_synchrony,
        mean_metabolite=frame.metrics.mean_metabolite,
        min_metabolite=frame.metrics.min_metabolite,
        mean_trace=frame.metrics.mean_trace,
        max_trace=frame.metrics.max_trace,
    )


def region_snapshot(
    frame: SimulationFrame,
    mask: BoolArray,
    name: RegionName,
) -> RegionSnapshot:
    require_matching_shape(frame, mask)
    phase = frame.state.phase[mask]
    metabolite = frame.state.metabolite[mask]
    trace = frame.state.trace[mask]
    frequency = frame.state.frequency[mask]
    coupling = frame.state.coupling[mask]
    return RegionSnapshot(
        name=name,
        phase_coherence=phase_order(phase, 1),
        mean_local_synchrony=float(np.mean(frame.local_synchrony[mask])),
        mean_metabolite=float(np.mean(metabolite)),
        min_metabolite=float(np.min(metabolite)),
        mean_trace=float(np.mean(trace)),
        max_trace=float(np.max(trace)),
        mean_frequency=float(np.mean(frequency)),
        frequency_spread=float(np.std(frequency)),
        mean_coupling=float(np.mean(coupling)),
        coupling_spread=float(np.std(coupling)),
    )


def pattern_snapshot(state: FieldState, mask: BoolArray) -> PatternSnapshot:
    if state.phase.shape != mask.shape:
        msg = "state and pattern mask must have matching shapes"
        raise ValueError(msg)
    if not np.any(mask):
        msg = "pattern mask must not be empty"
        raise ValueError(msg)

    phase = state.phase[mask]
    trace = state.trace[mask]
    metabolite_stress = 1.0 - state.metabolite[mask]
    frequency = state.frequency[mask]
    return PatternSnapshot(
        phase_order_1=phase_order(phase, 1),
        phase_order_2=phase_order(phase, 2),
        phase_order_3=phase_order(phase, 3),
        trace_mean=float(np.mean(trace)),
        trace_contrast=float(np.std(trace)),
        metabolite_stress=float(np.mean(metabolite_stress)),
        metabolite_contrast=float(np.std(metabolite_stress)),
        trace_phase_lock=weighted_phase_lock(phase, trace),
        metabolite_phase_lock=weighted_phase_lock(phase, metabolite_stress),
        frequency_mean=float(np.mean(frequency)),
        frequency_spread=float(np.std(frequency)),
    )


def phase_order(phase: np.ndarray, harmonic: int) -> float:
    if harmonic < 1:
        msg = "harmonic must be positive"
        raise ValueError(msg)
    return float(np.abs(np.mean(np.exp(1j * harmonic * phase))))


def weighted_phase_lock(phase: np.ndarray, weights: np.ndarray) -> float:
    if phase.shape != weights.shape:
        msg = "phase and weights must have matching shapes"
        raise ValueError(msg)
    clipped = np.clip(weights, 0.0, None)
    weight_sum = float(np.sum(clipped))
    if weight_sum <= 1e-12:
        return 0.0
    return float(np.abs(np.sum(clipped * np.exp(1j * phase)) / weight_sum))


def require_matching_shape(frame: SimulationFrame, mask: BoolArray) -> None:
    if frame.state.phase.shape != mask.shape:
        msg = "frame and region mask must have matching shapes"
        raise ValueError(msg)
    if frame.local_synchrony.shape != mask.shape:
        msg = "local synchrony and region mask must have matching shapes"
        raise ValueError(msg)
    if not np.any(mask):
        msg = "region mask must not be empty"
        raise ValueError(msg)
