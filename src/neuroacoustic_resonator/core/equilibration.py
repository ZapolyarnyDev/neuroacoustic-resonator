from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuroacoustic_resonator.core.simulation import Simulation


@dataclass(frozen=True)
class FieldEquilibrationConfig:
    neutral_steps: int = 512
    phase_damping: float = 0.2
    metabolite_baseline: float | None = 1.0

    def __post_init__(self) -> None:
        if self.neutral_steps < 0:
            msg = "neutral_steps must be non-negative"
            raise ValueError(msg)
        if not 0.0 <= self.phase_damping <= 1.0:
            msg = "phase_damping must be between 0 and 1"
            raise ValueError(msg)
        if self.metabolite_baseline is not None and not (
            0.0 <= self.metabolite_baseline <= 1.0
        ):
            msg = "metabolite_baseline must be between 0 and 1"
            raise ValueError(msg)


def equilibrate_simulation(
    simulation: Simulation,
    config: FieldEquilibrationConfig,
) -> dict[str, Any]:
    before = simulation.field.metrics(step=simulation.step_index)
    for _ in range(config.neutral_steps):
        simulation.step_with_input(0.0)
    simulation.field.damp_initial_phases(config.phase_damping)
    if config.metabolite_baseline is not None:
        simulation.field.set_metabolite_baseline(config.metabolite_baseline)
    after = simulation.field.metrics(step=simulation.step_index)
    return {
        "neutral_steps": config.neutral_steps,
        "phase_damping": config.phase_damping,
        "metabolite_baseline": config.metabolite_baseline,
        "start_step": before.step,
        "end_step": after.step,
        "start_global_synchrony": before.global_synchrony,
        "end_global_synchrony": after.global_synchrony,
        "start_mean_metabolite": before.mean_metabolite,
        "end_mean_metabolite": after.mean_metabolite,
    }
