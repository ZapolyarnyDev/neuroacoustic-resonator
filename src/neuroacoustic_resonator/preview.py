from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from neuroacoustic_resonator.simulation import SimulationFrame


def save_phase_preview(frame: SimulationFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image = axis.imshow(frame.state.phase, cmap="twilight", origin="lower")
    axis.set_title(
        f"phase field step={frame.metrics.step} R={frame.metrics.global_synchrony:.3f}"
    )
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="phase")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)

    return output_path
