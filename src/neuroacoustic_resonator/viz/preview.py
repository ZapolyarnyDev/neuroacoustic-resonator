from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from neuroacoustic_resonator.core.simulation import SimulationFrame


def save_field_preview(frame: SimulationFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panels = (
        ("phase", frame.state.phase, "twilight", None, None),
        ("local synchrony", frame.local_synchrony, "viridis", 0.0, 1.0),
        ("metabolite", frame.state.metabolite, "magma", 0.0, 1.0),
        ("trace", frame.state.trace, "cividis", 0.0, None),
    )

    for axis, (title, values, cmap, vmin, vmax) in zip(axes.flat, panels, strict=True):
        image = axis.imshow(values, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        f"field preview step={frame.metrics.step} "
        f"R={frame.metrics.global_synchrony:.3f} "
        f"M={frame.metrics.mean_metabolite:.3f}"
    )
    figure.savefig(output_path, dpi=140)
    plt.close(figure)

    return output_path


def save_phase_preview(frame: SimulationFrame, path: str | Path) -> Path:
    return save_field_preview(frame, path)
