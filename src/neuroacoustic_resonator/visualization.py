from __future__ import annotations

import argparse
import importlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from neuroacoustic_resonator.config import SimulationConfig
from neuroacoustic_resonator.field import FloatArray
from neuroacoustic_resonator.regions import RegionMasks
from neuroacoustic_resonator.simulation import Simulation, SimulationFrame


@dataclass(frozen=True)
class LiveVisualizationConfig:
    config_path: Path = Path("configs") / "default.yaml"
    interval_ms: int = 30
    steps_per_update: int = 1
    history_size: int = 512

    def __post_init__(self) -> None:
        if self.interval_ms < 1:
            msg = "interval_ms must be positive"
            raise ValueError(msg)
        if self.steps_per_update < 1:
            msg = "steps_per_update must be positive"
            raise ValueError(msg)
        if self.history_size < 2:
            msg = "history_size must be at least 2"
            raise ValueError(msg)


@dataclass(frozen=True)
class VisualizationFrame:
    phase: FloatArray
    local_synchrony: FloatArray
    metabolite: FloatArray
    trace: FloatArray
    region_labels: np.ndarray
    step: int
    global_synchrony: float
    mean_metabolite: float


def frame_to_visualization(
    frame: SimulationFrame,
    regions: RegionMasks,
) -> VisualizationFrame:
    if frame.state.phase.shape != regions.shape:
        msg = "frame and regions must have matching shapes"
        raise ValueError(msg)

    return VisualizationFrame(
        phase=frame.state.phase,
        local_synchrony=frame.local_synchrony,
        metabolite=frame.state.metabolite,
        trace=frame.state.trace,
        region_labels=regions.labels(),
        step=frame.metrics.step,
        global_synchrony=frame.metrics.global_synchrony,
        mean_metabolite=frame.metrics.mean_metabolite,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show a live pyqtgraph view of the oscillator field.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "default.yaml",
        help="YAML simulation config path.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=30,
        help="GUI update interval in milliseconds.",
    )
    parser.add_argument(
        "--steps-per-update",
        type=int,
        default=1,
        help="Simulation steps advanced per GUI update.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=512,
        help="Number of metric points retained in the live plots.",
    )
    return parser


def run_live_visualizer(config: LiveVisualizationConfig) -> int:
    from PySide6 import QtCore, QtWidgets

    pg = cast(Any, importlib.import_module("pyqtgraph"))

    simulation_config = SimulationConfig.from_file(config.config_path)
    simulation = Simulation.from_config(simulation_config)
    regions = RegionMasks.from_size(simulation_config.field.size)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = pg.GraphicsLayoutWidget(title="Neuroacoustic Resonator")
    window.resize(1200, 800)

    phase_item = pg.ImageItem()
    synchrony_item = pg.ImageItem()
    metabolite_item = pg.ImageItem()
    trace_item = pg.ImageItem()
    regions_item = pg.ImageItem()

    panels = (
        ("phase", phase_item),
        ("local synchrony", synchrony_item),
        ("metabolite", metabolite_item),
        ("trace", trace_item),
        ("regions", regions_item),
    )
    for index, (title, item) in enumerate(panels):
        plot = window.addPlot(row=index // 3, col=index % 3, title=title)
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.addItem(item)

    metrics_plot = window.addPlot(row=1, col=2, title="global synchrony / mean M")
    synchrony_curve = metrics_plot.plot(pen="y")
    metabolite_curve = metrics_plot.plot(pen="c")
    steps: deque[int] = deque(maxlen=config.history_size)
    synchrony_values: deque[float] = deque(maxlen=config.history_size)
    metabolite_values: deque[float] = deque(maxlen=config.history_size)

    def update() -> None:
        frame = simulation.snapshot()
        for _ in range(config.steps_per_update):
            frame = simulation.step()
        view_frame = frame_to_visualization(frame, regions)

        phase_item.setImage(view_frame.phase.T, autoLevels=False)
        synchrony_item.setImage(view_frame.local_synchrony.T, levels=(0.0, 1.0))
        metabolite_item.setImage(view_frame.metabolite.T, levels=(0.0, 1.0))
        trace_item.setImage(view_frame.trace.T)
        regions_item.setImage(view_frame.region_labels.T, autoLevels=False)

        steps.append(view_frame.step)
        synchrony_values.append(view_frame.global_synchrony)
        metabolite_values.append(view_frame.mean_metabolite)
        synchrony_curve.setData(list(steps), list(synchrony_values))
        metabolite_curve.setData(list(steps), list(metabolite_values))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(config.interval_ms)
    update()
    window.show()
    return int(app.exec())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_live_visualizer(
        LiveVisualizationConfig(
            config_path=args.config,
            interval_ms=args.interval_ms,
            steps_per_update=args.steps_per_update,
            history_size=args.history_size,
        )
    )
