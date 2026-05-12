from __future__ import annotations

import argparse
import importlib
import traceback
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


def region_boundary_columns(regions: RegionMasks) -> tuple[int, int]:
    labels = regions.labels()
    first_output_column = int(np.min(np.flatnonzero(np.any(regions.output, axis=0))))
    first_assoc_column = int(np.min(np.flatnonzero(np.any(regions.assoc, axis=0))))
    if not np.any(labels):
        msg = "regions must not be empty"
        raise ValueError(msg)
    return first_assoc_column, first_output_column


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
    visualizer = _LiveFieldWindow(
        app=app,
        qt_core=QtCore,
        pg=pg,
        simulation=simulation,
        regions=regions,
        config=config,
    )
    visualizer.show()
    return int(app.exec())


class _LiveFieldWindow:
    def __init__(
        self,
        *,
        app: Any,
        qt_core: Any,
        pg: Any,
        simulation: Simulation,
        regions: RegionMasks,
        config: LiveVisualizationConfig,
    ) -> None:
        self._app = app
        self._qt_core = qt_core
        self._pg = pg
        self._simulation = simulation
        self._regions = regions
        self._config = config
        self._steps: deque[int] = deque(maxlen=config.history_size)
        self._synchrony_values: deque[float] = deque(maxlen=config.history_size)
        self._metabolite_values: deque[float] = deque(maxlen=config.history_size)

        self._window = pg.GraphicsLayoutWidget(title="Neuroacoustic Resonator")
        self._window.resize(1200, 800)
        self._phase_item = pg.ImageItem()
        self._synchrony_item = pg.ImageItem()
        self._metabolite_item = pg.ImageItem()
        self._trace_item = pg.ImageItem()
        self._regions_item = pg.ImageItem()
        self._boundary_columns = region_boundary_columns(regions)

        for index, (title, item) in enumerate(
            (
                ("phase", self._phase_item),
                ("local synchrony", self._synchrony_item),
                ("metabolite", self._metabolite_item),
                ("trace", self._trace_item),
                ("regions", self._regions_item),
            )
        ):
            self._add_image_panel(title, item, row=index // 3, col=index % 3)

        self._metrics_plot = self._window.addPlot(
            row=1,
            col=2,
            title="global synchrony / mean M",
        )
        self._synchrony_curve = self._metrics_plot.plot(pen="y")
        self._metabolite_curve = self._metrics_plot.plot(pen="c")
        self._timer = qt_core.QTimer(self._window)
        self._timer.timeout.connect(self.update)

    def _add_image_panel(self, title: str, item: Any, *, row: int, col: int) -> None:
        plot = self._window.addPlot(row=row, col=col, title=title)
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.addItem(item)
        for boundary in self._boundary_columns:
            line = self._pg.InfiniteLine(
                pos=boundary - 0.5,
                angle=90,
                pen=self._pg.mkPen((255, 80, 80), width=1),
            )
            plot.addItem(line)

    def show(self) -> None:
        self._window.show()
        self._qt_core.QTimer.singleShot(0, self.update)
        self._timer.start(self._config.interval_ms)

    def update(self) -> None:
        try:
            frame = self._simulation.snapshot()
            for _ in range(self._config.steps_per_update):
                frame = self._simulation.step()
            view_frame = frame_to_visualization(frame, self._regions)

            self._phase_item.setImage(view_frame.phase.T, autoLevels=True)
            self._synchrony_item.setImage(
                view_frame.local_synchrony.T,
                levels=(0.0, 1.0),
            )
            self._metabolite_item.setImage(view_frame.metabolite.T, levels=(0.0, 1.0))
            self._trace_item.setImage(view_frame.trace.T, autoLevels=True)
            self._regions_item.setImage(view_frame.region_labels.T, levels=(1, 3))

            self._steps.append(view_frame.step)
            self._synchrony_values.append(view_frame.global_synchrony)
            self._metabolite_values.append(view_frame.mean_metabolite)
            self._synchrony_curve.setData(
                list(self._steps),
                list(self._synchrony_values),
            )
            self._metabolite_curve.setData(
                list(self._steps),
                list(self._metabolite_values),
            )
        except Exception:
            self._timer.stop()
            traceback.print_exc()
            self._app.quit()


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
