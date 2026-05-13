from __future__ import annotations

import argparse
import importlib
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from neuroacoustic_resonator.analysis.metrics import (
    RegionalActivityMetrics,
    RegionalActivityTracker,
    compute_regional_activity_metrics,
)
from neuroacoustic_resonator.audio.output import (
    ContinuousAudioRenderer,
    EventDrivenAudioRenderer,
    GatedAudioRenderer,
    SlopeTriggeredAudioRenderer,
)
from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.field import FieldState, FloatArray
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation, SimulationFrame


@dataclass(frozen=True)
class LiveVisualizationConfig:
    config_path: Path = Path("configs") / "default.yaml"
    interval_ms: int = 30
    steps_per_update: int = 1
    history_size: int = 512
    audio_enabled: bool = False
    audio_mode: str = "gated"
    audio_sample_rate: int = 48_000
    audio_frame_size: int = 512
    audio_gain: float = 0.2
    audio_carrier_frequency: float = 220.0
    audio_frequency_scale: float = 1.0
    audio_gate_threshold: float = 0.002
    audio_gate_sensitivity: float = 24.0

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
        if self.audio_mode not in {"continuous", "gated", "event", "slope"}:
            msg = "audio_mode must be 'continuous', 'gated', 'event', or 'slope'"
            raise ValueError(msg)
        if self.audio_sample_rate < 1:
            msg = "audio_sample_rate must be positive"
            raise ValueError(msg)
        if self.audio_frame_size < 1:
            msg = "audio_frame_size must be positive"
            raise ValueError(msg)
        if self.audio_gain < 0.0:
            msg = "audio_gain must be non-negative"
            raise ValueError(msg)
        if self.audio_carrier_frequency <= 0.0:
            msg = "audio_carrier_frequency must be positive"
            raise ValueError(msg)
        if self.audio_frequency_scale <= 0.0:
            msg = "audio_frequency_scale must be positive"
            raise ValueError(msg)
        if self.audio_gate_threshold < 0.0:
            msg = "audio_gate_threshold must be non-negative"
            raise ValueError(msg)
        if self.audio_gate_sensitivity <= 0.0:
            msg = "audio_gate_sensitivity must be positive"
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
    regional_metrics: RegionalActivityMetrics


@dataclass(frozen=True)
class DiagnosticCurveSpec:
    key: str
    label: str
    color: tuple[int, int, int]
    width: float = 1.5


def diagnostic_curve_specs() -> tuple[DiagnosticCurveSpec, ...]:
    return (
        DiagnosticCurveSpec(
            key="global_synchrony",
            label="global synchrony",
            color=(250, 220, 70),
        ),
        DiagnosticCurveSpec(
            key="mean_metabolite",
            label="mean metabolite",
            color=(70, 210, 230),
        ),
        DiagnosticCurveSpec(
            key="output_activity",
            label="output activity",
            color=(255, 135, 70),
            width=2.0,
        ),
        DiagnosticCurveSpec(
            key="output_event_score",
            label="output event score",
            color=(185, 135, 255),
        ),
        DiagnosticCurveSpec(
            key="input_value",
            label="input pulse |value|",
            color=(80, 235, 130),
        ),
        DiagnosticCurveSpec(
            key="audio_envelope",
            label="audio envelope",
            color=(245, 245, 245),
        ),
    )


def diagnostic_legend_html(specs: tuple[DiagnosticCurveSpec, ...]) -> str:
    rows = [
        "<div style='font-size: 11pt; font-weight: 600; color: #dddddd;'>diagnostics</div>"
    ]
    for spec in specs:
        red, green, blue = spec.color
        rows.append(
            "<div style='font-size: 10pt; margin-top: 10px;'>"
            f"<span style='color: rgb({red}, {green}, {blue});'>--</span> "
            f"<span style='color: #d8d8d8;'>{spec.label}</span>"
            "</div>"
        )
    return "".join(rows)


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
    *,
    input_value: float = 0.0,
    regional_metrics: RegionalActivityMetrics | None = None,
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
        regional_metrics=regional_metrics
        or compute_regional_activity_metrics(
            frame,
            regions,
            input_value=input_value,
        ),
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
    parser.add_argument(
        "--audio", action="store_true", help="Enable live audio output."
    )
    parser.add_argument(
        "--audio-mode",
        choices=("continuous", "gated", "event", "slope"),
        default="gated",
        help="Audio monitor mode used when --audio is enabled.",
    )
    parser.add_argument("--audio-sample-rate", type=int, default=48_000)
    parser.add_argument("--audio-frame-size", type=int, default=512)
    parser.add_argument("--audio-gain", type=float, default=0.2)
    parser.add_argument("--audio-carrier-frequency", type=float, default=220.0)
    parser.add_argument("--audio-frequency-scale", type=float, default=1.0)
    parser.add_argument("--audio-gate-threshold", type=float, default=0.002)
    parser.add_argument("--audio-gate-sensitivity", type=float, default=24.0)
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
        self._diagnostic_specs = diagnostic_curve_specs()
        self._diagnostic_values: dict[str, deque[float]] = {
            spec.key: deque(maxlen=config.history_size)
            for spec in self._diagnostic_specs
        }
        self._diagnostic_curves: dict[str, Any] = {}
        self._diagnostics_legend_label: Any | None = None
        self._regional_tracker = RegionalActivityTracker()

        self._window = pg.GraphicsLayoutWidget(title="Neuroacoustic Resonator")
        self._window.setWindowTitle("Neuroacoustic Resonator - live field diagnostics")
        self._window.setBackground((5, 5, 5))
        self._window.resize(1680, 900)
        self._configure_layout_columns()
        self._phase_item = pg.ImageItem()
        self._synchrony_item = pg.ImageItem()
        self._metabolite_item = pg.ImageItem()
        self._trace_item = pg.ImageItem()
        self._regions_item = pg.ImageItem()
        self._boundary_columns = region_boundary_columns(regions)
        self._audio_output = (
            _LiveAudioOutput(config=config, regions=regions)
            if config.audio_enabled
            else None
        )

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
            row=0,
            col=3,
            rowspan=2,
            title="diagnostics",
        )
        self._configure_diagnostics_plot()
        self._timer = qt_core.QTimer(self._window)
        self._timer.timeout.connect(self.update)

    def _configure_layout_columns(self) -> None:
        layout = self._window.ci.layout
        for column in range(3):
            layout.setColumnMinimumWidth(column, 360)
            layout.setColumnStretchFactor(column, 1)
        layout.setColumnMinimumWidth(3, 520)
        layout.setColumnStretchFactor(3, 2)
        layout.setColumnMinimumWidth(4, 190)
        layout.setColumnStretchFactor(4, 0)

    def _add_image_panel(self, title: str, item: Any, *, row: int, col: int) -> None:
        plot = self._window.addPlot(row=row, col=col, title=title)
        plot.setAspectLocked(True)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
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

    def _configure_diagnostics_plot(self) -> None:
        self._metrics_plot.setMenuEnabled(False)
        self._metrics_plot.showGrid(x=True, y=True, alpha=0.25)
        self._metrics_plot.setLabel("bottom", "step")
        self._metrics_plot.setLabel("left", "value")
        self._metrics_plot.getViewBox().setDefaultPadding(0.02)
        self._diagnostics_legend_label = self._pg.LabelItem(justify="left")
        self._diagnostics_legend_label.setText(
            diagnostic_legend_html(self._diagnostic_specs),
        )
        self._window.addItem(
            self._diagnostics_legend_label,
            row=0,
            col=4,
            rowspan=2,
        )
        for spec in self._diagnostic_specs:
            curve = self._metrics_plot.plot(
                pen=self._pg.mkPen(spec.color, width=spec.width),
            )
            self._diagnostic_curves[spec.key] = curve

    def show(self) -> None:
        self._window.show()
        if self._audio_output is not None:
            self._audio_output.start()
        self._qt_core.QTimer.singleShot(0, self.update)
        self._timer.start(self._config.interval_ms)

    def update(self) -> None:
        try:
            frame = self._simulation.snapshot()
            for _ in range(self._config.steps_per_update):
                frame = self._simulation.step()
            regional_metrics = self._regional_tracker.update(
                frame,
                self._regions,
                input_value=self._simulation.last_input_value,
            )
            view_frame = frame_to_visualization(
                frame,
                self._regions,
                regional_metrics=regional_metrics,
            )

            self._phase_item.setImage(view_frame.phase.T, autoLevels=True)
            self._synchrony_item.setImage(
                view_frame.local_synchrony.T,
                levels=(0.0, 1.0),
            )
            self._metabolite_item.setImage(view_frame.metabolite.T, levels=(0.0, 1.0))
            self._trace_item.setImage(view_frame.trace.T, autoLevels=True)
            self._regions_item.setImage(view_frame.region_labels.T, levels=(1, 3))

            self._steps.append(view_frame.step)
            if self._audio_output is not None:
                self._audio_output.update_state(frame.state)
            audio_envelope = (
                self._audio_output.envelope if self._audio_output is not None else 0.0
            )
            self._append_diagnostics(view_frame, audio_envelope)
        except KeyboardInterrupt:
            self._timer.stop()
            if self._audio_output is not None:
                self._audio_output.stop()
            self._app.quit()
        except Exception:
            self._timer.stop()
            if self._audio_output is not None:
                self._audio_output.stop()
            traceback.print_exc()
            self._app.quit()

    def _append_diagnostics(
        self,
        view_frame: VisualizationFrame,
        audio_envelope: float,
    ) -> None:
        values = {
            "global_synchrony": view_frame.global_synchrony,
            "mean_metabolite": view_frame.mean_metabolite,
            "output_activity": view_frame.regional_metrics.output_activity,
            "output_event_score": view_frame.regional_metrics.output_event_score,
            "input_value": abs(view_frame.regional_metrics.input_value),
            "audio_envelope": audio_envelope,
        }
        for key, value in values.items():
            self._diagnostic_values[key].append(value)
            self._diagnostic_curves[key].setData(
                list(self._steps),
                list(self._diagnostic_values[key]),
            )


class _LiveAudioOutput:
    def __init__(
        self,
        *,
        config: LiveVisualizationConfig,
        regions: RegionMasks,
        stream_factory: Any | None = None,
    ) -> None:
        self._config = config
        self._regions = regions
        self._lock = threading.Lock()
        self._state: FieldState | None = None
        self._stream: Any | None = None
        self._stream_factory = stream_factory
        self._renderer = self._build_renderer(config.audio_frame_size)
        self._last_status_text: str | None = None

    def _build_renderer(
        self, frame_size: int
    ) -> (
        ContinuousAudioRenderer
        | GatedAudioRenderer
        | EventDrivenAudioRenderer
        | SlopeTriggeredAudioRenderer
    ):
        if self._config.audio_mode == "gated":
            return GatedAudioRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
                gate_threshold=self._config.audio_gate_threshold,
                gate_sensitivity=self._config.audio_gate_sensitivity,
            )
        if self._config.audio_mode == "event":
            return EventDrivenAudioRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
            )
        if self._config.audio_mode == "slope":
            return SlopeTriggeredAudioRenderer(
                sample_rate=self._config.audio_sample_rate,
                frame_size=frame_size,
                carrier_frequency=self._config.audio_carrier_frequency,
                frequency_scale=self._config.audio_frequency_scale,
                gain=self._config.audio_gain,
            )
        return ContinuousAudioRenderer(
            sample_rate=self._config.audio_sample_rate,
            frame_size=frame_size,
            carrier_frequency=self._config.audio_carrier_frequency,
            frequency_scale=self._config.audio_frequency_scale,
            gain=self._config.audio_gain,
        )

    def start(self) -> None:
        if self._stream is not None:
            return
        factory = self._stream_factory
        if factory is None:
            sounddevice = cast(Any, importlib.import_module("sounddevice"))
            factory = sounddevice.OutputStream
        self._stream = factory(
            samplerate=self._config.audio_sample_rate,
            blocksize=self._config.audio_frame_size,
            channels=1,
            dtype="float32",
            callback=self.callback,
        )
        self._stream.__enter__()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.__exit__(None, None, None)
        self._stream = None

    def update_state(self, state: FieldState) -> None:
        with self._lock:
            self._state = state

    @property
    def envelope(self) -> float:
        return float(getattr(self._renderer, "envelope", 0.0))

    def callback(
        self,
        outdata: NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del time_info
        if status:
            self._report_status(status)

        with self._lock:
            state = self._state
        if state is None:
            outdata[:, 0] = 0.0
            return

        if frames != self._renderer.frame_size:
            self._renderer = self._build_renderer(frames)
        audio = self._renderer.render_frame(state, self._regions)
        outdata[:, 0] = np.asarray(audio, dtype=np.float32)

    def _report_status(self, status: Any) -> None:
        status_text = str(status)
        if status_text == self._last_status_text:
            return
        self._last_status_text = status_text
        print(f"audio status: {status_text}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_live_visualizer(
        LiveVisualizationConfig(
            config_path=args.config,
            interval_ms=args.interval_ms,
            steps_per_update=args.steps_per_update,
            history_size=args.history_size,
            audio_enabled=args.audio,
            audio_mode=args.audio_mode,
            audio_sample_rate=args.audio_sample_rate,
            audio_frame_size=args.audio_frame_size,
            audio_gain=args.audio_gain,
            audio_carrier_frequency=args.audio_carrier_frequency,
            audio_frequency_scale=args.audio_frequency_scale,
            audio_gate_threshold=args.audio_gate_threshold,
            audio_gate_sensitivity=args.audio_gate_sensitivity,
        )
    )
